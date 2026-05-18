"""Analytic marginalization of timing-model parameters.

The marginalization is performed analytically via the Woodbury identity.
For a parameter ``y_i`` in ``over`` with an ``ImproperPrior`` (the only
prior shape currently accepted), the integral is exact when residuals 
are linear in ``y_i`` and is augmented onto the
existing noise covariance as a low-rank Woodbury update with
``Φ = 1e40`` (discovery-equivalent).  The marginalized parameter
disappears from the returned callable's signature.

This module provides:

- :func:`marg_set_from_priors` — derive an ``over`` set from a prior dict.
- :func:`marginalize` — wrap a likelihood to integrate out the given
  parameters; returns ``(callable, sampled_priors, reduced_skeleton)``.

"""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
)

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.bayes.priors import ImproperPrior, Prior
from jaxpint.bayes.validate import PriorValidationError
from jaxpint.likelihood import single_pulsar_logL
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import ParameterVector, TOAData
from jaxpint.utils import concat_woodbury_blocks

if TYPE_CHECKING:
    pass


__all__ = [
    "marginalize",
    "marg_set_from_priors",
]


# ---------------------------------------------------------------------------
# Helper: derive a marginalization set from a prior dict
# ---------------------------------------------------------------------------


def marg_set_from_priors(
    priors: Mapping[str, Prior],
    *,
    prior_class: type = ImproperPrior,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> set[str]:
    """Compute a marginalization set from the assigned prior shapes.

    The prior dict already carries the information about how each parameter
    should be treated (the convention: ``ImproperPrior`` → marginalize;
    ``Gaussian`` → keep explicit;
    ``Uniform`` → sample).  This helper exposes that convention as a
    one-line filter.

    Parameters
    ----------
    priors
        Mapping of parameter name → :class:`Prior` instance.
    prior_class
        Only names whose assigned prior is an instance of this class enter
        the default set.  Defaults to :class:`ImproperPrior`, matching the
        NanoGrav convention "marg the timing params, sample the rest".
    include
        Names to add to the default set regardless of their prior shape.
        Use this to force a parameter into ``over`` (e.g. a Gaussian-prior'd
        PX in a phase that supports Gaussian marginalization).
    exclude
        Names to remove from the default set regardless of their prior
        shape.  Use this to keep a specific parameter sampled.

    Returns
    -------
    set of str
        Names selected for analytic marginalization.

    Examples
    --------
    >>> # NanoGrav default: marg every ImproperPrior'd param
    >>> over = marg_set_from_priors(priors)

    >>> # Same, but also include PX (even though it has a Gaussian prior)
    >>> over = marg_set_from_priors(priors, include={"J1614-2230_PX"})

    >>> # Default but keep one timing param sampled (e.g., nonlinear binary)
    >>> over = marg_set_from_priors(priors, exclude={"J1614-2230_PB"})
    """
    base = {n for n, p in priors.items() if isinstance(p, prior_class)}
    return (base | set(include)) - set(exclude)


# ---------------------------------------------------------------------------
# Linearity check
# ---------------------------------------------------------------------------


def _check_linearity(
    timing_model: TimingModel,
    toa_data: TOAData,
    fiducial_params: ParameterVector,
    *,
    over_indices: tuple[int, ...],
    over_names: tuple[str, ...],
    trust_radii: Float[Array, " n_marg"],
    M_full_cols: Float[Array, "n_toas n_params"],
    tol: float,
) -> list[tuple[str, float]]:
    """Heuristic to flag parameters whose residuals are nonlinear in y at y_fid.

    For each marginalized param i, computes the ratio
        max_t |H[t, i, i] * sigma_i| / max_t |M[t, i]|
    where H is the Hessian of residuals w.r.t. the full param vector at
    y_fid, M is the corresponding Jacobian column, and sigma_i is a
    physically-motivated trust radius for that parameter.

    A ratio above ``tol`` flags the linearization as approximate: stepping
    one trust-radius away from y_fid changes the local slope of the residual
    function appreciably, so r(y) is NOT well-approximated by its
    linearization within the range the prior gives non-negligible weight to.

    Returns a list of (name, ratio) pairs for parameters that exceed ``tol``.

    Notes
    -----
    This is the most expensive step in marginalize() setup (~O(n_params) more
    expensive than the Jacobian alone).  It is skipped when the user sets
    ``validate_linearity=False`` on :func:`marginalize`.
    """

    from jaxpint.fitters._base import compute_time_residuals

    def time_resid_fn(values):
        p = eqx.tree_at(lambda pv: pv.values, fiducial_params, values)
        return compute_time_residuals(timing_model, toa_data, p)

    # Hessian is shape (n_toas, n_params, n_params).  We only need the
    # diagonal block at the marg'd indices
    H_full = jax.jacfwd(jax.jacrev(time_resid_fn))(fiducial_params.values)

    flagged: list[tuple[str, float]] = []
    for k, idx in enumerate(over_indices):
        H_ii = H_full[:, idx, idx]
        M_col = M_full_cols[:, k]
        denom = jnp.max(jnp.abs(M_col))
        # Guard against an all-zero design column (param does not affect
        # residuals at all — degenerate, but we should not propagate NaN).
        if float(denom) == 0.0:
            ratio = float("inf")
        else:
            ratio = float(jnp.max(jnp.abs(H_ii * trust_radii[k])) / denom)
        if ratio > tol:
            flagged.append((over_names[k], ratio))
    return flagged


def _trust_radius_for_prior(
    prior: Prior,
    idx: int,
    M_full: Float[Array, "n_toas n_params"],
    Ndiag: Float[Array, " n_toas"],
) -> float:
    """Pick the physical trust radius for a marg'd parameter.

    - Gaussian prior: prior σ (the radius the prior gives weight to).
    - ImproperPrior:  WLS posterior σ from the diagonal of (Mᵀ N⁻¹ M)⁻¹
                      (the data-driven scale; the only physical scale
                       available when no prior is assigned).
    """
    if not isinstance(prior, ImproperPrior):
        from jaxpint.bayes.priors import Gaussian
        # Defaulting to Gaussian for now. TODO
        if isinstance(prior, Gaussian):
            return float(prior.sigma)
    # ImproperPrior path: use the WLS posterior sigma for this column.
    Ninv = 1.0 / Ndiag
    fisher_diag = float(jnp.sum(M_full[:, idx] ** 2 * Ninv))
    if fisher_diag <= 0.0:
        return float("inf")  # degenerate — let the check fire loudly
    return 1.0 / jnp.sqrt(fisher_diag)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def marginalize(
    likelihood: Callable,
    *,
    over: Iterable[str],
    priors: Mapping[str, Prior],
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    fiducial_params: ParameterVector,
    allow_nonlinear: bool = False,
    validate_linearity: bool = True,
    laplace_error_tol: float = 1e-6,
) -> tuple[Callable, dict[str, Prior], ParameterVector]:
    """Build an analytically-marginalized single-pulsar log-likelihood.

    Returns ``(g, sampled_priors, reduced_skeleton)`` where:

    - ``g(reduced_params, *, external_delay=None, external_cov=None)`` is the
      marginalized log-likelihood callable.  The parameters in ``over`` no
      longer appear in its parameter dict — they have been integrated out
      analytically against their assigned priors.
    - ``sampled_priors`` is ``priors`` with the marg'd entries removed.
      Pass this (not the full ``priors`` dict) to
      :func:`jaxpint.bayes.combine_log_prob` to avoid double-counting.
    - ``reduced_skeleton`` is ``fiducial_params.with_marginalized(over)`` —
      a :class:`ParameterVector` whose ``marginalized_mask`` is True for
      every name in ``over``.  Use it in the sampling loop as the skeleton
      for ``with_free_values()`` so that ``.free_values()`` returns the
      kept-entry slice that ``g`` expects.

    currently only accepts only :class:`ImproperPrior` in ``over``; Gaussian
     and Uniform  raise    ``NotImplementedError`` with a clear message.

    Parameters
    ----------
    likelihood
        Currently must be :func:`jaxpint.likelihood.single_pulsar_logL`
        (or a callable with the same signature).  Multi-pulsar
        marginalization is deferred.
    over
        Fully-qualified names of parameters to marginalize out.  Must be a
        subset of ``fiducial_params.names``.
    priors
        Mapping from parameter name to :class:`Prior` instance.  Must
        contain an entry for every name in ``over``.  Extras are ignored
        (but propagated unchanged to the returned ``sampled_priors`` if
        their name is not in ``over``).
    toa_data, timing_model, noise_model
        Static likelihood arguments, captured into the returned closure.
    fiducial_params
        The linearization point y_fid.  The design matrix
        ``M = ∂r/∂y|_{y_fid}`` and (optional) Hessian are computed once
        here and cached.  For the analytic Woodbury integral to be exact,
        residuals must be linear in each marg'd parameter; the linearity
        check (``validate_linearity=True``, the default) enforces this.
    allow_nonlinear
        If ``False`` (default), marg'd parameters that fail the
        Laplace-error check raise ``NotImplementedError`` at setup.  If
        ``True``, proceed with the Laplace approximation around ``y_fid``
        and emit a warning naming the affected parameters and their error
        ratios.
    validate_linearity
        If ``True`` (default), compute the Hessian at setup and run the
        linearity check.  If ``False``, skip the Hessian computation
        entirely (the user takes responsibility for linearity).
        ``allow_nonlinear`` becomes irrelevant in that case.
    laplace_error_tol
        Threshold for the relative Laplace-approximation error within the
        prior support:
        ``max_t |H[t, i, i] * sigma_i| / max_t |M[t, i]|``,
        where ``sigma_i`` is the prior σ (Gaussian) or the data-driven WLS
        posterior σ (Improper).  The ratio quantifies how much the residual's
        slope-w.r.t.-y changes over one prior-σ step away from y_fid,
        relative to the slope itself — equivalently, the relative size of
        the quadratic Taylor term that the linearization drops, evaluated
        on the scale the prior gives non-negligible weight to.  Above this
        threshold, the Woodbury integral departs measurably from the true
        marginal.  For a perfectly linear parameter (``H = 0``), the ratio
        is zero regardless of σ.  Defaults to ``1e-6``.

    Returns
    -------
    g : callable
        Marginalized log-likelihood.  Takes a :class:`ParameterVector`
        whose ``marginalized_mask`` matches ``reduced_skeleton`` and
        returns a scalar.  Optional ``external_delay`` and ``external_cov``
        kwargs pass through to :func:`single_pulsar_logL` for composition
        with signal injectors.
    sampled_priors : dict[str, Prior]
        ``priors`` with marg'd entries removed.
    reduced_skeleton : ParameterVector
        Same structure as ``fiducial_params`` but with
        ``marginalized_mask=True`` for every name in ``over``.

    Raises
    ------
    PriorValidationError
        If any name in ``over`` is missing from ``priors``.
    NotImplementedError
        If ``likelihood`` is not :func:`single_pulsar_logL` (phase-2 will
        add :func:`pta_logL` dispatch); if ``over`` contains a parameter
        whose assigned prior is not an :class:`ImproperPrior`; or if it
        contains a parameter that fails the linearity check when
        ``allow_nonlinear=False``.
    """
    # --- 0. Likelihood-dispatch check (phase 1: single_pulsar_logL only) -
    # For now, anything other than single_pulsar_logL is
    # rejected loudly so the contract matches the documentation.
    if likelihood is not single_pulsar_logL:
        raise NotImplementedError(
            "marginalize() phase 1 supports only single_pulsar_logL as the "
            f"`likelihood` argument; got {getattr(likelihood, '__name__', repr(likelihood))!r}. "
            "Multi-pulsar marginalization (pta_logL) is planned for phase 2."
        )

    over_set = set(over)
    over_list = sorted(over_set)  # deterministic ordering for caching

    # --- 1. Completeness check ----------------------------------------
    missing = over_set - set(priors.keys())
    if missing:
        raise PriorValidationError(
            f"marginalize: priors dict is missing entries for {sorted(missing)} "
            f"(found in `over` but not in `priors`)."
        )

    # --- 2. Prior-shape check (TEMP. Improper only)  ----------------
    bad = [
        (n, type(priors[n]).__name__)
        for n in over_list
        if not isinstance(priors[n], ImproperPrior)
    ]
    if bad:
        raise NotImplementedError(
            "marginalize() phase 1 supports only ImproperPrior in `over`. "
            f"Got non-Improper priors for: {bad}. "
            "Keep these parameters sampled (combine_log_prob will apply their "
            "priors), or assign ImproperPrior() if you want analytic "
            "marginalization with no prior information."
        )

    # --- 3. Validate over_set fiducial_params.names --------------------
    unknown = over_set - set(fiducial_params.names)
    if unknown:
        raise ValueError(
            f"marginalize: parameter name(s) {sorted(unknown)} in `over` "
            f"are not present in fiducial_params.names."
        )

    # --- 4. Design matrix at y_fid  ---------------
    # NOTE: compute_design_matrix returns shape (n_toas, n_free [+ 1 offset]).
    # We need columns indexed by name in over, not by free-parameter position.
    # So we build the FULL Jacobian here via the same internal trick.
    from jaxpint.fitters._base import compute_time_residuals

    over_indices = tuple(fiducial_params.param_index(n) for n in over_list)

    if len(over_indices) == 0:
        # Empty marg set: skip Jacobian entirely. The wrapper passes None
        # through concat_woodbury_blocks so the user's external_cov flows
        # through unchanged (no spurious empty column appended).
        J_full = None
        M_marg = None
        marg_cov_cached = None
    else:
        def _resid_fn(values):
            p = eqx.tree_at(lambda pv: pv.values, fiducial_params, values)
            return compute_time_residuals(timing_model, toa_data, p)

        J_full = jax.jacobian(_resid_fn)(fiducial_params.values)
        marg_idx_array = jnp.asarray(over_indices, dtype=jnp.int32)
        M_marg = -J_full[:, marg_idx_array]   # (n_toas, n_marg)
        Phi_marg = jnp.full(len(over_indices), 1e40, dtype=jnp.float64)
        marg_cov_cached = (M_marg, Phi_marg)

    # --- 6. Linearity check  -------------------------------
    if validate_linearity and len(over_indices) > 0:
        # Trust radii for each marginalized param.
        Ndiag = noise_model.scaled_sigma(toa_data, fiducial_params) ** 2
        trust_radii = jnp.asarray(
            [
                _trust_radius_for_prior(priors[n], i, J_full, Ndiag)
                for n, i in zip(over_list, over_indices)
            ]
        )
        flagged = _check_linearity(
            timing_model, toa_data, fiducial_params,
            over_indices=over_indices,
            over_names=tuple(over_list),
            trust_radii=trust_radii,
            M_full_cols=M_marg,
            tol=laplace_error_tol,
        )
        if flagged:
            if not allow_nonlinear:
                msg = "\n".join(
                    f"  - {name}: Laplace error ratio = {ratio:.2e} > tol = {laplace_error_tol:.2e}"
                    for name, ratio in flagged
                )
                raise NotImplementedError(
                    "marginalize: cannot analytically marginalize parameter(s) "
                    "whose Laplace-approximation error exceeds the tolerance "
                    "within the prior support (i.e., the residual's second-order "
                    "Taylor term is not negligible compared to its linear term "
                    "over one prior-σ step from fiducial_params):\n"
                    f"{msg}\n\n"
                    "Options:\n"
                    "  - Remove these from `over` and sample them instead.\n"
                    "  - Pass allow_nonlinear=True to accept the Laplace approximation\n"
                    "    around fiducial_params (matches discovery's behavior).\n"
                    "  - Raise laplace_error_tol if you've assessed the error and find it acceptable."
                )
            else:
                msg = "; ".join(
                    f"{name} (ratio={ratio:.2e})" for name, ratio in flagged
                )
                warnings.warn(
                    f"marginalize: marginalizing nonlinear parameter(s) via "
                    f"Laplace approximation around fiducial_params: {msg}. "
                    "Marginalized likelihood is accurate to O((y - y_fid)^2). "
                    "For exact treatment, sample these parameters instead.",
                    stacklevel=2,
                )

    # --- 7. Build the sampled_priors dict ----------------------------
    sampled_priors: dict[str, Prior] = {
        n: p for n, p in priors.items() if n not in over_set
    }

    # 8. Reconstruct ParameterVector with the marginalized_mask updated
    full_skeleton = fiducial_params.with_marginalized(over_set)
    reduced_skeleton = full_skeleton  

    # --- 9. Build the wrapper ---------------------------------------
    def likelihood_marg(
        reduced_params: ParameterVector,
        *,
        external_delay: Optional[Float[Array, " n_toas"]] = None,
        external_cov: Optional[
            tuple[Float[Array, "n_toas k"], Float[Array, " k"]]
        ] = None,
    ) -> Float[Array, ""]:
        # Reconstruct the full parameter vector: kept entries come from the
        # user-supplied reduced_params; marg'd entries stay at y_fid (the
        # values cached inside full_skeleton).
        full = full_skeleton.with_free_values(reduced_params.free_values())
        # Compose user's external_cov with the cached marg'd block.
        ext_cov = concat_woodbury_blocks(external_cov, marg_cov_cached)
        return single_pulsar_logL(
            toa_data,
            timing_model,
            noise_model,
            full,
            external_delay=external_delay,
            external_cov=ext_cov,
        )

    return likelihood_marg, sampled_priors, reduced_skeleton
