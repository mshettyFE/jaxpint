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

import dataclasses
import warnings
from typing import (
    TYPE_CHECKING,
    Callable,
    Iterable,
    Mapping,
    Optional,
)

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.bayes.priors import ImproperPrior, Prior
from jaxpint.bayes.validate import PriorValidationError
from jaxpint.likelihood import single_pulsar_logL
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.pta.likelihood import PTAConfig, SignalInjector, pta_logL
from jaxpint.types import GlobalParams
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
        p = fiducial_params.with_values(values)
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
    return float(1.0 / jnp.sqrt(fisher_diag))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def marginalize(
    likelihood: Callable,
    *,
    over: Iterable[str],
    priors: Mapping[str, Prior],
    # single-pulsar arguments (required when likelihood is single_pulsar_logL):
    toa_data: Optional[TOAData] = None,
    timing_model: Optional[TimingModel] = None,
    noise_model: Optional[NoiseModel] = None,
    fiducial_params: Optional[ParameterVector] = None,
    # PTA arguments (required when likelihood is pta_logL):
    config: Optional[PTAConfig] = None,
    pulsar_names: Optional[tuple[str, ...]] = None,
    fiducial_pulsar_params: Optional[tuple[ParameterVector, ...]] = None,
    fiducial_global_params: Optional[GlobalParams] = None,
    # common options:
    allow_nonlinear: bool = False,
    validate_linearity: bool = True,
    laplace_error_tol: float = 1e-6,
) -> tuple[Callable, dict[str, Prior], object]:
    """Build an analytically-marginalized log-likelihood.

    Dispatches on ``likelihood`` to the appropriate per-backend marg path.
    Supports :func:`~jaxpint.likelihood.single_pulsar_logL` (per-pulsar
    timing-model marg) and :func:`~jaxpint.pta.pta_logL`
    (per-pulsar timing-model marg across an entire PTA, including in the
    presence of correlated injectors).  Global-parameter marg (CW / GWB
    hyperparameters) is intentionally not supported — discovery samples
    those via MCMC and we follow that convention.

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

    Currently only accepts :class:`ImproperPrior` in ``over``; Gaussian
    and Uniform raise ``NotImplementedError`` with a clear message.

    Parameters
    ----------
    likelihood
        Currently must be :func:`~jaxpint.likelihood.single_pulsar_logL`
        (or a callable with the same signature).  Multi-pulsar
        marginalization is planned for the next phase.
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
        Required when ``likelihood is single_pulsar_logL``.
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
        kwargs pass through to :func:`~jaxpint.likelihood.single_pulsar_logL` for composition
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
        If ``likelihood`` is not a supported callable; if ``over`` contains
        a parameter whose assigned prior is not an :class:`ImproperPrior`;
        or if it contains a parameter that fails the linearity check when
        ``allow_nonlinear=False``.
    """
    if likelihood is single_pulsar_logL:
        if (
            toa_data is None
            or timing_model is None
            or noise_model is None
            or fiducial_params is None
        ):
            raise TypeError(
                "marginalize(single_pulsar_logL, ...) requires the kwargs "
                "`toa_data`, `timing_model`, `noise_model`, `fiducial_params`."
            )
        return _marginalize_single_pulsar(
            over=over,
            priors=priors,
            toa_data=toa_data,
            timing_model=timing_model,
            noise_model=noise_model,
            fiducial_params=fiducial_params,
            allow_nonlinear=allow_nonlinear,
            validate_linearity=validate_linearity,
            laplace_error_tol=laplace_error_tol,
        )

    if likelihood is pta_logL:
        if (
            config is None
            or pulsar_names is None
            or fiducial_pulsar_params is None
            or fiducial_global_params is None
        ):
            raise TypeError(
                "marginalize(pta_logL, ...) requires the kwargs "
                "`config`, `pulsar_names`, `fiducial_pulsar_params`, "
                "`fiducial_global_params`."
            )
        return _marginalize_pta(
            over=over,
            priors=priors,
            config=config,
            pulsar_names=tuple(pulsar_names),
            fiducial_pulsar_params=tuple(fiducial_pulsar_params),
            fiducial_global_params=fiducial_global_params,
            allow_nonlinear=allow_nonlinear,
            validate_linearity=validate_linearity,
            laplace_error_tol=laplace_error_tol,
        )

    raise NotImplementedError(
        "marginalize() supports only single_pulsar_logL and pta_logL as the "
        f"`likelihood` argument; got "
        f"{getattr(likelihood, '__name__', repr(likelihood))!r}."
    )


def _marginalize_single_pulsar(
    *,
    over: Iterable[str],
    priors: Mapping[str, Prior],
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    fiducial_params: ParameterVector,
    allow_nonlinear: bool,
    validate_linearity: bool,
    laplace_error_tol: float,
) -> tuple[Callable, dict[str, Prior], ParameterVector]:
    """Single-pulsar marg implementation.

    Internal helper called by :func:`marginalize` when
    ``likelihood is single_pulsar_logL``.  Same semantics as the public
    function; see :func:`marginalize` for parameter / return documentation.
    """
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
            p = fiducial_params.with_values(values)
            return compute_time_residuals(timing_model, toa_data, p)

        # Forward-mode: the design matrix dr/dy is tall (n_toas >> n_params),
        # so jacfwd costs O(n_toas * n_params); reverse mode (jax.jacobian)
        # vmaps n_toas backward passes -> O(n_toas**2) and OOMs on
        # high-cadence pulsars (e.g. 35k TOAs).
        J_full = jax.jacfwd(_resid_fn)(fiducial_params.values)
        marg_idx_array = jnp.asarray(over_indices, dtype=jnp.int32)
        M_marg = -J_full[:, marg_idx_array]  # (n_toas, n_marg)
        Phi_marg = jnp.full(len(over_indices), 1e40, dtype=jnp.float64)
        marg_cov_cached = (M_marg, Phi_marg)

    # --- 6. Linearity check  -------------------------------
    if validate_linearity and len(over_indices) > 0:
        # len(over_indices) > 0 means the Jacobian branch above ran.
        assert J_full is not None and M_marg is not None
        # Trust radii for each marginalized param.
        Ndiag = noise_model.scaled_sigma(toa_data, fiducial_params) ** 2
        trust_radii = jnp.asarray(
            [
                _trust_radius_for_prior(priors[n], i, J_full, Ndiag)
                for n, i in zip(over_list, over_indices)
            ]
        )
        flagged = _check_linearity(
            timing_model,
            toa_data,
            fiducial_params,
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
            # The marg block is the timing design matrix M at Φ=1e40 -- genuinely
            # collinear for multi-parameter MSPs. The QR (square-root) Woodbury
            # avoids the Gram-squaring that costs the Cholesky form ~4 digits.
            use_qr=True,
        )

    return likelihood_marg, sampled_priors, reduced_skeleton


# ---------------------------------------------------------------------------
# PTA marginalization
# ---------------------------------------------------------------------------


class _MarginalizationInjector(SignalInjector):
    """Injects cached per-pulsar marg Woodbury blocks at evaluation time.

    Internal helper used by :func:`_marginalize_pta`.  The block for pulsar
    ``p`` is fixed at setup (computed from the fiducial Jacobian); the
    injector ignores its ``pulsar_params`` and ``global_params`` arguments
    and just returns the cached tuple.

    Stored as a regular Python attribute (not an :class:`eqx.field`),
    mirroring the convention used by concrete :class:`SignalInjector`
    subclasses like
    :class:`~jaxpint.pta.signals.gwb.CURNInjector` and
    :class:`~jaxpint.pta.signals.correlated_gwb.HDCorrelatedGWBInjector`.
    """

    def __init__(
        self,
        cached_blocks: tuple[
            Optional[tuple[Float[Array, "n_toas n_marg_p"], Float[Array, " n_marg_p"]]],
            ...,
        ],
    ):
        self.cached_blocks = cached_blocks

    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        # Marg'd parameters are per-pulsar and held at fiducial — no new
        # globals registered.
        return global_params

    def covariance(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[tuple[Float[Array, "n_toas n_marg_p"], Float[Array, " n_marg_p"]]]:
        return self.cached_blocks[p]


def _resolve_pta_marg_targets(
    over: Iterable[str],
    pulsar_names: tuple[str, ...],
    fiducial_pulsar_params: tuple[ParameterVector, ...],
    fiducial_global_params: GlobalParams,
) -> tuple[list[list[str]], list[list[int]]]:
    """Classify each FQN in ``over`` as a per-pulsar marg target.

    Returns ``(bare_names_per_pulsar, bare_indices_per_pulsar)`` — both
    are lists of length ``n_psr``.  For pulsar ``p``, the lists contain
    the bare parameter names and corresponding indices within
    ``fiducial_pulsar_params[p].values`` for parameters marg'd in that
    pulsar.  Pulsars with no marg targets get empty lists.

    Raises ``NotImplementedError`` for global-parameter names (deferred)
    and ``ValueError`` for unresolvable names.
    """
    n_psr = len(pulsar_names)
    global_name_set = set(fiducial_global_params.names)

    # Per-pulsar accumulators (preserve sort order for deterministic caching).
    bare_names_per_pulsar: list[list[str]] = [[] for _ in range(n_psr)]
    bare_indices_per_pulsar: list[list[int]] = [[] for _ in range(n_psr)]

    for fqn in sorted(set(over)):
        # Global names take precedence — they're handled separately (and
        # currently rejected).
        if fqn in global_name_set:
            raise NotImplementedError(
                f"marginalize: global-parameter marginalization is not yet "
                f"implemented; got {fqn!r} in `over` (matches a name in "
                f"`fiducial_global_params.names`).  CW and GWB hyperparameters "
                "should be sampled, following the convention used by `discovery`."
            )

        # Find matching pulsar(s).  Iterate all pulsars to handle the case
        # where one pulsar's name is a prefix of another (e.g. "J1234" and
        # "J1234_extra"): both might syntactically match different bare-name
        # splits, so we resolve by checking which split yields a real bare
        # name in the corresponding pulsar_params.
        matches: list[tuple[int, str]] = []
        for p in range(n_psr):
            prefix = f"{pulsar_names[p]}_"
            if not fqn.startswith(prefix):
                continue
            bare = fqn[len(prefix) :]
            if bare in fiducial_pulsar_params[p].names:
                matches.append((p, bare))

        if not matches:
            raise ValueError(
                f"marginalize: name {fqn!r} in `over` matches no pulsar "
                f"and is not a global-parameter name.  Expected either a "
                f"fully-qualified per-pulsar name (e.g. "
                f"f'{{pulsar_name}}_{{param_name}}') or a name in "
                f"`fiducial_global_params.names`."
            )
        if len(matches) > 1:
            raise ValueError(
                f"marginalize: name {fqn!r} in `over` matches multiple "
                f"pulsars: {[pulsar_names[p] for p, _ in matches]}.  "
                "Pulsar names may not be ambiguous prefixes of each other "
                "for the same bare parameter."
            )

        p, bare = matches[0]
        bare_names_per_pulsar[p].append(bare)
        bare_indices_per_pulsar[p].append(fiducial_pulsar_params[p].param_index(bare))

    return bare_names_per_pulsar, bare_indices_per_pulsar


def _marginalize_pta(
    *,
    over: Iterable[str],
    priors: Mapping[str, Prior],
    config: PTAConfig,
    pulsar_names: tuple[str, ...],
    fiducial_pulsar_params: tuple[ParameterVector, ...],
    fiducial_global_params: GlobalParams,
    allow_nonlinear: bool,
    validate_linearity: bool,
    laplace_error_tol: float,
) -> tuple[Callable, dict[str, Prior], tuple[ParameterVector, ...]]:
    """PTA marg implementation (per-pulsar timing parameters).

    Internal helper called by :func:`marginalize` when
    ``likelihood is pta_logL``.  Builds per-pulsar Woodbury marg blocks,
    wraps them in a :class:`_MarginalizationInjector`, and returns a
    callable that evaluates ``pta_logL`` against the marg-augmented
    config.  See :func:`marginalize` for parameter / return documentation.
    """
    from jaxpint.fitters._base import compute_time_residuals

    over_set = set(over)
    n_psr = len(pulsar_names)

    # --- 1. Validate setup ---------------------------------------------------
    if len(fiducial_pulsar_params) != n_psr:
        raise ValueError(
            f"marginalize: pulsar_names has length {n_psr} but "
            f"fiducial_pulsar_params has length {len(fiducial_pulsar_params)}."
        )
    if config.n_pulsars != n_psr:
        raise ValueError(
            f"marginalize: pulsar_names has length {n_psr} but "
            f"config has {config.n_pulsars} pulsars."
        )

    # --- 2. Completeness check ----------------------------------------------
    missing = over_set - set(priors.keys())
    if missing:
        raise PriorValidationError(
            f"marginalize: priors dict is missing entries for {sorted(missing)} "
            f"(found in `over` but not in `priors`)."
        )

    # --- 3. Prior-shape check (ImproperPrior only) --------------------------
    bad = [
        (n, type(priors[n]).__name__)
        for n in sorted(over_set)
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

    # --- 4. Resolve over → per-pulsar (bare, index) lists ------------------
    bare_names_per_pulsar, bare_indices_per_pulsar = _resolve_pta_marg_targets(
        over_set,
        pulsar_names,
        fiducial_pulsar_params,
        fiducial_global_params,
    )

    # --- 5. Per-pulsar Jacobian + cached Woodbury blocks --------------------
    cached_blocks: list[
        Optional[tuple[Float[Array, "n_toas n_marg_p"], Float[Array, " n_marg_p"]]]
    ] = []
    # For the linearity check we also retain per-pulsar M_p and over_indices.
    per_pulsar_M: list[Optional[Float[Array, "n_toas n_marg_p"]]] = []
    per_pulsar_J_full: list[Optional[Float[Array, "n_toas n_params_p"]]] = []
    for p in range(n_psr):
        bare_indices = bare_indices_per_pulsar[p]
        if len(bare_indices) == 0:
            cached_blocks.append(None)
            per_pulsar_M.append(None)
            per_pulsar_J_full.append(None)
            continue

        timing_model_p = config.timing_models[p]
        toa_data_p = config.toa_data_list[p]
        fiducial_p = fiducial_pulsar_params[p]

        def _resid_fn(values, _tm=timing_model_p, _td=toa_data_p, _fp=fiducial_p):
            params = _fp.with_values(values)
            return compute_time_residuals(_tm, _td, params)

        # Forward-mode (see _marginalize_single_pulsar): tall design matrix,
        # so jacfwd is O(n_toas * n_params) vs jacrev's O(n_toas**2) blowup.
        J_full_p = jax.jacfwd(_resid_fn)(fiducial_p.values)
        idx_array = jnp.asarray(bare_indices, dtype=jnp.int32)
        M_p = -J_full_p[:, idx_array]
        Phi_p = jnp.full(len(bare_indices), 1e40, dtype=jnp.float64)
        cached_blocks.append((M_p, Phi_p))
        per_pulsar_M.append(M_p)
        per_pulsar_J_full.append(J_full_p)

    # --- 6. Linearity check (per pulsar) ------------------------------------
    if validate_linearity:
        flagged_all: list[tuple[str, float]] = []
        for p in range(n_psr):
            bare_names = bare_names_per_pulsar[p]
            if not bare_names:
                continue
            bare_indices = bare_indices_per_pulsar[p]
            J_full_p = per_pulsar_J_full[p]
            M_p = per_pulsar_M[p]
            assert J_full_p is not None and M_p is not None  # have marg targets

            noise_model_p = config.noise_models[p]
            toa_data_p = config.toa_data_list[p]
            fiducial_p = fiducial_pulsar_params[p]

            Ndiag_p = noise_model_p.scaled_sigma(toa_data_p, fiducial_p) ** 2
            trust_radii_p = jnp.asarray(
                [
                    _trust_radius_for_prior(priors[fqn], idx, J_full_p, Ndiag_p)
                    for fqn, idx in zip(
                        (f"{pulsar_names[p]}_{bare}" for bare in bare_names),
                        bare_indices,
                    )
                ]
            )
            flagged_p = _check_linearity(
                config.timing_models[p],
                toa_data_p,
                fiducial_p,
                over_indices=tuple(bare_indices),
                over_names=tuple(f"{pulsar_names[p]}_{bare}" for bare in bare_names),
                trust_radii=trust_radii_p,
                M_full_cols=M_p,
                tol=laplace_error_tol,
            )
            flagged_all.extend(flagged_p)

        if flagged_all:
            if not allow_nonlinear:
                msg = "\n".join(
                    f"  - {name}: Laplace error ratio = {ratio:.2e} > tol = {laplace_error_tol:.2e}"
                    for name, ratio in flagged_all
                )
                raise NotImplementedError(
                    "marginalize: cannot analytically marginalize parameter(s) "
                    "whose Laplace-approximation error exceeds the tolerance "
                    "within the prior support (i.e., the residual's second-order "
                    "Taylor term is not negligible compared to its linear term "
                    "over one prior-σ step from fiducial):\n"
                    f"{msg}\n\n"
                    "Options:\n"
                    "  - Remove these from `over` and sample them instead.\n"
                    "  - Pass allow_nonlinear=True to accept the Laplace approximation\n"
                    "    around the fiducial parameters (matches discovery's behavior).\n"
                    "  - Raise laplace_error_tol if you've assessed the error and find it acceptable."
                )
            else:
                msg = "; ".join(
                    f"{name} (ratio={ratio:.2e})" for name, ratio in flagged_all
                )
                warnings.warn(
                    f"marginalize: marginalizing nonlinear parameter(s) via "
                    f"Laplace approximation around fiducial: {msg}. "
                    "Marginalized likelihood is accurate to O((y - y_fid)^2). "
                    "For exact treatment, sample these parameters instead.",
                    stacklevel=2,
                )

    # --- 7. Build modified config with marg injector appended --------------
    marg_injector = _MarginalizationInjector(tuple(cached_blocks))
    # ``signal_injectors`` is a static-field tuple of ``SignalInjector``
    # instances.  ``eqx.tree_at`` interprets ``c.signal_injectors`` as a
    # pytree of leaves rather than as a single field to replace, so use
    # the standard dataclass replacement instead — eqx.Module is a
    # frozen dataclass under the hood.
    modified_config = dataclasses.replace(
        config,
        signal_injectors=config.signal_injectors + (marg_injector,),
    )

    # --- 8. Reduced per-pulsar skeletons -----------------------------------
    reduced_pulsar_skeletons = tuple(
        fiducial_pulsar_params[p].with_marginalized(bare_names_per_pulsar[p])
        for p in range(n_psr)
    )

    # --- 9. Sampled priors --------------------------------------------------
    sampled_priors: dict[str, Prior] = {
        n: pri for n, pri in priors.items() if n not in over_set
    }

    # --- 10. Build the wrapper ---------------------------------------------
    def likelihood_marg_pta(
        global_params: GlobalParams,
        reduced_pulsar_params: tuple[ParameterVector, ...],
    ) -> Float[Array, ""]:
        # Reconstruct full per-pulsar params: kept entries come from the
        # user-supplied reduced_pulsar_params; marg'd entries stay at the
        # fiducial values cached inside reduced_pulsar_skeletons[p].
        full_pulsar_params = tuple(
            skeleton.with_free_values(rp.free_values())
            for skeleton, rp in zip(reduced_pulsar_skeletons, reduced_pulsar_params)
        )
        return pta_logL(global_params, full_pulsar_params, modified_config)

    return likelihood_marg_pta, sampled_priors, reduced_pulsar_skeletons
