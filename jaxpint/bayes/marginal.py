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
- :func:`marginalize_single_pulsar` / :func:`marginalize_pta` — integrate out
  the given timing parameters from the single-pulsar / PTA likelihood; each
  returns ``(callable, sampled_priors, reduced_skeleton(s))``.

"""

from __future__ import annotations

import dataclasses
import warnings
from typing import (
    Callable,
    Iterable,
    Mapping,
    Optional,
    Sequence,
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


__all__ = [
    "marginalize_single_pulsar",
    "marginalize_pta",
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
    M_full_cols: Float[Array, "n_toas n_marg"],
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

    Parameters
    ----------
    timing_model : TimingModel
        Model the time residuals are computed through
    toa_data : TOAData
        The TOAs the residuals are evaluated on.
    fiducial_params : ParameterVector
        Linearization point ``y_fid``.
    over_indices : tuple of int
        Indices, into the full parameter vector, of the marginalized parameters
        being checked.
    over_names : tuple of str
        Names of those parameters, parallel to ``over_indices``; used only to
        label the flagged entries in the return value.
    trust_radii : array, shape (n_marg,)
        Per-parameter trust radius ``sigma_i`` (prior σ for a Gaussian prior,
        WLS posterior σ for an ``ImproperPrior``), parallel to ``over_indices``
        -- the scale over which the linearization must remain valid.  See
        :func:`_trust_radius_for_prior`.
    M_full_cols : array, shape (n_toas, n_marg)
        Design-matrix (Jacobian) columns of the marginalized parameters at
        ``y_fid``, in ``over`` order: column ``k`` is ``d r / d y_k``.  Supplies
        the linear-term magnitude (the denominator) each curvature term is
        compared against.
    tol : float
        Ratio threshold; a parameter is flagged when its curvature-to-linear
        ratio exceeds this.

    Returns
    -------
    list of (str, float)
        ``(name, ratio)`` for each marginalized parameter whose ratio exceeds
        ``tol`` (empty when all are acceptably linear).

    Notes
    -----
    This is the most expensive step in marginalization setup (~O(n_params) more
    expensive than the Jacobian alone).  It is skipped when the user sets
    ``validate_linearity=False`` on :func:`marginalize_single_pulsar` /
    :func:`marginalize_pta`.
    """

    from jaxpint.fitters._base import compute_time_residuals

    assert (
        len(over_names)
        == len(over_indices)
        == trust_radii.shape[0]
        == M_full_cols.shape[1]
    )

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
    col: int,
    M_marg: Float[Array, "n_toas n_marg"],
    Ndiag: Float[Array, " n_toas"],
) -> float:
    """Pick the physical trust radius for a marg'd parameter.

    ``col`` is the parameter's column in the marginalization design matrix
    ``M_marg`` (positional, ``0..n_marg-1``), not its global parameter index.

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
    fisher_diag = float(jnp.sum(M_marg[:, col] ** 2 * Ninv))
    if fisher_diag <= 0.0:
        return float("inf")  # degenerate — let the check fire loudly
    return float(1.0 / jnp.sqrt(fisher_diag))


# ---------------------------------------------------------------------------
# Shared marginalization helpers (used by both backends)
# ---------------------------------------------------------------------------


def _validate_marg_inputs(
    over_set: set[str],
    priors: Mapping[str, Prior],
    *,
    valid_names: Optional[Iterable[str]] = None,
) -> None:
    """Validate the priors (and, optionally, the names) for a marginalization set.

    Raises
    ------
    PriorValidationError
        If any name in ``over_set`` has no entry in ``priors``.
    NotImplementedError
        If any prior assigned to an ``over_set`` name is not an
        :class:`ImproperPrior` (the only shape currently supported in ``over``).
    ValueError
        If ``valid_names`` is supplied and ``over_set`` contains a name not in
        it.  The single-pulsar backend passes ``fiducial_params.names``; the PTA
        backend leaves this ``None`` and validates names per pulsar via
        :func:`_resolve_pta_marg_targets` instead.
    """
    missing = over_set - set(priors.keys())
    if missing:
        raise PriorValidationError(
            f"marginalize: priors dict is missing entries for {sorted(missing)} "
            f"(found in `over` but not in `priors`)."
        )
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
    if valid_names is not None:
        unknown = over_set - set(valid_names)
        if unknown:
            raise ValueError(
                f"marginalize: parameter name(s) {sorted(unknown)} in `over` "
                f"are not present in the fiducial parameter vector."
            )


def _marg_woodbury_block(
    timing_model: TimingModel,
    toa_data: TOAData,
    fiducial: ParameterVector,
    indices: tuple[int, ...],
) -> tuple[
    Optional[Float[Array, "n_toas n_marg"]],
    Optional[Float[Array, " n_marg"]],
]:
    """Design matrix and the Φ=1e40 Woodbury block for one pulsar's marg targets.

    Returns ``(M, Phi)`` where ``M = -∂r/∂y[:, indices]`` (the marginalized-target
    columns of the design matrix, in ``indices`` order) and ``Phi`` is the
    improper-prior regularizer (``1e40``).  Returns ``(None, None)`` when ``indices``
    is empty (nothing to marginalize for this pulsar).

    Column-only forward-mode: we push forward ONLY the ``indices`` one-hot tangents,
    building just the ``n_marg`` columns we keep -- never the full
    ``(n_toas, n_params)`` Jacobian.  On 15yr pulsars the parameter vector is dominated
    by DMX bins (``n_params`` in the hundreds to >1000) while ``n_marg`` is a handful,
    so a full ``jacfwd`` + slice would materialize ~``n_params`` columns of activations
    only to discard all but ``n_marg`` -- the peak-memory blowup that OOMs high-cadence
    pulsars.  This ``jvp`` over the marg one-hot basis is exact (identical to the sliced
    full Jacobian) at ``n_marg / n_params`` the memory.  Reverse mode is worse still: it
    vmaps ``n_toas`` backward passes -> O(n_toas**2), and OOMs even sooner.
    """
    if len(indices) == 0:
        return None, None

    from jaxpint.fitters._base import compute_time_residuals

    def _resid_fn(values):
        params = fiducial.with_values(values)
        return compute_time_residuals(timing_model, toa_data, params)

    idx_array = jnp.asarray(indices, dtype=jnp.int32)
    tangents = jax.nn.one_hot(
        idx_array, fiducial.values.shape[0], dtype=fiducial.values.dtype
    )  # (n_marg, n_params): one-hot rows for the marg targets
    cols = jax.vmap(lambda t: jax.jvp(_resid_fn, (fiducial.values,), (t,))[1])(
        tangents
    )  # (n_marg, n_toas): row k = ∂r/∂y_{indices[k]}
    M = -cols.T  # (n_toas, n_marg)
    Phi = jnp.full(len(indices), 1e40, dtype=jnp.float64)
    return M, Phi


def _marg_trust_radii(
    priors: Mapping[str, Prior],
    fqns: Sequence[str],
    M_marg: Float[Array, "n_toas n_marg"],
    Ndiag: Float[Array, " n_toas"],
) -> Float[Array, " n_marg"]:
    """Per-target prior trust radii for the linearity check.

    ``fqns`` are the prior-dict keys for the targets (bare names for a single
    pulsar; ``"{prefix}_{bare}"`` for a PTA pulsar), aligned column-for-column
    with the marginalization design matrix ``M_marg`` (target ``j`` ↔ column ``j``).
    """
    return jnp.asarray(
        [
            _trust_radius_for_prior(priors[fqn], j, M_marg, Ndiag)
            for j, fqn in enumerate(fqns)
        ]
    )


def _raise_or_warn_nonlinear(
    flagged: Sequence[tuple[str, float]],
    *,
    allow_nonlinear: bool,
    tol: float,
) -> None:
    """Act on parameters flagged by the linearity check (shared by both backends).

    Raises ``NotImplementedError`` when ``allow_nonlinear`` is ``False``, or
    emits a :class:`UserWarning` when ``True``, naming the offending parameters
    and their Laplace-error ratios.  No-op when ``flagged`` is empty.
    """
    if not flagged:
        return
    if not allow_nonlinear:
        msg = "\n".join(
            f"  - {name}: Laplace error ratio = {ratio:.2e} > tol = {tol:.2e}"
            for name, ratio in flagged
        )
        raise NotImplementedError(
            "marginalize: cannot analytically marginalize parameter(s) "
            "whose Laplace-approximation error exceeds the tolerance "
            "within the prior support (i.e., the residual's second-order "
            "Taylor term is not negligible compared to its linear term "
            "over one prior-σ step from the fiducial parameters):\n"
            f"{msg}\n\n"
            "Options:\n"
            "  - Remove these from `over` and sample them instead.\n"
            "  - Pass allow_nonlinear=True to accept the Laplace approximation\n"
            "    around the fiducial parameters (matches discovery's behavior).\n"
            "  - Raise laplace_error_tol if you've assessed the error and find it acceptable."
        )
    msg = "; ".join(f"{name} (ratio={ratio:.2e})" for name, ratio in flagged)
    warnings.warn(
        f"marginalize: marginalizing nonlinear parameter(s) via "
        f"Laplace approximation around the fiducial parameters: {msg}. "
        "Marginalized likelihood is accurate to O((y - y_fid)^2). "
        "For exact treatment, sample these parameters instead.",
        stacklevel=3,
    )


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def marginalize_single_pulsar(
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
    """Analytically marginalize single-pulsar timing parameters out of the likelihood.

    Wraps :func:`~jaxpint.likelihood.single_pulsar_logL`, integrating out the
    parameters in ``over`` against their priors via the Woodbury identity (see
    the module docstring).  The integral is exact when the residuals are linear
    in each marginalized parameter; the linearity check
    (``validate_linearity=True``, the default) enforces this.  Global-parameter
    marginalization is intentionally not supported.

    Returns ``(g, sampled_priors, reduced_skeleton)`` where:

    - ``g(reduced_params, *, external_delay=None, external_cov=None)`` is the
      marginalized log-likelihood.  The names in ``over`` no longer appear in
      its parameter dict.  The ``external_delay`` / ``external_cov`` kwargs pass
      through to :func:`~jaxpint.likelihood.single_pulsar_logL` for composition
      with signal injectors.
    - ``sampled_priors`` is ``priors`` with the marginalized entries removed —
      pass this (not the full dict) to :func:`jaxpint.bayes.combine_log_prob`
      to avoid double-counting.
    - ``reduced_skeleton`` is ``fiducial_params.with_marginalized(over)``; use
      it in the sampling loop so ``.free_values()`` returns the kept-entry
      slice that ``g`` expects.

    Currently only :class:`ImproperPrior` is accepted in ``over``; other prior
    shapes raise ``NotImplementedError``.

    Parameters
    ----------
    over
        Fully-qualified names of parameters to marginalize out; a subset of
        ``fiducial_params.names``.
    priors
        Mapping from parameter name to :class:`Prior`.  Must contain an entry
        for every name in ``over``; extras are propagated to ``sampled_priors``.
    toa_data, timing_model, noise_model
        Static likelihood arguments, captured into the returned closure.
    fiducial_params
        The linearization point ``y_fid``; the design matrix
        ``M = ∂r/∂y|_{y_fid}`` (and optional Hessian) is computed once here.
    allow_nonlinear
        If ``False`` (default), parameters that fail the Laplace-error check
        raise ``NotImplementedError``; if ``True``, proceed with the Laplace
        approximation around ``y_fid`` and warn.
    validate_linearity
        If ``True`` (default), compute the Hessian and run the linearity check;
        if ``False``, skip it (the caller takes responsibility for linearity).
    laplace_error_tol
        Threshold for the relative Laplace-approximation error within the prior
        support (default ``1e-6``); see the module docstring for the formula.

    Returns
    -------
    g : callable
        Marginalized log-likelihood, taking a :class:`ParameterVector` whose
        ``marginalized_mask`` matches ``reduced_skeleton``.
    sampled_priors : dict[str, Prior]
        ``priors`` with the marginalized entries removed.
    reduced_skeleton : ParameterVector
        ``fiducial_params`` with ``marginalized_mask=True`` for every name in
        ``over``.

    Raises
    ------
    PriorValidationError
        If any name in ``over`` is missing from ``priors``.
    NotImplementedError
        If ``over`` contains a non-:class:`ImproperPrior` prior, or a parameter
        that fails the linearity check when ``allow_nonlinear=False``.
    """
    over_set = set(over)
    over_list = sorted(over_set)  # deterministic ordering for caching

    _validate_marg_inputs(over_set, priors, valid_names=fiducial_params.names)

    # --- Design matrix + cached Woodbury block at y_fid -------------------
    # Empty marg set -> block is None; the wrapper passes None through
    # concat_woodbury_blocks so the user's external_cov flows through unchanged.
    over_indices = tuple(fiducial_params.param_index(n) for n in over_list)
    M_marg, Phi_marg = _marg_woodbury_block(
        timing_model, toa_data, fiducial_params, over_indices
    )
    if M_marg is None:
        marg_cov_cached = None
    else:
        assert Phi_marg is not None  # M and Phi are returned together
        marg_cov_cached = (M_marg, Phi_marg)

    if validate_linearity and M_marg is not None:
        Ndiag = noise_model.scaled_sigma(toa_data, fiducial_params) ** 2
        trust_radii = _marg_trust_radii(priors, over_list, M_marg, Ndiag)
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
        _raise_or_warn_nonlinear(
            flagged, allow_nonlinear=allow_nonlinear, tol=laplace_error_tol
        )

    sampled_priors: dict[str, Prior] = {
        n: p for n, p in priors.items() if n not in over_set
    }

    reduced_skeleton = fiducial_params.with_marginalized(over_set)

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
        # values cached inside reduced_skeleton).
        full = reduced_skeleton.with_free_values(reduced_params.free_values())
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

    Internal helper used by :func:`marginalize_pta`.  The block for pulsar
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


def marginalize_pta(
    *,
    over: Iterable[str],
    priors: Mapping[str, Prior],
    config: PTAConfig,
    pulsar_names: tuple[str, ...],
    fiducial_pulsar_params: tuple[ParameterVector, ...],
    fiducial_global_params: GlobalParams,
    allow_nonlinear: bool = False,
    validate_linearity: bool = True,
    laplace_error_tol: float = 1e-6,
) -> tuple[Callable, dict[str, Prior], tuple[ParameterVector, ...]]:
    """Analytically marginalize per-pulsar timing parameters across a PTA.

    Wraps :func:`~jaxpint.pta.pta_logL`, integrating out the parameters in
    ``over`` against their priors via the Woodbury identity (see the module
    docstring), per pulsar, across the entire PTA — including in the presence
    of correlated injectors.  Builds per-pulsar Woodbury marginalization blocks,
    wraps them in an internal marginalization injector, and returns a callable
    that evaluates ``pta_logL`` against the marg-augmented config.  Each name in
    ``over`` is matched to the unique pulsar whose parameter vector contains it;
    global-parameter marginalization is intentionally not supported.

    Returns ``(g, sampled_priors, reduced_skeletons)`` where:

    - ``g(reduced_pulsar_params, global_params)`` is the marginalized PTA
      log-likelihood.  The names in ``over`` no longer appear in the per-pulsar
      parameter dicts.
    - ``sampled_priors`` is ``priors`` with the marginalized entries removed —
      pass this (not the full dict) to :func:`jaxpint.bayes.combine_log_prob`.
    - ``reduced_skeletons`` is a tuple of per-pulsar
      ``fiducial_pulsar_params[p].with_marginalized(...)`` skeletons, one per
      pulsar, each marking only that pulsar's marginalized names.

    Currently only :class:`ImproperPrior` is accepted in ``over``; other prior
    shapes raise ``NotImplementedError``.

    Parameters
    ----------
    over
        Fully-qualified names of parameters to marginalize out.  Each must
        belong to exactly one pulsar's ``fiducial_pulsar_params[p].names``.
    priors
        Mapping from parameter name to :class:`Prior`.  Must contain an entry
        for every name in ``over``; extras are propagated to ``sampled_priors``.
    config
        The PTA configuration; ``g`` evaluates ``pta_logL`` against a copy
        augmented with the marginalization injector.
    pulsar_names
        Per-pulsar name prefixes used to resolve which pulsar each ``over``
        entry belongs to.  Length must equal ``config.n_pulsars``.
    fiducial_pulsar_params
        Per-pulsar linearization points ``y_fid``; one design matrix (and
        optional Hessian) is computed per pulsar.
    fiducial_global_params
        Fiducial shared parameters, captured into the returned closure.
    allow_nonlinear
        If ``False`` (default), parameters that fail the Laplace-error check
        raise ``NotImplementedError``; if ``True``, proceed with the Laplace
        approximation and warn.
    validate_linearity
        If ``True`` (default), compute the Hessian and run the linearity check;
        if ``False``, skip it (the caller takes responsibility for linearity).
    laplace_error_tol
        Threshold for the relative Laplace-approximation error within the prior
        support (default ``1e-6``); see the module docstring for the formula.

    Returns
    -------
    g : callable
        Marginalized PTA log-likelihood, taking the tuple of per-pulsar
        :class:`ParameterVector` (matching ``reduced_skeletons``) and the
        shared :class:`GlobalParams`.
    sampled_priors : dict[str, Prior]
        ``priors`` with the marginalized entries removed.
    reduced_skeletons : tuple of ParameterVector
        Per-pulsar skeletons with ``marginalized_mask=True`` for that pulsar's
        names in ``over``.

    Raises
    ------
    PriorValidationError
        If any name in ``over`` is missing from ``priors``.
    ValueError
        If ``pulsar_names`` / ``fiducial_pulsar_params`` / ``config`` disagree
        on pulsar count, or a name in ``over`` matches no or multiple pulsars.
    NotImplementedError
        If ``over`` contains a non-:class:`ImproperPrior` prior, a global
        parameter, or a parameter that fails the linearity check when
        ``allow_nonlinear=False``.
    """
    over_set = set(over)
    n_psr = len(pulsar_names)

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
    _validate_marg_inputs(over_set, priors)

    bare_names_per_pulsar, bare_indices_per_pulsar = _resolve_pta_marg_targets(
        over_set,
        pulsar_names,
        fiducial_pulsar_params,
        fiducial_global_params,
    )

    # Retain per-pulsar marg design matrix M_p for the linearity check below.
    cached_blocks: list[
        Optional[tuple[Float[Array, "n_toas n_marg_p"], Float[Array, " n_marg_p"]]]
    ] = []
    per_pulsar_M: list[Optional[Float[Array, "n_toas n_marg_p"]]] = []
    for p in range(n_psr):
        M_p, Phi_p = _marg_woodbury_block(
            config.timing_models[p],
            config.toa_data_list[p],
            fiducial_pulsar_params[p],
            tuple(bare_indices_per_pulsar[p]),
        )
        if M_p is None:
            cached_blocks.append(None)
        else:
            assert Phi_p is not None  # M and Phi are returned together
            cached_blocks.append((M_p, Phi_p))
        per_pulsar_M.append(M_p)

    if validate_linearity:
        flagged_all: list[tuple[str, float]] = []
        for p in range(n_psr):
            bare_names = bare_names_per_pulsar[p]
            if not bare_names:
                continue
            bare_indices = bare_indices_per_pulsar[p]
            M_p = per_pulsar_M[p]
            assert M_p is not None  # have marg targets

            toa_data_p = config.toa_data_list[p]
            fiducial_p = fiducial_pulsar_params[p]
            fqns = tuple(f"{pulsar_names[p]}_{bare}" for bare in bare_names)
            Ndiag_p = config.noise_models[p].scaled_sigma(toa_data_p, fiducial_p) ** 2
            trust_radii_p = _marg_trust_radii(priors, fqns, M_p, Ndiag_p)
            flagged_p = _check_linearity(
                config.timing_models[p],
                toa_data_p,
                fiducial_p,
                over_indices=tuple(bare_indices),
                over_names=fqns,
                trust_radii=trust_radii_p,
                M_full_cols=M_p,
                tol=laplace_error_tol,
            )
            flagged_all.extend(flagged_p)

        _raise_or_warn_nonlinear(
            flagged_all, allow_nonlinear=allow_nonlinear, tol=laplace_error_tol
        )

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

    reduced_pulsar_skeletons = tuple(
        fiducial_pulsar_params[p].with_marginalized(bare_names_per_pulsar[p])
        for p in range(n_psr)
    )

    sampled_priors: dict[str, Prior] = {
        n: pri for n, pri in priors.items() if n not in over_set
    }

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
