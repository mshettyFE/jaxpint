"""Multi-pulsar PTA log-likelihood.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars, with signal injections (CW, GWB, etc.) mediated by the
:class:`SignalInjector` abstract base class.  Optional cross-pulsar
correlations (e.g. a Hellings-Downs gravitational-wave background, a
clock-error monopole, an ephemeris dipole — anything whose covariance is
described by a per-pulsar Fourier basis and an ORF) are modeled by
:class:`CorrelatedSignalInjector` instances and handled via a two-tier
Woodbury scheme:

- **Inner tier** (per-pulsar): :func:`~jaxpint.utils.woodbury_dot` /
  :func:`~jaxpint.utils.woodbury_solve` evaluate the per-pulsar Gaussian
  likelihood without forming the full covariance.
- **Outer tier** (cross-pulsar, only when ``config.correlated_injectors``
  is non-empty): a dense Cholesky solve on the compressed Fourier-basis
  system couples pulsars via each injector's ORF.

The global covariance is ``C = N + V @ Phi_corr @ V^T`` where
``N = blockdiag(C_1, ..., C_n)`` is the block-diagonal per-pulsar noise,
``V = blockdiag(F_1, ..., F_n)`` collects per-pulsar Fourier bases for
the correlated signal, and ``Phi_corr = Gamma kron diag(S)`` is the
ORF-weighted spectrum of the correlated signal.

References
----------
.. [pta_vh09] van Haasteren et al. (2009), "On measuring the gravitational-wave
   background using pulsar timing arrays", MNRAS 395, 1005.
.. [pta_vh14] van Haasteren & Vallisneri (2014), PRD 90, 104012.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, cast

import jax
import jax.numpy as jnp
import equinox as eqx

try:
    from beartype import beartype
except ModuleNotFoundError:  # dev-only extra; without it jaxtyped is a no-op
    beartype = None
from jaxtyping import Array, Float, jaxtyped

from jaxpint.likelihood import (
    _residuals_and_woodbury,
    _validate_coefficient_length,
    precompute_single_pulsar_factor,
    single_pulsar_logL,
    single_pulsar_logL_with_factor,
)
from jaxpint.utils import WoodburyFactor
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import concat_woodbury_blocks, woodbury_dot, woodbury_solve

from jaxpint.types import GlobalParams


# The injector contracts live in their own leaf module so the engine and the
# signal implementations can both depend on them without depending on each other.
from jaxpint.pta.injectors import CorrelatedSignalInjector, SignalInjector


# ---------------------------------------------------------------------------
# PTA configuration
# ---------------------------------------------------------------------------


class PTAConfig(eqx.Module):
    """Configuration for PTA likelihood evaluation.

    Holds the per-pulsar TOA data, timing/noise models, per-pulsar
    :class:`SignalInjector` instances (CW, CURN, etc.), and optionally
    :class:`CorrelatedSignalInjector` instances (e.g. an HD-correlated GWB).

    When ``correlated_injectors`` is empty (the default), :func:`pta_logL`
    is a sum of independent per-pulsar log-likelihoods.  When non-empty,
    each correlated injector contributes an outer-tier Cholesky solve over
    the cross-pulsar Fourier-basis system.

    ``toa_data_list`` and ``noise_models`` are *dynamic* (traced) fields;
    marking them static balloons jit memory because the per-pulsar arrays
    get baked into the compiled HLO. The remaining fields are compile-time
    constants.

    Raises
    ------
    ValueError
        If ``toa_data_list``, ``timing_models``, and ``noise_models`` do
        not all have the same length.
    """

    toa_data_list: tuple[TOAData, ...]
    noise_models: tuple[NoiseModel, ...]
    timing_models: tuple[TimingModel, ...] = eqx.field(static=True)
    signal_injectors: tuple[SignalInjector, ...] = eqx.field(static=True)
    correlated_injectors: tuple[CorrelatedSignalInjector, ...] = eqx.field(
        static=True, default=()
    )

    def __post_init__(self):
        n_toa = len(self.toa_data_list)
        n_tm = len(self.timing_models)
        n_nm = len(self.noise_models)
        if not (n_toa == n_tm == n_nm):
            raise ValueError(
                f"Mismatched pulsar counts: {n_toa} TOA datasets, "
                f"{n_tm} timing models, {n_nm} noise models. "
                f"All three must have the same length (one per pulsar)."
            )
        # GP basis times must live in one frame across the whole PTA: the
        # correlated (HD/CURN) kernels compare basis phases *between* pulsars,
        # so a config mixing e.g. barycentric- and TDB-based pulsars computes
        # cross-terms in inconsistent coordinates.  ``basis_coord`` is each
        # producer's declaration (see TOAData.basis_coord); unset entries are
        # allowed here — pulsars without GP components never read it, and
        # ``require_basis_seconds`` raises for the ones that do.
        # getattr: tests construct shape-only configs with None placeholders
        # in toa_data_list, which have no coordinate to check.
        coords = {
            p: getattr(td, "basis_coord", None)
            for p, td in enumerate(self.toa_data_list)
        }
        coords = {p: c for p, c in coords.items() if c is not None}
        if len(set(coords.values())) > 1:
            raise ValueError(
                "Mixed GP basis time coordinates across pulsars: "
                f"{coords}. All TOAData in one PTAConfig must declare the "
                "same basis_coord ('barycentric' for bridge/native-loaded "
                "real data, 'tdb' for zero-geometry synthetic data)."
            )

    @property
    def n_pulsars(self) -> int:
        """Number of pulsars in this PTA configuration.

        Returns
        -------
        int
            Length of ``toa_data_list``.
        """
        return len(self.toa_data_list)


# ---------------------------------------------------------------------------
# Shared per-pulsar aggregation helper
# ---------------------------------------------------------------------------


def _collect_injector_ext_delay(
    p: int,
    toa_data_p: TOAData,
    pulsar_params_p: ParameterVector,
    global_params: GlobalParams,
    signal_injectors,
) -> Optional[Float[Array, " n_toas"]]:
    """Sum per-pulsar deterministic-delay contributions from injectors.

    Returns the summed delay or ``None`` (if no injector contributes).
    """
    delays = [
        inj.delay(p, toa_data_p, pulsar_params_p, global_params)
        for inj in signal_injectors
    ]
    delays = [d for d in delays if d is not None]
    return cast(Float[Array, " n_toas"], sum(delays)) if delays else None


def _collect_injector_ext_cov(
    p: int,
    toa_data_p: TOAData,
    pulsar_params_p: ParameterVector,
    global_params: GlobalParams,
    signal_injectors,
):
    """Concatenate per-pulsar covariance contributions from injectors.

    Returns ``(U_ext, Phi_ext)`` or ``None`` (if no injector contributes).
    """
    covs = [
        inj.covariance(p, toa_data_p, pulsar_params_p, global_params)
        for inj in signal_injectors
    ]
    return concat_woodbury_blocks(*covs)


def _collect_per_pulsar_external_inputs(
    p: int,
    toa_data: TOAData,
    pulsar_params: ParameterVector,
    global_params: GlobalParams,
    signal_injectors: tuple[SignalInjector, ...],
) -> tuple[
    Optional[Float[Array, " n_toas"]],
    Optional[tuple[Float[Array, "n_toas k"], Float[Array, " k"]]],
]:
    """Aggregate per-pulsar ``(ext_delay, ext_cov)`` from all signal injectors.

    The combined form used by :func:`pta_logL`; a thin composition of
    :func:`_collect_injector_ext_delay` and :func:`_collect_injector_ext_cov`,
    which are also callable individually by paths that need only one.
    """
    return (
        _collect_injector_ext_delay(
            p, toa_data, pulsar_params, global_params, signal_injectors
        ),
        _collect_injector_ext_cov(
            p, toa_data, pulsar_params, global_params, signal_injectors
        ),
    )


# ---------------------------------------------------------------------------
# Per-pulsar intermediates (inner tier for the correlated outer-tier solve)
# ---------------------------------------------------------------------------


@jaxtyped(typechecker=beartype)
def _per_pulsar_intermediates(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    F_corr: Float[Array, "n_toas n_basis"],
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_ext"], Float[Array, " n_ext"]]
    ] = None,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_basis"],
    Float[Array, "n_basis n_basis"],
]:
    """Compute per-pulsar Woodbury intermediates for the outer tier.

    Uses the existing per-pulsar Woodbury solver (inner tier) to compute
    quantities projected onto the correlated-signal Fourier basis (HD
    GWB, dipole-correlated noise, monopole clock errors, etc. — anything
    a :class:`CorrelatedSignalInjector` provides).

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data for this pulsar.
    timing_model : TimingModel
        Timing model for this pulsar.
    noise_model : NoiseModel
        Noise model (white + correlated) for this pulsar.
    params : ParameterVector
        Timing and noise parameters for this pulsar.
    F_corr : (n_toas, n_basis) array
        Fourier design matrix for this pulsar from one or more
        :class:`CorrelatedSignalInjector` instances.  When multiple
        correlated injectors are present this is the column-concatenated
        stack of their per-pulsar bases.
    external_delay : (n_toas,) array, optional
        Deterministic signal delay (e.g. CW).
    external_cov : (U_ext, Phi_ext) tuple, optional
        Per-pulsar stochastic covariance from SignalInjectors (e.g. CURN).

    Returns
    -------
    rCr_p : scalar
        ``r_p^T C_p^{-1} r_p`` — pulsar p's residual quadratic form under
        inverse per-pulsar noise.
    logdetC_p : scalar
        ``log|C_p|`` — pulsar p's noise log-determinant.
    basis_proj_residual_p : (n_basis,) array
        ``F_p^T C_p^{-1} r_p`` — pulsar p's residuals projected onto the
        correlated-signal Fourier basis, weighted by inverse per-pulsar
        noise.  If you were doing a generalized least-squares fit for the
        Fourier coefficients, this is the right-hand-side of the normal
        equations.
    basis_overlap_p : (n_basis, n_basis) array
        ``F_p^T C_p^{-1} F_p`` — the correlated-signal Fourier basis's
        self-overlap matrix under the ``C_p^{-1}`` inner product.  This is
        the Fisher information matrix for pulsar p's Fourier coefficients.
    """
    # 1-2. Residuals and the per-pulsar Woodbury blocks (shared preamble)
    r, Ndiag, U, Phi = _residuals_and_woodbury(
        toa_data, timing_model, noise_model, params, external_delay, external_cov
    )

    # 3. Inner tier: per-pulsar Woodbury
    rCr_p, logdetC_p = woodbury_dot(Ndiag, U, Phi, r, r)

    # 4. C_p^{-1} r_p and C_p^{-1} F_corr via Woodbury solve
    #    Combine into one solve: B = [r[:, None], F_corr]
    B = jnp.concatenate([r[:, None], F_corr], axis=1)  # (n_toas, 1 + n_basis)
    Cinv_B = woodbury_solve(Ndiag, U, Phi, B)
    Cinv_r = Cinv_B[:, 0]  # (n_toas,)
    Cinv_F = Cinv_B[:, 1:]  # (n_toas, n_basis)

    # 5. Project onto the correlated-signal Fourier basis
    basis_proj_residual_p = F_corr.T @ Cinv_r  # (n_basis,)
    basis_overlap_p = F_corr.T @ Cinv_F  # (n_basis, n_basis)

    return rCr_p, logdetC_p, basis_proj_residual_p, basis_overlap_p


# ---------------------------------------------------------------------------
# Joint outer-tier helpers (one Cholesky over all correlated injectors)
# ---------------------------------------------------------------------------
#
# Index notation used throughout this section:
#
#   k : correlated-injector index, k = 0, ..., K-1, where K = len(config.correlated_injectors).
#       Each injector k contributes its own Fourier basis F_{k,p} per pulsar
#       (shape (n_toas_p, n_basis_k)), PSD S_k (shape (n_basis_k,)),
#       and ORF matrix Γ_k (shape (n_psr, n_psr)).
#   p : pulsar index, p = 0, ..., n_psr - 1.
#   b : basis-function index within one injector's Fourier basis, b = 0, ..., n_basis_k - 1.
#       Note n_basis_k can vary per injector.
#
# Two derived sizes show up everywhere:
#
#   n_basis_total = sum_k n_basis_k          (length of the stacked per-pulsar basis)
#   n_joint       = sum_k n_psr * n_basis_k  (size of the joint outer-tier system)
#
# Two natural orderings of the joint outer-tier index space appear:
#
#   (k, b)    — flat layout of one pulsar's slab: each pulsar's Fourier basis
#               is the column-concat [F_{0,p} | F_{1,p} | ... | F_{K-1,p}]
#               and the inner tier produces basis_proj_residual_p /
#               basis_overlap_p in this layout.
#   (k, p, b) — flat layout of the joint outer-tier vector / matrix: k slowest,
#               p middle, b fastest.  Both basis_proj_residual_joint and
#               Sigma_joint live here.
#
# Helpers below convert per-pulsar (k, b) slabs into joint-system (k, p, b)
# objects, and build Φ_joint in (k, p, b) directly.


def _stacked_fourier_basis(
    correlated_injectors: tuple[CorrelatedSignalInjector, ...],
    toa_data: TOAData,
) -> Float[Array, "n_toas n_basis_total"]:
    """Concatenate per-pulsar Fourier bases across all correlated injectors.

    For one pulsar, returns ``F_stack = [F_{0,p} | F_{1,p} | ... | F_{K-1,p}]``,
    where ``F_{k,p}`` is injector ``k``'s per-pulsar basis.  Shape
    ``(n_toas, n_basis_total)``; columns are in (k, b) layout (injector index
    outermost, basis index within injector inside): columns
    ``[0:n_basis_0]`` are injector 0's basis, columns
    ``[n_basis_0:n_basis_0+n_basis_1]`` are injector 1's, etc.

    Fed to :func:`_per_pulsar_intermediates` so that the inner tier computes,
    in one Woodbury solve, the projections / overlaps of all injectors' bases
    against the per-pulsar noise covariance ``C_p``.
    """
    return jnp.concatenate(
        [cinj.get_fourier_basis(toa_data) for cinj in correlated_injectors],
        axis=1,
    )


def n_basis_per_injector(
    correlated_injectors: tuple[CorrelatedSignalInjector, ...],
    toa_data_0: TOAData,
) -> tuple[int, ...]:
    """Per-injector basis widths ``(n_basis_0, n_basis_1, ..., n_basis_{K-1})``.

    Each ``n_basis_k`` is the number of Fourier columns injector ``k``
    contributes per pulsar.  This is the same for every pulsar (the basis
    width is set at injector construction), so we read it off a single TOA
    dataset.
    """
    return tuple(
        cinj.get_fourier_basis(toa_data_0).shape[1] for cinj in correlated_injectors
    )


def _phi_and_phi_inv_joint(
    correlated_injectors: tuple[CorrelatedSignalInjector, ...],
    global_params: GlobalParams,
) -> tuple[Float[Array, "n_joint n_joint"], Float[Array, "n_joint n_joint"]]:
    """Build joint ``(Phi_joint, Phi_joint_inv)`` in (k, p, b) ordering.

    ``Phi_joint = blockdiag_k( Γ_k ⊗ diag(S_k) )``.  The K diagonal blocks
    are independent across injectors (different correlated signals have
    independent priors); inside each block, pulsars are coupled via Γ_k
    and basis functions are independent (diagonal in b).

    Each block has shape ``(n_psr * n_basis_k, n_psr * n_basis_k)``.  The
    full matrix has shape ``(n_joint, n_joint)``.
    """
    Phi_blocks = []
    Phi_inv_blocks = []
    for cinj in correlated_injectors:
        Gamma = cinj.get_orf_matrix()
        S = cinj.get_psd(global_params)
        Phi_blocks.append(jnp.kron(Gamma, jnp.diag(S)))
        Phi_inv_blocks.append(jnp.kron(jnp.linalg.inv(Gamma), jnp.diag(1.0 / S)))
    return (
        jax.scipy.linalg.block_diag(*Phi_blocks),
        jax.scipy.linalg.block_diag(*Phi_inv_blocks),
    )


def _assemble_basis_overlap_joint_kpb(
    basis_overlap_per_pulsar: list,
    n_basis_per_k: tuple[int, ...],
    n_psr: int,
) -> Float[Array, "n_joint n_joint"]:
    """Assemble joint ``basis_overlap_joint`` in (k, p, b) layout.

    Each per-pulsar slab ``basis_overlap_per_pulsar[p] = F_stack_pᵀ C_p⁻¹
    F_stack_p`` is a dense ``(n_basis_total, n_basis_total)`` matrix in
    (k, b) × (k, b) layout (because the inner tier consumed the stacked
    basis ``[F_{0,p} | F_{1,p} | ...]``).  Within that slab, slicing rows
    by ``local_slices[k_a]`` and columns by ``local_slices[k_b]`` selects
    the cross-injector block ``F_{k_a,p}ᵀ C_p⁻¹ F_{k_b,p}``.

    The result is the K×K block matrix whose ``(k_a, k_b)`` outer block is
    block-diagonal across pulsars with per-pulsar entries
    ``F_{k_a,p}ᵀ C_p⁻¹ F_{k_b,p}``.  The full matrix has shape
    ``(n_joint, n_joint)`` and matches :func:`_phi_and_phi_inv_joint`'s
    (k, p, b) layout so that ``Σ_joint = Phi_joint_inv + basis_overlap_joint``
    is a legal addition.
    """
    local_slices = []
    offset = 0
    for nb in n_basis_per_k:
        local_slices.append(slice(offset, offset + nb))
        offset += nb
    K = len(n_basis_per_k)
    return jnp.block(
        [
            [
                jax.scipy.linalg.block_diag(
                    *[
                        basis_overlap_per_pulsar[p][
                            local_slices[k_a], local_slices[k_b]
                        ]
                        for p in range(n_psr)
                    ]
                )
                for k_b in range(K)
            ]
            for k_a in range(K)
        ]
    )


def _assemble_basis_proj_residual_joint_kpb(
    basis_proj_residual_per_pulsar: list,
    n_basis_per_k: tuple[int, ...],
    n_psr: int,
) -> Float[Array, " n_joint"]:
    """Assemble joint ``basis_proj_residual_joint`` in (k, p, b) layout.

    Each per-pulsar slab ``basis_proj_residual_per_pulsar[p] = F_stack_pᵀ
    C_p⁻¹ r_p`` is a length-``n_basis_total`` vector in (k, b) layout.
    Slicing it by ``local_slices[k]`` gives ``F_{k,p}ᵀ C_p⁻¹ r_p``, pulsar
    ``p``'s projection onto injector ``k``'s basis.

    The output is the flat (k, p, b) layout: outer concat over ``k``
    (injector), middle concat over ``p`` (pulsar), inner entries are the
    ``n_basis_k`` basis components.  Matches the layout of
    :func:`_phi_and_phi_inv_joint` so that ``basis_proj_residual_joint``
    can be solved against ``Σ_joint``.
    """
    local_slices = []
    offset = 0
    for nb in n_basis_per_k:
        local_slices.append(slice(offset, offset + nb))
        offset += nb
    return jnp.concatenate(
        [
            jnp.concatenate(
                [
                    basis_proj_residual_per_pulsar[p][local_slices[k]]
                    for p in range(n_psr)
                ]
            )
            for k in range(len(n_basis_per_k))
        ]
    )


def _logdet_from_cho(cf) -> Float[Array, ""]:
    """``log|A|`` from a ``cho_factor`` output ``cf`` (``= 2 sum log diag L``).

    Uses the Cholesky diagonal rather than ``slogdet``: the sign branch of
    ``slogdet`` is non-smooth and NaNs out 2nd-order autodiff even when the
    sign is constantly +1 (matches ``utils.py`` and the correlated
    ``pta_logL`` tail).
    """
    return 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(cf[0]))))


class JointCorrelatedBlocks(NamedTuple):
    r"""Shared joint outer-tier blocks for the correlated-signal system.

    The substrate both the correlated :func:`pta_logL` branch and
    :func:`~jaxpint.pta.conditional_gwb` build on: each pulsar's
    inner-tier Woodbury projections summed / stacked into the joint ``(k, p, b)``
    layout, plus the joint prior and its inverse.  ``pta_logL`` forms the
    marginal likelihood from these; the conditional posterior reads
    ``Phi_joint_inv`` / ``basis_overlap_joint`` as its precision and
    ``basis_proj_residual_joint`` as its right-hand side.  They differ *only* in
    what they do with these blocks — so the (injector, pulsar, basis)/(k,p,b)
    layout contract lives in one place (``joint_correlated_blocks``) rather than being duplicated.

    Attributes
    ----------
    sum_rCr : scalar
        ``\Sigma_p r_p^T C_p^{-1} r_p`` — each pulsar's noise ``C_p`` *excludes* the
        correlated signal.
    sum_logdetC : scalar
        ``\Sigma_p log|C_p|``.
    basis_proj_residual_joint : (n_joint,) array
        ``stack_p(F_p^T C_p^{-1} r_p)`` in (k, p, b) layout.
    basis_overlap_joint : (n_joint, n_joint) array
        ``blockdiag_p(F_p^T C_p^{-1} F_p)`` in (k, p, b) layout.
    Phi_joint_inv : (n_joint, n_joint) array
        Inverse joint prior (``kron(\Gamma_k^{-1}, diag(1/S_k))`` per
        injector) — the precision both consumers build ``\Sigma`` from.
    logdet_Phi_joint : scalar
        ``log|\Phi_\mathrm{joint}|`` with ``\Phi_\mathrm{joint} =
        blockdiag_k(\Gamma_k \otimes diag(S_k))``.  Precomputed once here (the
        prior depends only on ``global_params``); both the marginal
        ``pta_logL`` and ``pta_clogL`` need it, and the dense ``Phi_joint``
        itself has no other consumer, so only this scalar is carried.
    """

    sum_rCr: Float[Array, ""]
    sum_logdetC: Float[Array, ""]
    basis_proj_residual_joint: Float[Array, " n_joint"]
    basis_overlap_joint: Float[Array, "n_joint n_joint"]
    Phi_joint_inv: Float[Array, "n_joint n_joint"]
    logdet_Phi_joint: Float[Array, ""]


def joint_correlated_blocks(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> JointCorrelatedBlocks:
    """Assemble the shared joint outer-tier blocks for the correlated system.

    Runs each pulsar's inner-tier Woodbury solve against the stacked
    correlated-signal basis, accumulates the residual quadratic form ``sum_rCr``
    and noise log-determinant ``sum_logdetC``, and assembles the per-pulsar
    ``(k, b)`` slabs plus the joint prior into the ``(k, p, b)`` joint layout.

    Both the correlated :func:`pta_logL` branch and
    :func:`~jaxpint.pta.conditional_gwb` call this and then diverge:
    ``pta_logL`` takes the logdets and the Cholesky quadratic correction, the
    conditional takes the precision and RHS.  ``config.correlated_injectors``
    must be non-empty (both callers guard this upstream).
    """
    n_psr = config.n_pulsars
    n_basis_per_k = n_basis_per_injector(
        config.correlated_injectors, config.toa_data_list[0]
    )

    sum_rCr = jnp.float64(0.0)
    sum_logdetC = jnp.float64(0.0)
    # Per-pulsar slabs in (k, b) layout from the inner tier.
    basis_proj_residual_per_pulsar = []  # each (n_basis_total,)
    basis_overlap_per_pulsar = []  # each (n_basis_total, n_basis_total)

    for p in range(n_psr):
        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p,
            config.toa_data_list[p],
            pulsar_params[p],
            global_params,
            config.signal_injectors,
        )
        F_stack_p = _stacked_fourier_basis(
            config.correlated_injectors, config.toa_data_list[p]
        )
        (
            rCr_p,
            logdetC_p,
            basis_proj_residual_p,
            basis_overlap_p,
        ) = _per_pulsar_intermediates(
            config.toa_data_list[p],
            config.timing_models[p],
            config.noise_models[p],
            pulsar_params[p],
            F_stack_p,
            external_delay=ext_delay,
            external_cov=ext_cov,
        )
        sum_rCr = sum_rCr + rCr_p
        sum_logdetC = sum_logdetC + logdetC_p
        basis_proj_residual_per_pulsar.append(basis_proj_residual_p)
        basis_overlap_per_pulsar.append(basis_overlap_p)

    # Assemble the joint outer-tier system directly in (k, p, b) layout.
    Phi_joint, Phi_joint_inv = _phi_and_phi_inv_joint(
        config.correlated_injectors, global_params
    )
    # log|Phi_joint| is the prior's only remaining use, and both the marginal
    # and conditional likelihoods need it — compute it once here rather than
    # re-factoring the dense Phi_joint in each consumer.
    logdet_Phi_joint = _logdet_from_cho(jax.scipy.linalg.cho_factor(Phi_joint))
    basis_overlap_joint = _assemble_basis_overlap_joint_kpb(
        basis_overlap_per_pulsar, n_basis_per_k, n_psr
    )
    basis_proj_residual_joint = _assemble_basis_proj_residual_joint_kpb(
        basis_proj_residual_per_pulsar, n_basis_per_k, n_psr
    )
    return JointCorrelatedBlocks(
        sum_rCr=sum_rCr,
        sum_logdetC=sum_logdetC,
        basis_proj_residual_joint=basis_proj_residual_joint,
        basis_overlap_joint=basis_overlap_joint,
        Phi_joint_inv=Phi_joint_inv,
        logdet_Phi_joint=logdet_Phi_joint,
    )


# ---------------------------------------------------------------------------
# Marginal / conditional likelihoods from the shared joint blocks
# ---------------------------------------------------------------------------
#
# Both the marginal correlated ``pta_logL`` and the coefficient-conditioned
# ``pta_clogL`` are thin contractions of the *same* ``JointCorrelatedBlocks``
# (the expensive part is ``joint_correlated_blocks`` — the per-pulsar Woodbury
# solves).  Keeping them as two helpers over one ``blk`` lets ``pta_logL_and_clogL``
# hand back both from a single block computation, so a Gibbs/HMC step that
# alternates marginal (hyperparameter) and conditional (coefficient) moves does
# not pay for the inner tier twice.


def _marginal_logL_from_blocks(
    blk: JointCorrelatedBlocks,
    n_toas_total: int,
) -> Float[Array, ""]:
    r"""Marginal correlated log-likelihood from the shared joint blocks.

    Integrates the correlated coefficients out via the outer-tier Woodbury
    solve: with ``\Sigma = \Phi_\mathrm{joint}^{-1} + F^T C^{-1} F`` and
    ``correction = (F^T C^{-1} r)^T \Sigma^{-1} (F^T C^{-1} r)``,

    .. math::

        \mathrm{logL} = -\tfrac12(\mathrm{sum\_rCr} - \mathrm{correction})
          - \tfrac12(\mathrm{sum\_logdetC} + \log|\Phi_\mathrm{joint}| + \log|\Sigma|)
          - \tfrac12 n_\mathrm{toas} \log 2\pi.
    """
    Sigma_joint = blk.Phi_joint_inv + blk.basis_overlap_joint
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_joint)
    correction = jnp.dot(
        blk.basis_proj_residual_joint,
        jax.scipy.linalg.cho_solve(Sigma_cf, blk.basis_proj_residual_joint),
    )
    logdet_Sigma_joint = _logdet_from_cho(Sigma_cf)
    return (
        -0.5 * (blk.sum_rCr - correction)
        - 0.5 * (blk.sum_logdetC + blk.logdet_Phi_joint + logdet_Sigma_joint)
        - 0.5 * n_toas_total * jnp.log(2.0 * jnp.pi)
    )


def _clogL_from_blocks(
    blk: JointCorrelatedBlocks,
    coefficients: Float[Array, " n_joint"],
    n_toas_total: int,
) -> Float[Array, ""]:
    r"""Correlated log-likelihood *conditioned* on explicit coefficients.

    The density counterpart of :func:`_marginal_logL_from_blocks`: instead
    of integrating the coefficients out it holds ``a = coefficients`` fixed
    and evaluates the joint Gaussian density.  Using the expanded quadratic
    form (the data term runs against the per-pulsar Woodbury ``C_p``):

    .. math::

        \mathrm{clogL} =
          -\tfrac12\big(\mathrm{sum\_rCr}
            - 2\, a\!\cdot\!(F^T C^{-1} r)
            + a\!\cdot\!(F^T C^{-1} F)\,a\big)
          - \tfrac12 \mathrm{sum\_logdetC} - \tfrac12 n_\mathrm{toas}\log 2\pi
          - \tfrac12 a\!\cdot\!\Phi_\mathrm{joint}^{-1} a
          - \tfrac12 \log|\Phi_\mathrm{joint}| - \tfrac12 n_\mathrm{joint}\log 2\pi.

    ``coefficients`` follow the (k, p, b) layout of
    :func:`~jaxpint.pta.conditional_gwb`'s ``mean``.  Because
    ``basis_overlap_joint`` is block-diagonal over pulsars, the quadratic
    ``a·(basis_overlap_joint a)`` is exactly ``\Sigma_p a_p^T (F_p^T C_p^{-1}
    F_p) a_p``, i.e. the per-pulsar data misfit.
    """
    a = coefficients
    _validate_coefficient_length(
        a.shape[0],
        blk.basis_proj_residual_joint.shape[0],
        producer="pta_clogL",
        canonical="conditional_gwb",
        system="correlated system",
    )
    log2pi = jnp.log(2.0 * jnp.pi)
    data_quad = (
        blk.sum_rCr
        - 2.0 * jnp.dot(a, blk.basis_proj_residual_joint)
        + jnp.dot(a, blk.basis_overlap_joint @ a)
    )
    prior_quad = jnp.dot(a, blk.Phi_joint_inv @ a)
    n_joint = a.shape[0]
    return (
        -0.5 * data_quad
        - 0.5 * blk.sum_logdetC
        - 0.5 * n_toas_total * log2pi
        - 0.5 * prior_quad
        - 0.5 * blk.logdet_Phi_joint
        - 0.5 * n_joint * log2pi
    )


# ---------------------------------------------------------------------------
# PTA log-likelihood
# ---------------------------------------------------------------------------


def single_pulsar_pta_logL(
    p: int,
    global_params: GlobalParams,
    pulsar_params_p: ParameterVector,
    config: PTAConfig,
) -> Float[Array, ""]:
    """Per-pulsar log-likelihood with signal injections.

    Collects delay and covariance contributions from every
    :class:`SignalInjector` in *config* for pulsar ``p``, then delegates to
    :func:`jaxpint.likelihood.single_pulsar_logL`. Returns the scalar
    contribution of pulsar ``p`` to ``pta_logL``; summing over ``p``
    reproduces ``pta_logL`` exactly (the uncorrelated case).

    Used as the per-pulsar primitive by ``jaxpint.pta.scan.scan_logL``,
    which exploits the per-pulsar decomposition to avoid recomputing
    contributions whose params don't vary along any scan axis.

    Parameters
    ----------
    p : int
        Pulsar index within the PTA. Passed to each
        :class:`SignalInjector`'s ``delay`` / ``covariance`` methods so
        per-pulsar dispatch is consistent with :func:`pta_logL`.
    global_params : GlobalParams
        Shared parameters.
    pulsar_params_p : ParameterVector
        Pulsar ``p``'s timing/noise parameters (i.e. ``pulsar_params[p]``).
    config : PTAConfig
        Static configuration; only the ``p``-th element of
        ``toa_data_list``, ``timing_models``, ``noise_models`` and the
        ``signal_injectors`` tuple are read.

    Returns
    -------
    logL_p : scalar
        Pulsar ``p``'s contribution to the PTA log-likelihood.
    """
    toa_data_p = config.toa_data_list[p]
    timing_model_p = config.timing_models[p]
    noise_model_p = config.noise_models[p]

    ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
        p, toa_data_p, pulsar_params_p, global_params, config.signal_injectors
    )

    return single_pulsar_logL(
        toa_data_p,
        timing_model_p,
        noise_model_p,
        pulsar_params_p,
        external_delay=ext_delay,
        external_cov=ext_cov,
    )


def precompute_single_pulsar_pta_factor(
    p: int,
    global_params: GlobalParams,
    pulsar_params_p: ParameterVector,
    config: PTAConfig,
) -> WoodburyFactor:
    """Precompute pulsar ``p``'s Woodbury factor for the PTA likelihood.

    Captures the noise-side computation (Cholesky of ``Σ`` and the
    constant ``log det C``) including any per-pulsar covariance
    contributions from stochastic signal injectors. Pair with
    :func:`single_pulsar_pta_logL_with_factor` to evaluate the
    likelihood at varying timing-domain parameters without redoing the
    factorization.

    The factor is valid as long as ``noise_model.covariance(toa_data,
    params)`` and every injector's ``covariance(p, ...)`` return the
    same arrays for the values of ``params`` and ``global_params`` that
    the apply call uses. In practice this means: do not vary
    noise-model params (``EFAC``, ``EQUAD``, ``ECORR``, ``TNREDAMP``,
    ``TNREDGAM``, etc.) or any global parameter that a stochastic
    injector reads, between precompute and apply. Timing-domain
    parameters (``F0``, ``RAJ``, ``DECJ``, ``PX``, ``DM``, ...) and
    deterministic-injector globals (``cw_log10_h``, etc.) are safe to
    vary.

    Parameters
    ----------
    p
        Pulsar index. Used for static dispatch into ``config``'s
        per-pulsar lists and as the ``p`` argument to each injector's
        ``covariance(p, ...)`` call.
    global_params, pulsar_params_p, config
        Same semantics as :func:`single_pulsar_pta_logL`.
    """
    toa_data_p = config.toa_data_list[p]
    noise_model_p = config.noise_models[p]
    ext_cov = _collect_injector_ext_cov(
        p,
        toa_data_p,
        pulsar_params_p,
        global_params,
        config.signal_injectors,
    )
    return precompute_single_pulsar_factor(
        toa_data_p,
        noise_model_p,
        pulsar_params_p,
        external_cov=ext_cov,
    )


def single_pulsar_pta_logL_with_factor(
    p: int,
    global_params: GlobalParams,
    pulsar_params_p: ParameterVector,
    factor: WoodburyFactor,
    config: PTAConfig,
) -> Float[Array, ""]:
    """Per-pulsar PTA log-likelihood using a precomputed Woodbury factor.

    Functionally equivalent to :func:`single_pulsar_pta_logL` for the
    same configuration, but skips the per-call Cholesky factorization
    of the noise covariance — that work is replaced by a single
    ``cho_solve`` against the precomputed factor.

    The deterministic-delay path (``inj.delay(p, ...)``) is still run
    each call, so signals like CW that perturb the residuals are
    handled correctly. Stochastic-covariance contributions
    (``inj.covariance(p, ...)``) are baked into the factor and assumed
    unchanged.

    See :func:`precompute_single_pulsar_pta_factor` for the contract on
    when the factor is valid.
    """
    toa_data_p = config.toa_data_list[p]
    timing_model_p = config.timing_models[p]
    ext_delay = _collect_injector_ext_delay(
        p,
        toa_data_p,
        pulsar_params_p,
        global_params,
        config.signal_injectors,
    )
    return single_pulsar_logL_with_factor(
        toa_data_p,
        timing_model_p,
        factor,
        pulsar_params_p,
        external_delay=ext_delay,
    )


def pta_logL(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> Float[Array, ""]:
    """Multi-pulsar log-likelihood with signal injections.

    For each pulsar, collects delay and covariance contributions from every
    :class:`SignalInjector` in *config*, then delegates to
    :func:`jaxpint.likelihood.single_pulsar_logL`.  When
    ``config.correlated_injectors`` is non-empty, an additional cross-pulsar
    outer-tier Cholesky solve couples the per-pulsar Fourier-basis systems
    via each correlated injector's ORF.

    Parameters
    ----------
    global_params : GlobalParams
        Shared parameters (CW source properties, GWB spectrum, etc.).
        This is the first differentiable argument.
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.
        This is the second differentiable argument.
    config : PTAConfig
        Static configuration (TOA data, models, injectors, optional
        correlated injectors).

    Returns
    -------
    logL : scalar
        Sum of per-pulsar log-likelihoods, plus outer-tier corrections
        from any correlated injectors.
    """
    n_psr = config.n_pulsars

    # ---- Fast path: no correlated injectors → sum of independent per-pulsar logL.
    #      Same per-pulsar primitive that scan.scan_logL sums, so the two entry
    #      points stay in lockstep. ----
    if config.correlated_injectors == ():
        total = jnp.float64(0.0)
        for p in range(n_psr):
            total = total + single_pulsar_pta_logL(
                p, global_params, pulsar_params[p], config
            )
        return total

    # ---- Correlated path: the shared joint blocks (same substrate as
    #      conditional_gwb), then ONE outer-tier solve for the marginal logL. ----
    blk = joint_correlated_blocks(global_params, pulsar_params, config)
    n_toas_total = sum(td.n_toas for td in config.toa_data_list)
    return _marginal_logL_from_blocks(blk, n_toas_total)


def pta_clogL(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
    coefficients: Float[Array, " n_joint"],
) -> Float[Array, ""]:
    r"""Multi-pulsar log-likelihood *conditioned* on the correlated coefficients.

    The density dual of the correlated :func:`pta_logL` branch: instead of
    marginalizing the correlated-signal (e.g. GWB) Fourier coefficients it
    holds them fixed at ``coefficients`` and evaluates the joint Gaussian
    density (discovery's ``ArrayLikelihood.clogL``).  Per-pulsar
    uncorrelated processes (red noise, DM, ECORR, …) remain marginalized
    into each ``C_p``; only the *correlated* coefficients are explicit —
    exactly the ones :func:`~jaxpint.pta.conditional_gwb`
    returns a posterior over.

    ``coefficients`` must be in that function's (k, p, b) layout, so a
    :func:`~jaxpint.pta.sample_conditional` draw feeds straight in._logL`).

    Parameters
    ----------
    global_params, pulsar_params, config
        As for :func:`pta_logL`; ``config.correlated_injectors`` must be
        non-empty.
    coefficients : (n_joint,) array
        Correlated-signal coefficients in
        :func:`~jaxpint.pta.conditional_gwb`'s (k, p, b) layout.
        **The canonical way to obtain a valid vector is ``conditional_gwb``**
        — its ``.mean`` or a :func:`~jaxpint.pta.sample_conditional` draw —
        which is always the right length and layout.  To build one from
        scratch, its length is ``conditional_gwb(...).mean.shape[0]``.  A
        mismatched length raises a ``ValueError`` naming the expected size.

    Returns
    -------
    clogL : float
        Joint log-density of the residuals and the supplied coefficients.

    Raises
    ------
    ValueError
        If ``config.correlated_injectors`` is empty (use
        :func:`~jaxpint.likelihood.single_pulsar_clogL` for a purely
        uncorrelated model).
    """
    if not config.correlated_injectors:
        raise ValueError(
            "pta_clogL requires at least one correlated injector; for a "
            "purely uncorrelated model condition per pulsar with "
            "single_pulsar_clogL instead."
        )
    blk = joint_correlated_blocks(global_params, pulsar_params, config)
    n_toas_total = sum(td.n_toas for td in config.toa_data_list)
    return _clogL_from_blocks(blk, coefficients, n_toas_total)


def pta_logL_and_clogL(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
    coefficients: Float[Array, " n_joint"],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Both the marginal ``pta_logL`` and conditional ``pta_clogL`` from one solve.

    Convenience wrapper computing ``joint_correlated_blocks`` **once**
    and contracting it two ways — the marginal (coefficients integrated
    out) and the conditional (coefficients held at ``coefficients``).  The
    expensive per-pulsar inner tier is shared, so this is the right entry
    point for a Gibbs / HMC-within-Gibbs step that needs both at the same
    parameter point.  Equivalent to calling :func:`pta_logL` and
    :func:`pta_clogL` separately, but without recomputing the blocks.

    Parameters
    ----------
    global_params, pulsar_params, config, coefficients
        As for :func:`pta_clogL`.

    Returns
    -------
    (logL, clogL) : tuple of float
        The marginal and coefficient-conditioned log-likelihoods.
    """
    if not config.correlated_injectors:
        raise ValueError(
            "pta_logL_and_clogL requires at least one correlated injector."
        )
    blk = joint_correlated_blocks(global_params, pulsar_params, config)
    n_toas_total = sum(td.n_toas for td in config.toa_data_list)
    return (
        _marginal_logL_from_blocks(blk, n_toas_total),
        _clogL_from_blocks(blk, coefficients, n_toas_total),
    )


# ---------------------------------------------------------------------------
# Optimal-statistic block producer
# ---------------------------------------------------------------------------


class GWBlocks(NamedTuple):
    """Per-pulsar GW-basis projections plus shared spectrum/ORF.

    The canonical block producer the optimal statistic
    (:mod:`jaxpint.frequentist`) and the correlated :func:`pta_logL`
    inner tier share.  For pulsar ``p`` with per-pulsar noise covariance
    ``C_p`` (white + red + ecorr + timing, **excluding** the correlated GWB)
    and the GWB Fourier basis ``F_p``, these are the same objects
    ``_per_pulsar_intermediates`` returns, stacked across pulsars.

    Attributes
    ----------
    basis_proj_residual : (n_psr, n_basis) array
        ``F_pᵀ C_p⁻¹ r_p`` per pulsar (discovery's ``kv``).
    basis_overlap : (n_psr, n_basis, n_basis) array
        ``F_pᵀ C_p⁻¹ F_p`` per pulsar (discovery's ``km``), returned exactly
        symmetric (``0.5 (km + kmᵀ)``): the block is a Gram matrix whose only
        asymmetry is float noise from the ``F.T @ (C⁻¹F)`` product, and the
        Cholesky/eigh consumers downstream (e.g. the GX2 null) require symmetry.
    psd : (n_basis,) array
        Shared GWB PSD diagonal ``Φ`` (``UᵀU = Φ``) at ``global_params``.
    orf_matrix : (n_psr, n_psr) array
        Overlap-reduction (e.g. Hellings-Downs) matrix ``Γ``.
    """

    basis_proj_residual: Float[Array, "n_psr n_basis"]
    basis_overlap: Float[Array, "n_psr n_basis n_basis"]
    psd: Float[Array, " n_basis"]
    orf_matrix: Float[Array, "n_psr n_psr"]


def per_pulsar_gw_blocks(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> GWBlocks:
    """Per-pulsar GW-basis blocks ``(kv, km)`` plus shared ``Φ``, ``Γ``.

    Reuses the inner-tier machinery of the correlated :func:`pta_logL`
    (``_per_pulsar_intermediates``) so the blocks match those the
    correlated likelihood consumes (``km`` is additionally symmetrized; see
    :class:`GWBlocks`).  For each pulsar it forms
    ``kv_p = F_pᵀ C_p⁻¹ r_p`` and ``km_p = F_pᵀ C_p⁻¹ F_p``, where ``C_p`` is
    the per-pulsar noise covariance **excluding** the correlated GWB (the GWB
    enters only as the basis ``F_p`` and, downstream, the shared PSD ``Φ``) —
    exactly discovery's "kernel excluding the GW GP".  Any per-pulsar
    :class:`SignalInjector` contributions in ``config`` (deterministic delays,
    e.g. CW; stochastic covariances, e.g. CURN) are folded into the residual /
    ``C_p`` exactly as in :func:`pta_logL`.

    Pure function of the blocks and traceable, so the noise-marginalized
    optimal statistic is a plain ``jax.vmap`` of the downstream combiner over
    posterior draws.

    Parameters
    ----------
    global_params, pulsar_params, config
        As for :func:`pta_logL`.  ``config.correlated_injectors`` must contain
        exactly one injector (the single-component optimal statistic); ``Φ`` and
        ``Γ`` are read from it.

    Returns
    -------
    GWBlocks
        Stacked per-pulsar ``(kv, km)`` plus the shared ``Φ`` and ``Γ``.

    Raises
    ------
    ValueError
        If ``config.correlated_injectors`` does not contain exactly one injector.
    """
    cinjs = config.correlated_injectors
    if len(cinjs) != 1:
        raise ValueError(
            "per_pulsar_gw_blocks requires exactly one correlated injector "
            f"(the single-component optimal statistic); got {len(cinjs)}. "
            "Multi-correlation OS is not supported."
        )
    (cinj,) = cinjs

    kv_list = []
    km_list = []
    for p in range(config.n_pulsars):
        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p,
            config.toa_data_list[p],
            pulsar_params[p],
            global_params,
            config.signal_injectors,
        )
        F_p = cinj.get_fourier_basis(config.toa_data_list[p])
        _, _, basis_proj_residual_p, basis_overlap_p = _per_pulsar_intermediates(
            config.toa_data_list[p],
            config.timing_models[p],
            config.noise_models[p],
            pulsar_params[p],
            F_p,
            external_delay=ext_delay,
            external_cov=ext_cov,
        )
        kv_list.append(basis_proj_residual_p)
        # km = FᵀC⁻¹F is a Gram matrix (symmetric in exact arithmetic); the
        # F.T @ (C⁻¹F) product is symmetric only to float precision. Return the
        # exactly-symmetric form so downstream Cholesky/eigh paths are stable.
        km_list.append(0.5 * (basis_overlap_p + basis_overlap_p.T))

    return GWBlocks(
        basis_proj_residual=jnp.stack(kv_list),
        basis_overlap=jnp.stack(km_list),
        psd=cinj.get_psd(global_params),
        orf_matrix=cinj.get_orf_matrix(),
    )
