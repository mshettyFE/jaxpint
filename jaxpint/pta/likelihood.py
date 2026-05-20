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

from abc import ABC, abstractmethod
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.fitters import compute_time_residuals
from jaxpint.likelihood import single_pulsar_logL
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import concat_woodbury_blocks, woodbury_dot, woodbury_solve

from jaxpint.pta.params import GlobalParams


# ---------------------------------------------------------------------------
# Signal injector ABC
# ---------------------------------------------------------------------------


class SignalInjector(ABC):
    """Abstract base class for PTA signal components.

    Each injector:

    1. Registers its own parameters into :class:`GlobalParams` via
       :meth:`register_params` (**required** — abstract).
    2. Produces delay arrays and/or covariance ``(U, Phi)`` tuples per
       pulsar via :meth:`delay` / :meth:`covariance` (optional —
       default implementations return ``None``).

    Subclasses must implement :meth:`register_params`.  Override
    :meth:`delay` for deterministic signals (e.g. CW) and/or
    :meth:`covariance` for stochastic signals (e.g. GWB).

    :func:`pta_logL` is agnostic to the signal type.
    """

    @abstractmethod
    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        """Append this signal's parameters to *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Mutable accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this signal's parameters appended.
        """
        ...

    def delay(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[Float[Array, " n_toas"]]:
        """Return deterministic delay for pulsar *p*, or ``None``.

        Override for deterministic signals.  The default returns ``None``
        (no delay contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        (n_toas,) array or None
            Deterministic timing delay in seconds, or ``None`` if this
            injector does not contribute a delay.
        """
        return None

    def covariance(
        self,
        p: int,
        toa_data: TOAData,
        pulsar_params: ParameterVector,
        global_params: GlobalParams,
    ) -> Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ]:
        """Return ``(U, Phi)`` covariance contribution for pulsar *p*, or ``None``.

        Override for stochastic signals.  The default returns ``None``
        (no covariance contribution).

        Parameters
        ----------
        p : int
            Pulsar index within the PTA.
        toa_data : TOAData
            Pulse time-of-arrival data for pulsar *p*.
        pulsar_params : ParameterVector
            Timing and noise parameters for pulsar *p*.
        global_params : GlobalParams
            Shared PTA parameters (CW source properties, GWB spectrum, etc.).

        Returns
        -------
        tuple of ((n_toas, n_basis) array, (n_basis,) array) or None
            Design matrix ``U`` and diagonal PSD vector ``Phi``, or
            ``None`` if this injector does not contribute covariance.
        """
        return None


# ---------------------------------------------------------------------------
# Correlated signal injector ABC
# ---------------------------------------------------------------------------


class CorrelatedSignalInjector(ABC):
    """Abstract base class for cross-pulsar correlated signal components.

    Unlike :class:`SignalInjector`, which produces per-pulsar covariance
    contributions, a ``CorrelatedSignalInjector`` provides the ingredients
    to build a PTA-wide covariance with inter-pulsar correlations: a
    per-pulsar Fourier basis, a global PSD vector, and an overlap reduction
    function (ORF) matrix coupling pulsar pairs.
    """

    @abstractmethod
    def register_params(self, global_params: GlobalParams) -> GlobalParams:
        """Append this signal's parameters to *global_params*.

        Parameters
        ----------
        global_params : GlobalParams
            Accumulator of shared PTA parameters.

        Returns
        -------
        GlobalParams
            Updated copy with this signal's parameters appended.
        """
        ...

    @abstractmethod
    def get_fourier_basis(
        self,
        toa_data: TOAData,
    ) -> Float[Array, "n_toas n_basis"]:
        """Return the Fourier design matrix for a single pulsar.

        Parameters
        ----------
        toa_data : TOAData
            Pulse time-of-arrival data for one pulsar.

        Returns
        -------
        F : (n_toas, n_basis) array
            Fourier design matrix (sin/cos columns).
        """
        ...

    @abstractmethod
    def get_psd(
        self,
        global_params: GlobalParams,
    ) -> Float[Array, " n_basis"]:
        """Return the GWB power spectral density vector.

        Parameters
        ----------
        global_params : GlobalParams
            Shared PTA parameters (amplitude, spectral index, etc.).

        Returns
        -------
        S : (n_basis,) array
            PSD values for each Fourier basis function (sin and cos each
            get the same value for their frequency).
        """
        ...

    @abstractmethod
    def get_orf_matrix(self) -> Float[Array, "n_psr n_psr"]:
        """Return the overlap reduction function matrix.

        The matrix must be invertible (full rank). Rank-deficient ORFs
        such as the monopole (all ones) are not supported by the
        two-tier Woodbury scheme.

        Returns
        -------
        Gamma : (n_psr, n_psr) array
            Symmetric, positive-definite ORF matrix. ``Gamma[a, b]``
            is the correlation coefficient between pulsars *a* and *b*.
        """
        ...


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

    Shared across :func:`pta_logL`, :func:`pta_logL_chunked`, and the
    correlated counterparts so the per-pulsar injector dispatch lives in one
    place.
    """
    delays = [
        inj.delay(p, toa_data, pulsar_params, global_params)
        for inj in signal_injectors
    ]
    delays = [d for d in delays if d is not None]
    ext_delay = sum(delays) if delays else None

    covs = [
        inj.covariance(p, toa_data, pulsar_params, global_params)
        for inj in signal_injectors
    ]
    ext_cov = concat_woodbury_blocks(*covs)

    return ext_delay, ext_cov


# ---------------------------------------------------------------------------
# Per-pulsar intermediates (inner tier for the correlated outer-tier solve)
# ---------------------------------------------------------------------------


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
    # 1. Residuals
    r = compute_time_residuals(timing_model, toa_data, params)
    if external_delay is not None:
        r = r - external_delay

    # 2. Per-pulsar noise covariance (optionally augmented with external_cov)
    Ndiag, U_noise, Phi_noise = noise_model.covariance(toa_data, params)
    U, Phi = concat_woodbury_blocks((U_noise, Phi_noise), external_cov)

    # 3. Inner tier: per-pulsar Woodbury
    rCr_p, logdetC_p = woodbury_dot(Ndiag, U, Phi, r, r)

    # 4. C_p^{-1} r_p and C_p^{-1} F_corr via Woodbury solve
    #    Combine into one solve: B = [r[:, None], F_corr]
    B = jnp.concatenate([r[:, None], F_corr], axis=1)  # (n_toas, 1 + n_basis)
    Cinv_B = woodbury_solve(Ndiag, U, Phi, B)
    Cinv_r = Cinv_B[:, 0]                              # (n_toas,)
    Cinv_F = Cinv_B[:, 1:]                              # (n_toas, n_basis)

    # 5. Project onto the correlated-signal Fourier basis
    basis_proj_residual_p = F_corr.T @ Cinv_r          # (n_basis,)
    basis_overlap_p = F_corr.T @ Cinv_F                # (n_basis, n_basis)

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
#   n_total       = sum_k n_psr * n_basis_k  (size of the joint outer-tier system)
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


def _n_basis_per_injector(
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
        cinj.get_fourier_basis(toa_data_0).shape[1]
        for cinj in correlated_injectors
    )


def _phi_and_phi_inv_joint(
    correlated_injectors: tuple[CorrelatedSignalInjector, ...],
    global_params: GlobalParams,
) -> tuple[Float[Array, "n_total n_total"], Float[Array, "n_total n_total"]]:
    """Build joint ``(Phi_joint, Phi_joint_inv)`` in (k, p, b) ordering.

    ``Phi_joint = blockdiag_k( Γ_k ⊗ diag(S_k) )``.  The K diagonal blocks
    are independent across injectors (different correlated signals have
    independent priors); inside each block, pulsars are coupled via Γ_k
    and basis functions are independent (diagonal in b).

    Each block has shape ``(n_psr * n_basis_k, n_psr * n_basis_k)``.  The
    full matrix has shape ``(n_total, n_total)``.
    """
    Phi_blocks = []
    Phi_inv_blocks = []
    for cinj in correlated_injectors:
        Gamma = cinj.get_orf_matrix()
        S = cinj.get_psd(global_params)
        Phi_blocks.append(jnp.kron(Gamma, jnp.diag(S)))
        Phi_inv_blocks.append(
            jnp.kron(jnp.linalg.inv(Gamma), jnp.diag(1.0 / S))
        )
    return (
        jax.scipy.linalg.block_diag(*Phi_blocks),
        jax.scipy.linalg.block_diag(*Phi_inv_blocks),
    )


def _assemble_basis_overlap_joint_kpb(
    basis_overlap_per_pulsar: list,
    n_basis_per_k: tuple[int, ...],
    n_psr: int,
) -> Float[Array, "n_total n_total"]:
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
    ``(n_total, n_total)`` and matches :func:`_phi_and_phi_inv_joint`'s
    (k, p, b) layout so that ``Σ_joint = Phi_joint_inv + basis_overlap_joint``
    is a legal addition.
    """
    local_slices = []
    offset = 0
    for nb in n_basis_per_k:
        local_slices.append(slice(offset, offset + nb))
        offset += nb
    K = len(n_basis_per_k)
    return jnp.block([
        [
            jax.scipy.linalg.block_diag(*[
                basis_overlap_per_pulsar[p][local_slices[k_a], local_slices[k_b]]
                for p in range(n_psr)
            ])
            for k_b in range(K)
        ]
        for k_a in range(K)
    ])


def _assemble_basis_proj_residual_joint_kpb(
    basis_proj_residual_per_pulsar: list,
    n_basis_per_k: tuple[int, ...],
    n_psr: int,
) -> Float[Array, " n_total"]:
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
    return jnp.concatenate([
        jnp.concatenate([
            basis_proj_residual_per_pulsar[p][local_slices[k]]
            for p in range(n_psr)
        ])
        for k in range(len(n_basis_per_k))
    ])


# ---------------------------------------------------------------------------
# PTA log-likelihood
# ---------------------------------------------------------------------------


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

    # ---- Shared: per-pulsar SignalInjector contributions ----
    per_pulsar_delays: list[Optional[Float[Array, " n_toas"]]] = []
    per_pulsar_covs: list[
        Optional[tuple[Float[Array, "n_toas n_ext"], Float[Array, " n_ext"]]]
    ] = []
    for p in range(n_psr):
        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p,
            config.toa_data_list[p],
            pulsar_params[p],
            global_params,
            config.signal_injectors,
        )
        per_pulsar_delays.append(ext_delay)
        per_pulsar_covs.append(ext_cov)

    # ---- Fast path: no correlated injectors → sum of independent per-pulsar logL ----
    if config.correlated_injectors == ():
        total = jnp.float64(0.0)
        for p in range(n_psr):
            total += single_pulsar_logL(
                config.toa_data_list[p],
                config.timing_models[p],
                config.noise_models[p],
                pulsar_params[p],
                external_delay=per_pulsar_delays[p],
                external_cov=per_pulsar_covs[p],
            )
        return total

    # ---- Correlated path: ONE joint outer-tier solve over all correlated injectors ----
    n_basis_per_k = _n_basis_per_injector(
        config.correlated_injectors, config.toa_data_list[0]
    )

    sum_rCr = jnp.float64(0.0)
    sum_logdetC = jnp.float64(0.0)
    # Per-pulsar slabs in (k, b) layout from the inner tier.
    basis_proj_residual_per_pulsar = []   # each (n_basis_total,)
    basis_overlap_per_pulsar = []          # each (n_basis_total, n_basis_total)

    for p in range(n_psr):
        F_stack_p = _stacked_fourier_basis(
            config.correlated_injectors, config.toa_data_list[p]
        )
        (
            rCr_p, logdetC_p,
            basis_proj_residual_p, basis_overlap_p,
        ) = _per_pulsar_intermediates(
            config.toa_data_list[p],
            config.timing_models[p],
            config.noise_models[p],
            pulsar_params[p],
            F_stack_p,
            external_delay=per_pulsar_delays[p],
            external_cov=per_pulsar_covs[p],
        )
        sum_rCr = sum_rCr + rCr_p
        sum_logdetC = sum_logdetC + logdetC_p
        basis_proj_residual_per_pulsar.append(basis_proj_residual_p)
        basis_overlap_per_pulsar.append(basis_overlap_p)

    # Assemble the joint outer-tier system directly in (k, p, b) layout.
    Phi_joint, Phi_joint_inv = _phi_and_phi_inv_joint(
        config.correlated_injectors, global_params
    )
    basis_overlap_joint = _assemble_basis_overlap_joint_kpb(
        basis_overlap_per_pulsar, n_basis_per_k, n_psr,
    )
    basis_proj_residual_joint = _assemble_basis_proj_residual_joint_kpb(
        basis_proj_residual_per_pulsar, n_basis_per_k, n_psr,
    )

    Sigma_joint = Phi_joint_inv + basis_overlap_joint
    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_joint)
    correction = jnp.dot(
        basis_proj_residual_joint,
        jax.scipy.linalg.cho_solve(Sigma_cf, basis_proj_residual_joint),
    )

    _, logdet_Phi_joint = jnp.linalg.slogdet(Phi_joint)
    logdet_Sigma_joint = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL = (
        -0.5 * (sum_rCr - correction)
        - 0.5 * (sum_logdetC + logdet_Phi_joint + logdet_Sigma_joint)
        - 0.5 * n_total * jnp.log(2.0 * jnp.pi)
    )
    return total_logL


# ---------------------------------------------------------------------------
# Chunked PTA log-likelihood
# ---------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=("timing_models", "signal_injectors", "p_offset"),
)
def _chunk_logL(
    global_params: GlobalParams,
    pulsar_params_chunk: tuple[ParameterVector, ...],
    toa_data_chunk: tuple[TOAData, ...],
    noise_models_chunk: tuple[NoiseModel, ...],
    timing_models: tuple[TimingModel, ...],
    signal_injectors: tuple[SignalInjector, ...],
    p_offset: int,
) -> Float[Array, ""]:
    """JIT-compiled per-chunk log-likelihood (uncorrelated path).

    Mirrors the uncorrelated path of :func:`pta_logL`, restricted to a
    contiguous chunk of pulsars
    ``[p_offset, p_offset + len(pulsar_params_chunk))``.  Injectors and
    ``timing_models`` are indexed by the *global* pulsar index so each
    injector's per-pulsar dispatch is unchanged.

    Internal helper — callers should use :func:`pta_logL_chunked`.
    """
    total = jnp.float64(0.0)

    for p_local in range(len(pulsar_params_chunk)):
        p_global = p_offset + p_local

        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p_global,
            toa_data_chunk[p_local],
            pulsar_params_chunk[p_local],
            global_params,
            signal_injectors,
        )

        total += single_pulsar_logL(
            toa_data_chunk[p_local],
            timing_models[p_global],
            noise_models_chunk[p_local],
            pulsar_params_chunk[p_local],
            external_delay=ext_delay,
            external_cov=ext_cov,
        )

    return total


@partial(
    jax.jit,
    static_argnames=(
        "timing_models",
        "signal_injectors",
        "correlated_injector",
        "p_offset",
    ),
)
def _chunk_correlated(
    global_params: GlobalParams,
    pulsar_params_chunk: tuple[ParameterVector, ...],
    toa_data_chunk: tuple[TOAData, ...],
    noise_models_chunk: tuple[NoiseModel, ...],
    timing_models: tuple[TimingModel, ...],
    signal_injectors: tuple,
    correlated_injector: CorrelatedSignalInjector,
    p_offset: int,
) -> tuple[
    Float[Array, ""],
    Float[Array, ""],
    Float[Array, " n_chunk_pulsars_x_n_basis"],
    Float[Array, "n_chunk_pulsars_x_n_basis n_chunk_pulsars_x_n_basis"],
]:
    """JIT-compiled per-chunk Woodbury intermediates for the outer tier.

    For a contiguous chunk of pulsars
    ``[p_offset, p_offset + len(pulsar_params_chunk))`` and a single
    ``correlated_injector``, computes:

    - per-pulsar ``SignalInjector`` delays / covariances (CW, CURN, …),
    - per-pulsar Woodbury intermediates ``(rCr_p, logdetC_p,
      basis_proj_residual_p, basis_overlap_p)`` via
      :func:`_per_pulsar_intermediates`,
    - the within-chunk concatenation ``basis_proj_residual_chunk =
      concat(basis_proj_residual_p_for_p_in_chunk)`` and within-chunk
      block-diagonal ``basis_overlap_chunk =
      blockdiag(basis_overlap_p_for_p_in_chunk)``.

    The big ``(n_toas, n_basis)`` Fourier-basis matrices and inner-tier
    Woodbury working memory live only inside this JIT; they are freed
    when the function returns, leaving only the small chunk-sized
    outputs in device memory.

    Internal helper — callers should use :func:`pta_logL_chunked`.
    """
    chunk_sum_rCr = jnp.float64(0.0)
    chunk_sum_logdetC = jnp.float64(0.0)
    basis_proj_residual_list: list[Float[Array, " n_basis"]] = []
    basis_overlap_list: list[Float[Array, "n_basis n_basis"]] = []

    for p_local in range(len(pulsar_params_chunk)):
        p_global = p_offset + p_local

        ext_delay, ext_cov = _collect_per_pulsar_external_inputs(
            p_global,
            toa_data_chunk[p_local],
            pulsar_params_chunk[p_local],
            global_params,
            signal_injectors,
        )

        F_p = correlated_injector.get_fourier_basis(toa_data_chunk[p_local])
        (
            rCr_p, logdetC_p,
            basis_proj_residual_p, basis_overlap_p,
        ) = _per_pulsar_intermediates(
            toa_data_chunk[p_local],
            timing_models[p_global],
            noise_models_chunk[p_local],
            pulsar_params_chunk[p_local],
            F_p,
            external_delay=ext_delay,
            external_cov=ext_cov,
        )
        chunk_sum_rCr = chunk_sum_rCr + rCr_p
        chunk_sum_logdetC = chunk_sum_logdetC + logdetC_p
        basis_proj_residual_list.append(basis_proj_residual_p)
        basis_overlap_list.append(basis_overlap_p)

    basis_proj_residual_chunk = jnp.concatenate(basis_proj_residual_list)
    basis_overlap_chunk = jax.scipy.linalg.block_diag(*basis_overlap_list)
    return (
        chunk_sum_rCr, chunk_sum_logdetC,
        basis_proj_residual_chunk, basis_overlap_chunk,
    )


@partial(jax.jit, static_argnames=("correlated_injector",))
def _finalize_correlated_chunked(
    global_params: GlobalParams,
    correlated_injector: CorrelatedSignalInjector,
    sum_rCr: Float[Array, ""],
    sum_logdetC: Float[Array, ""],
    basis_proj_residual_chunks: tuple,
    basis_overlap_chunks: tuple,
) -> Float[Array, ""]:
    """Cross-pulsar Cholesky reduction over chunk-level intermediates.

    Assembles the full ``(n_psr * n_basis,)`` ``basis_proj_residual_joint``
    and the full ``(n_psr * n_basis, n_psr * n_basis)`` block-diagonal
    ``basis_overlap_joint`` from chunk-level pieces, builds
    ``Sigma_corr = Phi_corr_inv + basis_overlap_joint``, and returns this
    injector's contribution to ``logL``.

    The final dense matrices are small (``n_psr * n_basis`` is typically
    a few thousand) so this step runs once per call as a single JIT.
    """
    S = correlated_injector.get_psd(global_params)
    Gamma = correlated_injector.get_orf_matrix()
    Phi_corr = jnp.kron(Gamma, jnp.diag(S))
    Phi_corr_inv = jnp.kron(jnp.linalg.inv(Gamma), jnp.diag(1.0 / S))

    basis_proj_residual_joint = jnp.concatenate(list(basis_proj_residual_chunks))
    basis_overlap_joint = jax.scipy.linalg.block_diag(*basis_overlap_chunks)
    Sigma_corr = Phi_corr_inv + basis_overlap_joint

    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_corr)
    Sigma_inv_z = jax.scipy.linalg.cho_solve(Sigma_cf, basis_proj_residual_joint)
    correction = jnp.dot(basis_proj_residual_joint, Sigma_inv_z)

    _, logdet_Phi_corr = jnp.linalg.slogdet(Phi_corr)
    logdet_Sigma_corr = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

    contribution = -0.5 * (sum_rCr - correction)
    contribution = contribution - 0.5 * (
        sum_logdetC + logdet_Phi_corr + logdet_Sigma_corr
    )
    return contribution


def pta_logL_chunked(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
    *,
    chunk_size: int,
) -> float:
    """Memory-bounded multi-pulsar log-likelihood.

    Equivalent to :func:`pta_logL` but evaluates pulsars in
    ``chunk_size``-sized batches, with the JIT boundary placed *inside*
    each chunk.  Peak working memory is bounded by the largest chunk
    instead of growing with the total number of pulsars, which lets the
    full NANOGrav 15-yr PTA fit on hardware where wrapping
    :func:`pta_logL` in :func:`jax.jit` (or :func:`jax.hessian`, or the
    nested ``jit(vmap(...))`` inside the sweep helpers) would exceed
    available device memory.

    When ``config.correlated_injectors`` is empty (the default), each
    chunk produces an independent scalar via :func:`_chunk_logL`.  When
    non-empty, per-chunk Woodbury intermediates ``(sum_rCr, sum_logdetC,
    basis_proj_residual_chunk, basis_overlap_chunk)`` are collected via
    :func:`_chunk_correlated` and the cross-pulsar Cholesky reduction
    runs once at the end on the assembled ``(n_psr * n_basis)`` system
    inside :func:`_finalize_correlated_chunked`.

    .. warning::

       Do **not** wrap this function in :func:`jax.jit`,
       :func:`jax.vmap`, :func:`jax.grad`, or :func:`jax.hessian`.
       Doing so reintroduces the single-trace-over-all-pulsars failure
       this function exists to avoid: the outer transformation traces
       through the Python chunk loop and unrolls every chunk into one
       HLO graph.  Returns a Python ``float`` (not a traceable
       ``jax.Array``) to make this contract concrete — differentiable
       callers (Fisher, MCMC) keep using :func:`pta_logL`.

    Parameters
    ----------
    global_params : GlobalParams
        Shared parameters (CW source properties, GWB spectrum, etc.).
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.
    config : PTAConfig
        Static configuration (TOA data, models, injectors, optional
        correlated injectors).
    chunk_size : int
        Number of pulsars per JIT-compiled chunk.  Smaller values bound
        memory more tightly at the cost of more per-chunk compiles
        (each distinct ``(n_toas,)`` signature warms its own cache
        entry).  Must be positive.

    Returns
    -------
    logL : float
        Sum of per-pulsar log-likelihoods, plus outer-tier corrections
        from any correlated injectors.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    n = len(pulsar_params)

    # ---- Fast path: no correlated injectors ----
    if config.correlated_injectors == ():
        total = 0.0
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_total = _chunk_logL(
                global_params,
                pulsar_params[start:end],
                config.toa_data_list[start:end],
                config.noise_models[start:end],
                config.timing_models,
                config.signal_injectors,
                start,
            )
            # `float(...)` forces device->host transfer and blocks until the
            # chunk's compute finishes; without it JAX's async dispatch may
            # pipeline two chunks together and double the peak memory.
            total += float(chunk_total)
        return total

    # ---- Correlated path: accumulate chunk intermediates, finalize once ----
    total_logL = 0.0
    for cinj in config.correlated_injectors:
        sum_rCr = 0.0
        sum_logdetC = 0.0
        basis_proj_residual_chunks: list = []
        basis_overlap_chunks: list = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            (
                chunk_rCr, chunk_logdetC,
                basis_proj_residual_chunk, basis_overlap_chunk,
            ) = _chunk_correlated(
                global_params,
                pulsar_params[start:end],
                config.toa_data_list[start:end],
                config.noise_models[start:end],
                config.timing_models,
                config.signal_injectors,
                cinj,
                start,
            )
            # Block before the next chunk starts so the previous chunk's
            # transient (n_toas, n_basis) Fourier-basis matrices and
            # Woodbury workspace are freed before the next chunk allocates.
            jax.block_until_ready((
                chunk_rCr, chunk_logdetC,
                basis_proj_residual_chunk, basis_overlap_chunk,
            ))
            sum_rCr += float(chunk_rCr)
            sum_logdetC += float(chunk_logdetC)
            basis_proj_residual_chunks.append(basis_proj_residual_chunk)
            basis_overlap_chunks.append(basis_overlap_chunk)

        injector_contribution = _finalize_correlated_chunked(
            global_params,
            cinj,
            jnp.float64(sum_rCr),
            jnp.float64(sum_logdetC),
            tuple(basis_proj_residual_chunks),
            tuple(basis_overlap_chunks),
        )
        total_logL += float(injector_contribution)

    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL -= 0.5 * n_total * float(jnp.log(2.0 * jnp.pi))

    return total_logL
