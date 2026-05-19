"""Multi-pulsar PTA log-likelihood.

Composes :func:`jaxpint.likelihood.single_pulsar_logL` across multiple
pulsars, with signal injections (CW, GWB, etc.) mediated by the
:class:`SignalInjector` abstract base class.  Optional cross-pulsar
correlations (e.g. a Hellings-Downs gravitational-wave background) are
modeled by :class:`CorrelatedSignalInjector` instances and handled via a
two-tier Woodbury scheme:

- **Inner tier** (per-pulsar): :func:`~jaxpint.utils.woodbury_dot` /
  :func:`~jaxpint.utils.woodbury_solve` evaluate the per-pulsar Gaussian
  likelihood without forming the full covariance.
- **Outer tier** (cross-pulsar, only when ``config.correlated_injectors``
  is non-empty): a dense Cholesky solve on the compressed Fourier-basis
  system couples pulsars via the ORF.

The global covariance is ``C = D + V @ Phi_gwb @ V^T`` where
``D = blockdiag(C_1, ..., C_n)`` is the block-diagonal per-pulsar noise,
``V = blockdiag(F_1, ..., F_n)`` collects per-pulsar Fourier bases, and
``Phi_gwb = Gamma kron diag(S)`` is the ORF-weighted GWB PSD.

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
    F_gwb: Float[Array, "n_toas n_basis"],
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
    quantities projected onto the GWB Fourier basis.

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
    F_gwb : (n_toas, n_basis) array
        GWB Fourier design matrix for this pulsar.
    external_delay : (n_toas,) array, optional
        Deterministic signal delay (e.g. CW).
    external_cov : (U_ext, Phi_ext) tuple, optional
        Per-pulsar stochastic covariance from SignalInjectors (e.g. CURN).

    Returns
    -------
    rCr_p : scalar
        ``r_p^T C_p^{-1} r_p``.
    logdetC_p : scalar
        ``log|C_p|``.
    z_p : (n_basis,) array
        ``F_p^T C_p^{-1} r_p`` — residuals projected onto GWB basis.
    D_p : (n_basis, n_basis) array
        ``F_p^T C_p^{-1} F_p`` — noise-weighted Fourier basis overlap.
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

    # 4. C_p^{-1} r_p and C_p^{-1} F_gwb via Woodbury solve
    #    Combine into one solve: B = [r[:, None], F_gwb]
    B = jnp.concatenate([r[:, None], F_gwb], axis=1)  # (n_toas, 1 + n_basis)
    Cinv_B = woodbury_solve(Ndiag, U, Phi, B)
    Cinv_r = Cinv_B[:, 0]                              # (n_toas,)
    Cinv_F = Cinv_B[:, 1:]                              # (n_toas, n_basis)

    # 5. Project onto GWB Fourier basis
    z_p = F_gwb.T @ Cinv_r                              # (n_basis,)
    D_p = F_gwb.T @ Cinv_F                              # (n_basis, n_basis)

    return rCr_p, logdetC_p, z_p, D_p


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

    # ---- Correlated path: outer-tier ORF/GWB solve per correlated injector ----
    total_logL = jnp.float64(0.0)
    for cinj in config.correlated_injectors:
        S = cinj.get_psd(global_params)            # (n_basis,)
        Gamma = cinj.get_orf_matrix()              # (n_psr, n_psr)

        # Phi_gwb is (n_psr * n_basis, n_psr * n_basis)
        Phi_gwb = jnp.kron(Gamma, jnp.diag(S))
        Phi_gwb_inv = jnp.kron(
            jnp.linalg.inv(Gamma),
            jnp.diag(1.0 / S),
        )

        z_list = []
        D_list = []
        sum_rCr = jnp.float64(0.0)
        sum_logdetC = jnp.float64(0.0)

        for p in range(n_psr):
            F_p = cinj.get_fourier_basis(config.toa_data_list[p])
            rCr_p, logdetC_p, z_p, D_p = _per_pulsar_intermediates(
                config.toa_data_list[p],
                config.timing_models[p],
                config.noise_models[p],
                pulsar_params[p],
                F_p,
                external_delay=per_pulsar_delays[p],
                external_cov=per_pulsar_covs[p],
            )
            sum_rCr = sum_rCr + rCr_p
            sum_logdetC = sum_logdetC + logdetC_p
            z_list.append(z_p)
            D_list.append(D_p)

        # Outer tier: Sigma_gwb = Phi_gwb^{-1} + blockdiag(D_1, ..., D_n)
        z = jnp.concatenate(z_list)                    # (n_psr * n_basis,)
        D_block = jax.scipy.linalg.block_diag(*D_list)
        Sigma_gwb = Phi_gwb_inv + D_block

        Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_gwb)
        Sigma_inv_z = jax.scipy.linalg.cho_solve(Sigma_cf, z)
        correction = jnp.dot(z, Sigma_inv_z)

        _, logdet_Phi_gwb = jnp.linalg.slogdet(Phi_gwb)
        logdet_Sigma_gwb = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

        total_logL = total_logL - 0.5 * (sum_rCr - correction)
        total_logL = total_logL - 0.5 * (
            sum_logdetC + logdet_Phi_gwb + logdet_Sigma_gwb
        )

    # Constant term
    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL = total_logL - 0.5 * n_total * jnp.log(2.0 * jnp.pi)

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
    - per-pulsar Woodbury intermediates ``(rCr_p, logdetC_p, z_p, D_p)``
      via :func:`_per_pulsar_intermediates`,
    - the within-chunk concatenation ``z_chunk = concat(z_p_for_p_in_chunk)``
      and within-chunk block-diagonal
      ``D_chunk = blockdiag(D_p_for_p_in_chunk)``.

    The big ``(n_toas, n_basis)`` Fourier-basis matrices and inner-tier
    Woodbury working memory live only inside this JIT; they are freed
    when the function returns, leaving only the small chunk-sized
    outputs in device memory.

    Internal helper — callers should use :func:`pta_logL_chunked`.
    """
    chunk_sum_rCr = jnp.float64(0.0)
    chunk_sum_logdetC = jnp.float64(0.0)
    z_list: list[Float[Array, " n_basis"]] = []
    D_list: list[Float[Array, "n_basis n_basis"]] = []

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
        rCr_p, logdetC_p, z_p, D_p = _per_pulsar_intermediates(
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
        z_list.append(z_p)
        D_list.append(D_p)

    z_chunk = jnp.concatenate(z_list)
    D_chunk = jax.scipy.linalg.block_diag(*D_list)
    return chunk_sum_rCr, chunk_sum_logdetC, z_chunk, D_chunk


@partial(jax.jit, static_argnames=("correlated_injector",))
def _finalize_correlated_chunked(
    global_params: GlobalParams,
    correlated_injector: CorrelatedSignalInjector,
    sum_rCr: Float[Array, ""],
    sum_logdetC: Float[Array, ""],
    z_chunks: tuple,
    D_chunks: tuple,
) -> Float[Array, ""]:
    """Cross-pulsar Cholesky reduction over chunk-level intermediates.

    Assembles the full ``(n_psr * n_basis,)`` ``z`` and the full
    ``(n_psr * n_basis, n_psr * n_basis)`` block-diagonal ``D`` from
    chunk-level pieces, builds ``Sigma_gwb = Phi_gwb_inv + D``, and
    returns this injector's contribution to ``logL``.

    The final dense matrices are small (``n_psr * n_basis`` is typically
    a few thousand) so this step runs once per call as a single JIT.
    """
    S = correlated_injector.get_psd(global_params)
    Gamma = correlated_injector.get_orf_matrix()
    Phi_gwb = jnp.kron(Gamma, jnp.diag(S))
    Phi_gwb_inv = jnp.kron(jnp.linalg.inv(Gamma), jnp.diag(1.0 / S))

    z = jnp.concatenate(list(z_chunks))
    D_block = jax.scipy.linalg.block_diag(*D_chunks)
    Sigma_gwb = Phi_gwb_inv + D_block

    Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_gwb)
    Sigma_inv_z = jax.scipy.linalg.cho_solve(Sigma_cf, z)
    correction = jnp.dot(z, Sigma_inv_z)

    _, logdet_Phi_gwb = jnp.linalg.slogdet(Phi_gwb)
    logdet_Sigma_gwb = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

    contribution = -0.5 * (sum_rCr - correction)
    contribution = contribution - 0.5 * (
        sum_logdetC + logdet_Phi_gwb + logdet_Sigma_gwb
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
    z_chunk, D_chunk)`` are collected via :func:`_chunk_correlated` and
    the cross-pulsar Cholesky reduction runs once at the end on the
    assembled ``(n_psr * n_basis)`` system inside
    :func:`_finalize_correlated_chunked`.

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
        z_chunks: list = []
        D_chunks: list = []

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_rCr, chunk_logdetC, z_chunk, D_chunk = _chunk_correlated(
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
            jax.block_until_ready((chunk_rCr, chunk_logdetC, z_chunk, D_chunk))
            sum_rCr += float(chunk_rCr)
            sum_logdetC += float(chunk_logdetC)
            z_chunks.append(z_chunk)
            D_chunks.append(D_chunk)

        injector_contribution = _finalize_correlated_chunked(
            global_params,
            cinj,
            jnp.float64(sum_rCr),
            jnp.float64(sum_logdetC),
            tuple(z_chunks),
            tuple(D_chunks),
        )
        total_logL += float(injector_contribution)

    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL -= 0.5 * n_total * float(jnp.log(2.0 * jnp.pi))

    return total_logL
