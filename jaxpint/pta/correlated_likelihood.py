"""Correlated (cross-pulsar) PTA log-likelihood.

Extends the per-pulsar likelihood with inter-pulsar correlations induced by
a gravitational wave background (GWB) via overlap reduction functions (ORFs).

Uses a two-tier Woodbury scheme:

- **Inner tier** (per-pulsar): existing :func:`~jaxpint.utils.woodbury_dot`
  and :func:`~jaxpint.utils.woodbury_solve` handle white + correlated noise.
- **Outer tier** (cross-pulsar): a dense Cholesky solve on the compressed
  Fourier-basis system couples pulsars via the ORF.

The global covariance is ``C = D + V @ Phi_gwb @ V^T`` where:

- ``D = blockdiag(C_1, ..., C_n)`` per-pulsar noise
- ``V = blockdiag(F_1, ..., F_n)`` per-pulsar Fourier bases
- ``Phi_gwb = Gamma kron diag(S)`` ORF-weighted GWB PSD

References
----------
.. [cl_vh09] van Haasteren et al. (2009), MNRAS 395, 1005.
.. [cl_vh14] van Haasteren & Vallisneri (2014), PRD 90, 104012.
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
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import concat_woodbury_blocks, woodbury_dot, woodbury_solve

from jaxpint.pta.params import GlobalParams
from jaxpint.pta.likelihood import SignalInjector


# ---------------------------------------------------------------------------
# Correlated signal injector ABC
# ---------------------------------------------------------------------------


class CorrelatedSignalInjector(ABC):
    """Abstract base class for cross-pulsar correlated signal components.

    Unlike :class:`~jaxpint.pta.likelihood.SignalInjector`, which produces
    per-pulsar covariance contributions, a ``CorrelatedSignalInjector``
    provides the ingredients to build a PTA-wide covariance matrix with
    inter-pulsar correlations.

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
# Configuration
# ---------------------------------------------------------------------------


class CorrelatedPTAConfig(eqx.Module):
    """Configuration for the correlated PTA likelihood.

    Extends :class:`~jaxpint.pta.likelihood.PTAConfig` with a
    ``correlated_injectors`` field for cross-pulsar signals.

    ``toa_data_list`` and ``noise_models`` are *dynamic* (traced) fields;
    marking them static balloons jit memory because the per-pulsar arrays
    get baked into the compiled HLO. The remaining structural fields are
    compile-time constants.
    """

    toa_data_list: tuple[TOAData, ...]
    noise_models: tuple[NoiseModel, ...]
    timing_models: tuple[TimingModel, ...] = eqx.field(static=True)
    signal_injectors: tuple[SignalInjector, ...] = eqx.field(static=True)
    correlated_injectors: tuple[CorrelatedSignalInjector, ...] = eqx.field(
        static=True
    )

    def __post_init__(self):
        n_toa = len(self.toa_data_list)
        n_tm = len(self.timing_models)
        n_nm = len(self.noise_models)
        if not (n_toa == n_tm == n_nm):
            raise ValueError(
                f"Mismatched pulsar counts: {n_toa} TOA datasets, "
                f"{n_tm} timing models, {n_nm} noise models."
            )

    @property
    def n_pulsars(self) -> int:
        return len(self.toa_data_list)


# ---------------------------------------------------------------------------
# Per-pulsar intermediates (inner tier)
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
# Correlated PTA log-likelihood (outer tier)
# ---------------------------------------------------------------------------


def pta_logL_correlated(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: CorrelatedPTAConfig,
) -> Float[Array, ""]:
    """Multi-pulsar log-likelihood with cross-pulsar correlations.

    Implements a two-tier Woodbury scheme:

    1. Per-pulsar (inner tier): compute ``C_p^{-1} r_p``, ``C_p^{-1} F_p``,
       and per-pulsar log-likelihood contributions.
    2. Cross-pulsar (outer tier): assemble and solve the compressed
       ``Sigma_gwb`` system to account for ORF-mediated correlations.

    Parameters
    ----------
    global_params : GlobalParams
        Shared parameters (GWB amplitude/spectral index, CW source, etc.).
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.
    config : CorrelatedPTAConfig
        Static configuration.

    Returns
    -------
    logL : scalar
        Log-likelihood value.
    """
    n_psr = config.n_pulsars

    # ---- Collect per-pulsar SignalInjector contributions (same as pta_logL) ----
    per_pulsar_delays: list[Optional[Float[Array, " n_toas"]]] = []
    per_pulsar_covs: list[
        Optional[tuple[Float[Array, "n_toas n_ext"], Float[Array, " n_ext"]]]
    ] = []

    for p in range(n_psr):
        delays = [
            inj.delay(
                p,
                config.toa_data_list[p],
                pulsar_params[p],
                global_params,
            )
            for inj in config.signal_injectors
        ]
        delays = [d for d in delays if d is not None]
        per_pulsar_delays.append(sum(delays) if delays else None)

        covs = [
            inj.covariance(
                p,
                config.toa_data_list[p],
                pulsar_params[p],
                global_params,
            )
            for inj in config.signal_injectors
        ]
        per_pulsar_covs.append(concat_woodbury_blocks(*covs))

    # ---- Process each CorrelatedSignalInjector ----
    total_logL = jnp.float64(0.0)

    for cinj in config.correlated_injectors:
        # Get PSD and ORF matrix
        S = cinj.get_psd(global_params)            # (n_basis,)
        Gamma = cinj.get_orf_matrix()              # (n_psr, n_psr)
        n_basis = S.shape[0]

        # Build Phi_gwb = Gamma kron diag(S) and its inverse
        # Phi_gwb is (n_psr * n_basis, n_psr * n_basis)
        Phi_gwb = jnp.kron(Gamma, jnp.diag(S))
        Phi_gwb_inv = jnp.kron(
            jnp.linalg.inv(Gamma),
            jnp.diag(1.0 / S),
        )

        # Per-pulsar intermediates
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

        # Assemble outer tier
        z = jnp.concatenate(z_list)  # (n_psr * n_basis,)

        # Sigma_gwb = Phi_gwb^{-1} + blockdiag(D_1, ..., D_n)
        D_block = jax.scipy.linalg.block_diag(*D_list)  # (n_psr*n_basis, n_psr*n_basis)
        Sigma_gwb = Phi_gwb_inv + D_block

        # Cholesky solve
        Sigma_cf = jax.scipy.linalg.cho_factor(Sigma_gwb)
        Sigma_inv_z = jax.scipy.linalg.cho_solve(Sigma_cf, z)
        correction = jnp.dot(z, Sigma_inv_z)

        # Log-determinants
        _, logdet_Phi_gwb = jnp.linalg.slogdet(Phi_gwb)
        # log|Sigma| from Cholesky: 2 * sum(log(diag(L)))
        logdet_Sigma_gwb = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

        # Accumulate this injector's contribution
        total_logL = total_logL - 0.5 * (sum_rCr - correction)
        total_logL = total_logL - 0.5 * (
            sum_logdetC + logdet_Phi_gwb + logdet_Sigma_gwb
        )

    # Constant term
    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL = total_logL - 0.5 * n_total * jnp.log(2.0 * jnp.pi)

    return total_logL


# ---------------------------------------------------------------------------
# Chunked correlated PTA log-likelihood
# ---------------------------------------------------------------------------


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

    Internal helper — callers should use :func:`pta_logL_correlated_chunked`.
    """
    chunk_sum_rCr = jnp.float64(0.0)
    chunk_sum_logdetC = jnp.float64(0.0)
    z_list: list[Float[Array, " n_basis"]] = []
    D_list: list[Float[Array, "n_basis n_basis"]] = []

    for p_local in range(len(pulsar_params_chunk)):
        p_global = p_offset + p_local

        delays = [
            inj.delay(
                p_global,
                toa_data_chunk[p_local],
                pulsar_params_chunk[p_local],
                global_params,
            )
            for inj in signal_injectors
        ]
        delays = [d for d in delays if d is not None]
        ext_delay = sum(delays) if delays else None

        covs = [
            inj.covariance(
                p_global,
                toa_data_chunk[p_local],
                pulsar_params_chunk[p_local],
                global_params,
            )
            for inj in signal_injectors
        ]
        ext_cov = concat_woodbury_blocks(*covs)

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


def pta_logL_correlated_chunked(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: CorrelatedPTAConfig,
    *,
    chunk_size: int,
) -> float:
    """Memory-bounded multi-pulsar correlated log-likelihood.

    Chunked counterpart to :func:`pta_logL_correlated`.  The per-pulsar
    Woodbury intermediates are evaluated in ``chunk_size``-sized batches
    inside :func:`_chunk_correlated`; the cross-pulsar Cholesky reduction
    runs once at the end on the assembled ``(n_psr * n_basis)`` system.

    Peak working memory is bounded by the largest chunk's per-pulsar
    Fourier basis ``(n_toas, n_basis)`` plus the inner-tier Woodbury
    workspace, instead of growing with the total number of pulsars.

    .. warning::

       Do **not** wrap this function in :func:`jax.jit`,
       :func:`jax.vmap`, :func:`jax.grad`, or :func:`jax.hessian` — see
       :func:`pta_logL_chunked` for details.  Returns a Python ``float``.

    Parameters
    ----------
    global_params : GlobalParams
    pulsar_params : tuple of ParameterVector
    config : CorrelatedPTAConfig
    chunk_size : int
        Pulsars per JIT-compiled chunk.  Must be positive.

    Returns
    -------
    logL : float
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    n_psr = config.n_pulsars
    total_logL = 0.0

    for cinj in config.correlated_injectors:
        sum_rCr = 0.0
        sum_logdetC = 0.0
        z_chunks: list = []
        D_chunks: list = []

        for start in range(0, n_psr, chunk_size):
            end = min(start + chunk_size, n_psr)
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
