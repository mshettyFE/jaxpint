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

from typing import Optional, cast

import jax
import jax.numpy as jnp
import equinox as eqx
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from jaxpint.fitters import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import woodbury_dot, woodbury_solve

from jaxpint.types import GlobalParams
from jaxpint.pta.injectors import SignalInjector, CorrelatedSignalInjector


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
    correlated_injectors: tuple[CorrelatedSignalInjector, ...] = eqx.field(static=True)

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


@jaxtyped(typechecker=beartype)
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

    # 2. Per-pulsar noise covariance
    Ndiag, U, Phi = noise_model.covariance(toa_data, params)
    if external_cov is not None:
        U_ext, Phi_ext = external_cov
        U = jnp.concatenate([U, U_ext], axis=1)
        Phi = jnp.concatenate([Phi, Phi_ext])

    # 3. Inner tier: per-pulsar Woodbury
    rCr_p, logdetC_p = woodbury_dot(Ndiag, U, Phi, r, r)

    # 4. C_p^{-1} r_p and C_p^{-1} F_gwb via Woodbury solve
    #    Combine into one solve: B = [r[:, None], F_gwb]
    B = jnp.concatenate([r[:, None], F_gwb], axis=1)  # (n_toas, 1 + n_basis)
    Cinv_B = woodbury_solve(Ndiag, U, Phi, B)
    Cinv_r = Cinv_B[:, 0]  # (n_toas,)
    Cinv_F = Cinv_B[:, 1:]  # (n_toas, n_basis)

    # 5. Project onto GWB Fourier basis
    z_p = F_gwb.T @ Cinv_r  # (n_basis,)
    D_p = F_gwb.T @ Cinv_F  # (n_basis, n_basis)

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
        per_pulsar_delays.append(
            cast(Float[Array, " n_toas"], sum(delays)) if delays else None
        )

        covs = [
            inj.covariance(
                p,
                config.toa_data_list[p],
                pulsar_params[p],
                global_params,
            )
            for inj in config.signal_injectors
        ]
        covs = [c for c in covs if c is not None]
        if covs:
            per_pulsar_covs.append(
                (
                    jnp.concatenate([U for U, _ in covs], axis=1),
                    jnp.concatenate([Phi for _, Phi in covs]),
                )
            )
        else:
            per_pulsar_covs.append(None)

    # ---- Process each CorrelatedSignalInjector ----
    total_logL = jnp.float64(0.0)

    for cinj in config.correlated_injectors:
        S = cinj.get_psd(global_params)  # (n_basis,)
        Gamma = cinj.get_orf_matrix()  # (n_psr, n_psr)

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

        # Log-determinants. Cholesky-diag-log instead of slogdet:
        # slogdet's sign branch is non-smooth and NaNs out the Hessian.
        Phi_gwb_cf = jax.scipy.linalg.cho_factor(Phi_gwb)
        logdet_Phi_gwb = 2.0 * jnp.sum(jnp.log(jnp.abs(jnp.diag(Phi_gwb_cf[0]))))
        # log|Sigma| from Cholesky: 2 * sum(log(diag(L)))
        logdet_Sigma_gwb = 2.0 * jnp.sum(jnp.log(jnp.diag(Sigma_cf[0])))

        total_logL = total_logL - 0.5 * (sum_rCr - correction)
        total_logL = total_logL - 0.5 * (
            sum_logdetC + logdet_Phi_gwb + logdet_Sigma_gwb
        )

    # Constant term
    n_total = sum(td.n_toas for td in config.toa_data_list)
    total_logL = total_logL - 0.5 * n_total * jnp.log(2.0 * jnp.pi)

    return total_logL
