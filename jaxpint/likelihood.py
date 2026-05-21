"""Single-pulsar log-likelihood for JaxPINT.

Composes residuals, noise covariance, and the Woodbury solver into a
differentiable, JIT-compatible log-likelihood evaluation.

The Gaussian log-likelihood is evaluated via the Woodbury matrix identity
to avoid forming the full n_toas x n_toas covariance matrix; see
van Haasteren et al. (2009) [1]_ Appendix A and Lentati et al. (2013) [2]_
Section II.B.

References
----------
.. [1] van Haasteren et al. (2009), "On measuring the gravitational-wave
   background using pulsar timing arrays", MNRAS 395, 1005.
.. [2] Lentati et al. (2013), "Hyper-efficient model-independent Bayesian
   method for the analysis of pulsar timing data", PRD 87, 104021.
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.fitters import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import (
    WoodburyFactor,
    apply_woodbury_dot_factor,
    concat_woodbury_blocks,
    precompute_woodbury_factor,
    woodbury_dot,
)


def single_pulsar_logL(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ] = None,
) -> Float[Array, ""]:
    """Per-pulsar log-likelihood with optional external injections.

    Parameters
    ----------
    toa_data : TOAData
        Pulse time-of-arrival data.
    timing_model : TimingModel
        JaxPINT timing model (delay + phase components).
    noise_model : NoiseModel
        JaxPINT noise model (white + correlated noise).
    params : ParameterVector
        Timing and noise parameters for this pulsar.
    external_delay : optional array (n_toas,)
        Pre-computed external delay in seconds (e.g., sum of CW signals).
        Subtracted from residuals (positive delay = later arrival).
    external_cov
        Optional ``(U, Phi)`` tuple where ``U`` has shape ``(n_toas, n_basis)``
        and ``Phi`` has shape ``(n_basis,)``.  Augments the noise covariance:
        ``C += U @ diag(Phi) @ U.T``.

    Returns
    -------
    logL : float
        Log-likelihood value.
    """
    # 1. Residuals from the timing model
    r = compute_time_residuals(timing_model, toa_data, params)

    # 2. Subtract external delay (positive delay = later arrival)
    if external_delay is not None:
        r = r - external_delay

    # 3. Noise covariance, optionally augmented with external (U, Φ) blocks
    Ndiag, U_noise, Phi_noise = noise_model.covariance(toa_data, params)
    U, Phi = concat_woodbury_blocks((U_noise, Phi_noise), external_cov)

    # 5. Evaluate via Woodbury
    rCr, logdetC = woodbury_dot(Ndiag, U, Phi, r, r)
    n = r.shape[0]
    return -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)


def precompute_single_pulsar_factor(
    toa_data: TOAData,
    noise_model: NoiseModel,
    params: ParameterVector,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ] = None,
) -> WoodburyFactor:
    """Precompute the noise-only half of :func:`single_pulsar_logL`.

    Returns a :class:`~jaxpint.utils.WoodburyFactor` that captures the
    per-pulsar Cholesky factorization (independent of any timing-domain
    parameter the residual computation will later read). Pair with
    :func:`single_pulsar_logL_with_factor` for repeated evaluations
    where ``noise_model``, ``params``-noise-fields, and ``external_cov``
    don't change.

    Parameters
    ----------
    toa_data, noise_model, params
        Same as :func:`single_pulsar_logL`. ``params`` is read for
        noise-related fields only here (e.g. ``EFAC``, ``TNREDAMP``);
        timing-domain values are unused at this stage.
    external_cov
        Same as :func:`single_pulsar_logL` — augments ``(U, Phi)``.
    """
    Ndiag, U, Phi = noise_model.covariance(toa_data, params)
    if external_cov is not None:
        U_ext, Phi_ext = external_cov
        U = jnp.concatenate([U, U_ext], axis=1)
        Phi = jnp.concatenate([Phi, Phi_ext])
    return precompute_woodbury_factor(Ndiag, U, Phi)


def single_pulsar_logL_with_factor(
    toa_data: TOAData,
    timing_model: TimingModel,
    factor: WoodburyFactor,
    params: ParameterVector,
    external_delay: Optional[Float[Array, " n_toas"]] = None,
) -> Float[Array, ""]:
    """Per-pulsar log-likelihood using a precomputed Woodbury factor.

    Functionally equivalent to :func:`single_pulsar_logL` for the same
    underlying ``(noise_model, external_cov, params)`` configuration,
    but reuses a precomputed factor for the noise covariance side. The
    only per-call work is residuals + external-delay subtraction +
    factor application — no Cholesky, no `(n_toas, n_basis)`
    multiplications.

    The right tool when many evaluations share the same noise covariance
    and only the timing-domain residuals change (e.g. a 1D scan over a
    pulsar's PX or a CW global parameter, which does not enter the
    noise covariance).

    Parameters
    ----------
    toa_data, timing_model
        Same as :func:`single_pulsar_logL`.
    factor
        Output of :func:`precompute_single_pulsar_factor` for the
        target ``noise_model``, noise-side ``params`` and
        ``external_cov``.
    params
        Timing-domain parameters for this evaluation. The factor's
        construction-time noise-related values are *not* re-read; if
        the caller mutates noise-related entries of ``params`` between
        precompute and apply, the factor is stale and the result is
        wrong. (No runtime check — it would defeat the purpose.)
    external_delay
        Same as :func:`single_pulsar_logL`.
    """
    r = compute_time_residuals(timing_model, toa_data, params)
    if external_delay is not None:
        r = r - external_delay
    rCr, logdetC = apply_woodbury_dot_factor(factor, r, r)
    n = r.shape[0]
    return -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)
