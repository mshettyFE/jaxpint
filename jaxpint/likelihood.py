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

from jaxpint.fitter import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import woodbury_dot


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
    external_cov : optional (U, Phi) tuple
        U: (n_toas, n_basis), Phi: (n_basis,).
        Augments the noise covariance: C += U @ diag(Phi) @ U^T.

    Returns
    -------
    float
        Log-likelihood value.
    """
    # 1. Residuals from the timing model
    r = compute_time_residuals(timing_model, toa_data, params)

    # 2. Subtract external delay (positive delay = later arrival)
    if external_delay is not None:
        r = r - external_delay

    # 3. Noise covariance
    Ndiag, U, Phi = noise_model.covariance(toa_data, params)

    # 4. Augment with external covariance
    if external_cov is not None:
        U_ext, Phi_ext = external_cov
        U = jnp.concatenate([U, U_ext], axis=1)
        Phi = jnp.concatenate([Phi, Phi_ext])

    # 5. Evaluate via Woodbury
    rCr, logdetC = woodbury_dot(Ndiag, U, Phi, r, r)
    n = r.shape[0]
    return -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)
