"""Single-pulsar log-likelihood for JaxPINT.

Composes residuals, noise covariance, and the Woodbury solver into a
differentiable, JIT-compatible log-likelihood evaluation.

The Gaussian log-likelihood is evaluated via the Woodbury matrix identity
to avoid forming the full n_toas x n_toas covariance matrix; see
van Haasteren et al. (2009) [1]_ Appendix A and Lentati et al. (2013) [2]_
Section II.B.

There is also **conditioned** log-likelihood functionality where the GP coefficients are kept explicit (for HMC sampling).

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
    woodbury_dot_qr,
)


def _residuals_and_woodbury(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ] = None,
) -> tuple[
    Float[Array, " n_toas"],
    Float[Array, " n_toas"],
    Float[Array, "n_toas n_basis"],
    Float[Array, " n_basis"],
]:
    """Residuals and the assembled per-pulsar Woodbury blocks ``(r, Ndiag, U, Phi)``.

    The shared front half of every per-pulsar likelihood path
    (:func:`single_pulsar_logL`, :func:`single_pulsar_clogL`,
    :func:`~jaxpint.pta.conditional_single_pulsar`, and the PTA
    inner tier ``_per_pulsar_intermediates``): compute residuals, subtract
    any deterministic ``external_delay``, then stack the noise model's
    ``(U, Phi)`` with an optional ``external_cov`` block so
    ``C = diag(Ndiag) + U diag(Phi) Uᵀ``.
    """
    r = compute_time_residuals(timing_model, toa_data, params)
    if external_delay is not None:
        r = r - external_delay
    Ndiag, U_noise, Phi_noise = noise_model.covariance(toa_data, params)
    woodbury = concat_woodbury_blocks((U_noise, Phi_noise), external_cov)
    assert woodbury is not None  # first block is always non-None
    U, Phi = woodbury
    return r, Ndiag, U, Phi


def single_pulsar_logL(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ] = None,
    use_qr: bool = False,
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
    use_qr : bool
        If True, evaluate the Woodbury quadratic form / log-determinant with
        :func:`~jaxpint.utils.woodbury_dot_qr` (square-root form) instead of
        :func:`~jaxpint.utils.woodbury_dot` (Cholesky of the Gram).
        Use if covariance is genuinely collinear
        for multi-parameter MSPs, which cuases the Gram form loses several digits).

    Returns
    -------
    logL : float
        Log-likelihood value.
    """
    r, Ndiag, U, Phi = _residuals_and_woodbury(
        toa_data, timing_model, noise_model, params, external_delay, external_cov
    )

    # Evaluate via Woodbury (square-root QR form when the basis may be
    # collinear, e.g. the marginalization design-matrix block at Φ=1e40).
    dot = woodbury_dot_qr if use_qr else woodbury_dot
    rCr, logdetC = dot(Ndiag, U, Phi, r, r)
    n = r.shape[0]
    return -0.5 * rCr - 0.5 * logdetC - 0.5 * n * jnp.log(2 * jnp.pi)


def _validate_coefficient_length(
    got: int,
    expected: int,
    *,
    producer: str,
    canonical: str,
    system: str,
) -> None:
    """Raise an actionable error if a ``clogL`` coefficient vector is mis-sized.
    The expected length is static  so it runs once during jit tracing."""
    if got != expected:
        raise ValueError(
            f"{producer}: coefficients has length {got}, but this {system} "
            f"expects {expected}. Pass {canonical}(...).mean, or a "
            f"sample_conditional(key, {canonical}(...)) draw — both are "
            f"always the right length and column order."
        )


def single_pulsar_clogL(
    toa_data: TOAData,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    params: ParameterVector,
    coefficients: Float[Array, " n_coeff"],
    external_delay: Optional[Float[Array, " n_toas"]] = None,
    external_cov: Optional[
        tuple[Float[Array, "n_toas n_basis"], Float[Array, " n_basis"]]
    ] = None,
) -> Float[Array, ""]:
    r"""Per-pulsar log-likelihood *conditioned* on explicit GP coefficients.

    The counterpart of :func:`single_pulsar_logL`.  Where that function
    marginalizes the GP coefficients ``a`` out via the Woodbury identity,
    this one keeps them as an explicit input and evaluates the joint
    Gaussian density of data-and-coefficients:

    .. math::

        \mathrm{clogL}(\theta, a)
          = \underbrace{-\tfrac12 (r - U a)^T N^{-1} (r - U a)
              - \tfrac12 \log|2\pi N|}_{\text{data} \mid a}
            \underbrace{- \tfrac12 a^T \Phi^{-1} a
              - \tfrac12 \log|2\pi \Phi|}_{\text{coeff. prior}},

    where ``N`` is the white-noise diagonal, ``U`` the stacked correlated
    basis (every ``noise_model`` correlated block, plus ``external_cov``
    last) and ``\Phi`` the diagonal prior weights.  Both ``N`` and
    ``\Phi`` are diagonal, so no factorization is needed — this is cheaper
    than the marginal ``single_pulsar_logL``.

    Parameters
    ----------
    toa_data, timing_model, noise_model, params, external_delay, external_cov
        As for :func:`single_pulsar_logL`.
    coefficients : (n_coeff,) array
        GP coefficient vector, in the same stacked-column order as
        :func:`~jaxpint.pta.conditional_single_pulsar`'s
        ``mean`` (``noise_model.covariance`` columns first, then
        ``external_cov``).  **The canonical way to obtain a valid vector is
        the matching conditional** — its ``.mean`` or a
        :func:`~jaxpint.pta.sample_conditional` draw — which is always the
        right length and order.  To build one from scratch, its length is
        ``conditional_single_pulsar(...).mean.shape[0]``.  A mismatched
        length raises a ``ValueError`` naming the expected size.

    Returns
    -------
    clogL : float
        Joint log-density of the residuals and the supplied coefficients.
    """
    r, Ndiag, U, Phi = _residuals_and_woodbury(
        toa_data, timing_model, noise_model, params, external_delay, external_cov
    )

    _validate_coefficient_length(
        coefficients.shape[0],
        U.shape[1],
        producer="single_pulsar_clogL",
        canonical="conditional_single_pulsar",
        system="pulsar's GP basis",
    )

    # Data term against the white-noise diagonal, GP realization subtracted.
    resid = r - U @ coefficients
    n = r.shape[0]
    log2pi = jnp.log(2 * jnp.pi)
    data_term = (
        -0.5 * jnp.sum(resid**2 / Ndiag)
        - 0.5 * jnp.sum(jnp.log(Ndiag))
        - 0.5 * n * log2pi
    )
    # Coefficient prior (diagonal Phi).
    n_coeff = coefficients.shape[0]
    prior_term = (
        -0.5 * jnp.sum(coefficients**2 / Phi)
        - 0.5 * jnp.sum(jnp.log(Phi))
        - 0.5 * n_coeff * log2pi
    )
    return data_term + prior_term


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
