"""Fisher-matrix sky-localization for a continuous-GW source.

Companion to :mod:`jaxpint.pta.cw_upper_limit` (which gives analytic upper
limits on strain). Here we go the other way: assume a *known* signal of
amplitude ``h0`` and compute the per-pixel 2-D Fisher information for the GW
sky position ``(cos_gwtheta, gwphi)``. The 90% credible localization area is
``pi * 4.605 * sqrt(det Sigma)`` steradians where ``Sigma = (-Hessian)^{-1}``;
``(cos_gwtheta, gwphi)`` is the area-preserving sky parameterization (the
solid-angle element is ``d(cos theta) d phi`` with no extra Jacobian), so the
area maps cleanly to deg^2 via the standard ``(180/pi)^2`` factor.

The intended use is reproducing arXiv:2603.28897 (Wen et al. 2026) style
anchor-pulsar scaling plots — for each anchor configuration, build a CWInjector
with the matching ``pulsar_term_mask``, set ``h0`` per pixel so the optimal SNR
matches a target (typically 20), evaluate the sky Fisher at the truth point,
and report the 90% credible area as a function of sky direction.

The Fisher-matrix approximation is accurate at high SNR with well-localized
posteriors; it underestimates area in regimes where the pulsar-term-phase
likelihood is genuinely multi-modal (mostly the "no anchors" limit). For
order-of-magnitude scaling studies that's fine; for absolute numbers in the
no-anchor regime, prefer sampling-based methods.
"""
from __future__ import annotations

from typing import Callable, cast

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# Chi^2 quantile at p=0.9 for 2 d.o.f.: Δχ² = -2 ln(1-p) = 4.605... — sets the
# scale of the 90% credible ellipse for a 2-D Gaussian posterior.
_DCHI2_90 = -2.0 * jnp.log(0.1)

# Solid-angle conversion: (180/π)² deg² per steradian.
_STR_TO_DEG2 = (180.0 / jnp.pi) ** 2


def h0_for_snr(
    snr_target: float,
    Y: Float[Array, ""],
) -> Float[Array, ""]:
    """Calibrate ``h0`` so the optimal matched-filter SNR² equals ``snr_target²``.

    ``Y = (s_hat | s_hat)`` (from :func:`jaxpint.pta.cw_upper_limit.quadratic_coeffs`)
    is the unit-strain signal power; the optimal SNR² of a signal of amplitude
    ``h0`` is ``h0² * Y``. Setting that equal to ``snr_target²`` gives
    ``h0 = snr_target / sqrt(Y)``.
    """
    Y = jnp.maximum(Y, jnp.finfo(jnp.float64).tiny)
    return snr_target / jnp.sqrt(Y)


def sky_fisher(
    logL_of_sky: Callable[[Float[Array, " 2"]], Float[Array, ""]],
    sky_truth: Float[Array, " 2"],
) -> Float[Array, "2 2"]:
    """The 2x2 sky Fisher information at the truth point.

    Parameters
    ----------
    logL_of_sky
        Scalar log-likelihood as a function of the 2-vector
        ``(cos_gwtheta, gwphi)``. Should close over the (timing-marginalized)
        PTA likelihood with all other CW parameters — amplitude, orientation,
        frequency — held at their truth values.
    sky_truth
        The injected sky position, ``[cos_gwtheta, gwphi]``.

    Returns
    -------
    F : (2, 2) array
        Observed Fisher information matrix ``F = -d² logL / d sky²`` at the
        truth. Under the Gaussian likelihood approximation, the posterior on
        the sky around the truth is ``N(sky_truth, F^{-1})``.
    """
    return -jax.hessian(logL_of_sky)(sky_truth)


def gram_at_pixel(
    logL_2sky: Callable[
        [Float[Array, ""], Float[Array, ""], Float[Array, " 2"], Float[Array, " 2"]],
        Float[Array, ""],
    ],
    sky_pixel: Float[Array, " 2"],
) -> Float[Array, "2 2"]:
    r"""The 2x2 sky Gram matrix at ``sky_pixel`` — proper expected Fisher / h0^2.

    Computes

    .. math::
        \mathrm{Gram}_{ij}(\theta) = \big(\partial_i \hat s(\theta)\,\big|\,
                                          \partial_j \hat s(\theta)\big)_N

    — the noise-weighted Gram matrix of signal gradients.  By construction this
    is positive semi-definite, so ``F = h0^2 * Gram`` is the proper PSD Fisher
    information for sky parameters under Wen-style signal-injection sensitivity
    forecasts.  Unlike ``0.5*h0^2*Hessian_sky(Y)`` (which is ``Gram + Curv``,
    sign-indefinite in ``Curv``) and unlike data-injection ``-Hessian(logL)``
    (which equals ``h0^2 * Gram`` only *in expectation* over noise), this is
    exact per realization and noise-independent.

    Construction: ``logL_2sky(h_a, h_b, sky_a, sky_b)`` is the timing-marginalized
    log-likelihood with *two* CW injectors active (amplitudes ``h_a, h_b``, sky
    positions ``sky_a, sky_b``).  Since the Gaussian likelihood is *exactly
    bilinear* in ``(h_a, h_b)``,

    .. math::
        \log L = \text{const} + h_a X_a + h_b X_b - \tfrac{1}{2} h_a^2 Y_a
                 - \tfrac{1}{2} h_b^2 Y_b - h_a h_b Z(\theta_a, \theta_b),

    where ``Z(\theta_a, \theta_b) = (\hat s(\theta_a) | \hat s(\theta_b))_N``
    is the cross-template inner product.  Taking the mixed amplitude derivative
    isolates the cross-coefficient

    .. math::
        Z(\theta_a, \theta_b) = -\frac{\partial^2 \log L}{\partial h_a\,\partial h_b}\bigg|_{h_a = h_b = 0}.

    Then

    .. math::
        \mathrm{Gram}_{ij}(\theta) = \frac{\partial^2 Z}{\partial \theta_{a,i}\,
                                          \partial \theta_{b,j}}\bigg|_{\theta_a = \theta_b = \theta}.

    Total autodiff: 4th-order mixed (2 in amplitudes, 2 in sky).  Works given
    the slogdet and arccos refactors landed in jaxpint earlier.

    Parameters
    ----------
    logL_2sky : callable
        ``(h_a, h_b, sky_a, sky_b) -> scalar`` log-likelihood with both injectors
        active.  Non-amplitude, non-sky CW parameters (frequency, orientation)
        should be closed over in the caller, identical for the two injectors.
    sky_pixel : (2,) array
        Sky position ``(cos_gwtheta, gwphi)`` at which to evaluate.

    Returns
    -------
    Gram : (2, 2) array
        The 2x2 Gram matrix at ``sky_pixel``.  Multiply by ``h0_target**2`` to
        get the sky Fisher information.
    """
    def Z(sky_a, sky_b):
        # Bilinear in (h_a, h_b) ⇒ ∂²/∂h_a∂h_b extracts -Z exactly.  Evaluate at
        # (0, 0) for convenience; any (h_a, h_b) gives the same constant.
        f = lambda h_a, h_b: logL_2sky(h_a, h_b, sky_a, sky_b)
        return -jax.grad(jax.grad(f, argnums=0), argnums=1)(
            jnp.float64(0.0), jnp.float64(0.0)
        )

    return jax.jacfwd(jax.jacrev(Z, argnums=0), argnums=1)(sky_pixel, sky_pixel)


def gram_block_at_pair(
    logL_2sky: Callable[
        [Float[Array, ""], Float[Array, ""], Float[Array, " 2"], Float[Array, " 2"]],
        Float[Array, ""],
    ],
    sky_a_truth: Float[Array, " 2"],
    sky_b_truth: Float[Array, " 2"],
) -> Float[Array, "2 2"]:
    r"""Cross-template Gram block evaluated at two distinct sky positions.

    Generalization of :func:`gram_at_pixel` for multi-source forecasting.  Same
    mixed-derivative trick — `Z(\theta_a, \theta_b) = -\partial^2 \log L /
    \partial h_a \partial h_b` is extracted via the bilinearity of the
    Gaussian likelihood in the two amplitudes; the mixed sky-Hessian then
    gives the off-diagonal Gram block.  Reduces exactly to :func:`gram_at_pixel`
    when ``sky_a_truth == sky_b_truth`` (same source on both sides).

    The two CWInjectors that ``logL_2sky`` closes over should represent the
    *two source templates whose cross-coupling this block measures*:

    * **Diagonal `G_aa`**: both injectors bound to source `a`'s parameters
      (template + data convention; identical to single-pixel Level 1).
    * **Off-diagonal `G_ab`, a != b**: injector A bound to source `a`'s
      params; injector B bound to source `b`'s.  Encodes how the data
      jointly constrains both source positions through the per-pulsar
      pulsar-term phases.

    Parameters
    ----------
    logL_2sky : callable
        ``(h_a, h_b, sky_a, sky_b) -> scalar`` log-likelihood with both
        injectors active and bound to the appropriate source parameters
        (orientation, frequency).  All non-pair injectors should be held at
        zero amplitude in the closure.
    sky_a_truth : (2,) array
        First source's true sky position ``(cos_gwtheta, gwphi)``.
    sky_b_truth : (2,) array
        Second source's true sky position.  Same as ``sky_a_truth`` for a
        diagonal block.

    Returns
    -------
    Gram_ab : (2, 2) array
        The cross Gram block.  Multiply by ``h_a_target * h_b_target`` to
        get the Fisher-information contribution to the joint Fisher matrix.
    """
    def Z(sky_a, sky_b):
        f = lambda h_a, h_b: logL_2sky(h_a, h_b, sky_a, sky_b)
        return -jax.grad(jax.grad(f, argnums=0), argnums=1)(
            jnp.float64(0.0), jnp.float64(0.0)
        )

    return jax.jacfwd(jax.jacrev(Z, argnums=0), argnums=1)(sky_a_truth, sky_b_truth)


def assemble_joint_fisher(
    gram_blocks: dict[tuple[int, int], Float[Array, "2 2"]],
    h0_targets: Float[Array, " K"],
    K: int,
) -> Float[Array, "2K 2K"]:
    r"""Stack ``K(K+1)/2`` unique Gram blocks into the symmetric joint Fisher.

    The joint Fisher matrix for K simultaneous CGW sources has structure

    .. math::
        F_{(a,i),(b,j)} = h_{0,a}\,h_{0,b}\,(\partial_i \hat s_a | \partial_j \hat s_b)_N,

    so each ``2 x 2`` Gram block scales by the product of the two sources'
    calibrated amplitudes.  Caller only needs to provide unique blocks
    (``a <= b``); the function symmetrizes off-diagonals into the joint matrix
    automatically.

    Parameters
    ----------
    gram_blocks : dict
        Mapping ``(a, b) -> (2, 2)`` Gram block, for all unique pairs with
        ``0 <= a <= b < K``.  Length must be ``K * (K + 1) / 2``.
    h0_targets : (K,) array
        Per-source calibrated amplitudes ``h_a^{target} = SNR_a / sqrt(Y_a)``.
    K : int
        Number of sources.  Must match ``len(h0_targets)`` and the unique
        block count in ``gram_blocks``.

    Returns
    -------
    F : (2K, 2K) array
        Symmetric joint Fisher information matrix.  ``F[2a:2a+2, 2b:2b+2]``
        is the ``(a, b)``-block contribution.
    """
    F = jnp.zeros((2 * K, 2 * K), dtype=jnp.float64)
    for (a, b), G_ab in gram_blocks.items():
        assert a <= b, f"gram_blocks keys must satisfy a <= b; got ({a}, {b})"
        scaled = h0_targets[a] * h0_targets[b] * G_ab
        F = F.at[2 * a : 2 * a + 2, 2 * b : 2 * b + 2].set(scaled)
        if a != b:
            F = F.at[2 * b : 2 * b + 2, 2 * a : 2 * a + 2].set(scaled.T)
    return F


def per_source_credible_areas_deg2(
    F: Float[Array, "2K 2K"],
    K: int,
    level: float = 0.9,
) -> Float[Array, " K"]:
    r"""Per-source marginal credible area from the joint Fisher matrix.

    Inverts the joint Fisher to get ``Sigma = F^{-1}``, slices the ``2 x 2``
    marginal sky covariance per source ``Sigma_k = Sigma[2k:2k+2, 2k:2k+2]``,
    inverts to get the marginal Fisher per source, and computes the credible
    area via :func:`credible_area_deg2`.

    The marginal per-source Fisher is *strictly smaller* than the
    corresponding diagonal block ``F[2k:2k+2, 2k:2k+2]`` whenever the
    off-diagonal blocks of ``F`` are non-zero — this is the **source-confusion
    penalty**: simultaneous sources degrade each other's localization through
    the joint posterior's covariance, even at fixed per-source SNR.

    Returns ``inf`` for sources where the marginal covariance is degenerate
    (e.g., coincident sources, or any other case where the joint Fisher is
    near-rank-deficient).

    Parameters
    ----------
    F : (2K, 2K) array
        Joint Fisher information from :func:`assemble_joint_fisher`.
    K : int
        Number of sources (must match the Fisher matrix size).
    level : float
        Credible level (default 0.9 → 90% area).

    Returns
    -------
    areas : (K,) array
        Per-source marginal credible localization area in deg^2.
    """
    Sigma = jnp.linalg.inv(F)
    out = []
    for k in range(K):
        Sigma_k = Sigma[2 * k : 2 * k + 2, 2 * k : 2 * k + 2]
        # Marginal Fisher = inverse of marginal covariance.  credible_area_deg2
        # handles non-PSD / singular cases by returning inf.
        det_Sigma_k = jnp.linalg.det(Sigma_k)
        # Direct: area = pi * Delta_chi2 * sqrt(det Sigma_k) * (180/pi)^2.
        dchi2 = -2.0 * jnp.log(1.0 - level)
        det_safe = cast(Array, jnp.where(det_Sigma_k > 0.0, det_Sigma_k, jnp.inf))
        area_str = jnp.pi * dchi2 * jnp.sqrt(det_safe)
        out.append(area_str * _STR_TO_DEG2)
    return jnp.stack(out)


def credible_area_deg2(
    F: Float[Array, "2 2"],
    level: float = 0.9,
) -> Float[Array, ""]:
    r"""90% credible 2-D localization area in deg² from a sky Fisher matrix.

    For a 2-D Gaussian posterior ``N(0, Sigma)`` with ``Sigma = F^{-1}`` the
    ``level``-credible ellipse has area

    .. math::
        A = \pi \cdot \Delta\chi^2 \cdot \sqrt{\det \Sigma},

    with ``Δχ² = -2 ln(1 - level)`` (4.605 at 90%, 2 d.o.f.). The result is in
    the *parameter* units of ``F``. Since ``(cos_gwtheta, gwphi)`` is the
    area-preserving sky parameterization, this is also the solid-angle area in
    steradians — convert to deg² with the standard ``(180/pi)²`` factor.

    Returns ``inf`` for a singular / negative-determinant Fisher (degenerate /
    unphysical posterior) rather than raising — convenient when vmapping over
    a sky grid where a handful of pixels can degenerate.
    """
    dchi2 = -2.0 * jnp.log(1.0 - level)
    det_F = jnp.linalg.det(F)
    # det Sigma = 1 / det F; det F <= 0 → unphysical (non-positive-definite).
    det_sigma = cast(Array, jnp.where(det_F > 0.0, 1.0 / det_F, jnp.inf))
    area_str = jnp.pi * dchi2 * jnp.sqrt(det_sigma)
    return area_str * _STR_TO_DEG2
