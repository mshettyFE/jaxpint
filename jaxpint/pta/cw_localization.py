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

from typing import Callable

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
    det_sigma = jnp.where(det_F > 0.0, 1.0 / det_F, jnp.inf)
    area_str = jnp.pi * dchi2 * jnp.sqrt(det_sigma)
    return area_str * _STR_TO_DEG2
