"""Framework-specific noncentrality producer for the F-statistic sensitivity.

Calculates the per-orientation noncentrality ``lambda_1(theta)`` of a *unit-strain*
CW source, computed from the real timing-marginalized GLS likelihood.

"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.pta.cw_upper_limit import (
    _EXTRACTION_ORIENTATIONS,
    basis_quadratics,
    orientation_coeffs,
)
from jaxpint.pta.signals.cw import cw_delay_from_array

__all__ = ["earth_term_gram", "unit_noncentrality"]


def earth_term_gram(
    g: Callable,
    reduced_params,
    toa_data,
    pos: Float[Array, "3"],
    pulsar_dist,
    cos_gwtheta,
    gwphi,
    log10_fgw,
) -> Float[Array, "4 4"]:
    """The 4x4 Earth-term orientation Gram ``M`` for one pulsar at a fixed sky.

    ``M_ab = (basis_a | basis_b)`` over the 4 orientation-independent Earth-term CW
    waveforms, in the pulsar's timing-marginalized GLS metric.

    Parameters
    ----------
    g : callable
        Single-pulsar timing-marginalized log-likelihood,
        ``g(reduced_params, external_delay=...)`` (from
        :func:`jaxpint.bayes.marginalize_single_pulsar`).
    reduced_params
        The reduced-parameter skeleton ``g`` expects.
    toa_data : TOAData
        Pulse time-of-arrival data for this pulsar.
    pos : (3,) array
        Pulsar unit vector.
    pulsar_dist : scalar
        Pulsar parallax (mas); immaterial here (Earth-term only, no pulsar term).
    cos_gwtheta, gwphi : scalar
        GW source sky position (cos-colatitude, right ascension).
    log10_fgw : scalar
        ``log10`` GW frequency (Hz).

    Returns
    -------
    (4, 4) array
        The Earth-term orientation Gram ``M``.
    """

    def logL_at_orientation(amp, cos_inc, psi, phase0):
        cw = jnp.array([amp, cos_gwtheta, gwphi, log10_fgw, cos_inc, psi, phase0])
        delay = cw_delay_from_array(
            toa_data, pos, pulsar_dist, cw, earth_term_only=True, linear_amplitude=True
        )
        return g(reduced_params, external_delay=delay)

    M, _ = basis_quadratics(logL_at_orientation)
    return M


def unit_noncentrality(
    M: Float[Array, "4 4"],
    orientations: Float[Array, "n_theta 3"] = _EXTRACTION_ORIENTATIONS,
) -> Float[Array, " n_theta"]:
    """Per-orientation noncentrality ``lambda_1(theta) = c(theta)^T M c(theta)``.

    Parameters
    ----------
    M : (4, 4) array
        Earth-term orientation Gram, summed over pulsars (:func:`earth_term_gram`).
    orientations : (n_theta, 3) array
        ``(cos_inc, psi, phase0)`` draws to evaluate / average over.

    Returns
    -------
    (n_theta,) array
        ``lambda_1(theta)`` at each orientation.
    """
    C = jax.vmap(lambda o: orientation_coeffs(o[0], o[1], o[2]))(
        orientations
    )  # (n_theta, 4)
    return jnp.einsum("ta,ab,tb->t", C, M, C)
