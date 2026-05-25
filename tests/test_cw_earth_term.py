"""Tests for the ``earth_term_only`` mode of the CW delay.

The full CW residual is built from
``delta_sin = sin(phase_earth) - sin(phase_pulsar)`` and the analogous cosine
difference (written via the sum-to-product identity in ``cw_delay_from_array``).
With ``earth_term_only=True`` the pulsar-term sinusoid is dropped, leaving
``delta_sin = sin(phase_earth)``, ``delta_cos = cos(phase_earth)`` — and the
result no longer depends on ``pulsar_dist``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxpint.pta.signals.cw import cw_delay_from_array
from tests.helpers import make_toa_data


def _cw_params(
    log10_h=-14.0, cos_gwtheta=0.3, gwphi=1.7, log10_fgw=-8.0,
    cos_inc=0.4, psi=0.6, phase0=0.9,
):
    return jnp.array(
        [log10_h, cos_gwtheta, gwphi, log10_fgw, cos_inc, psi, phase0]
    )


def _toa():
    t = np.array([59000.0, 59200.0, 59400.0, 59600.0, 59800.0, 60000.0])
    return make_toa_data(t_mjd=t)

class TestEarthTermOnly:
    def test_independent_of_pulsar_distance(self):
        """Earth-term-only delay must not depend on PX."""
        toa = _toa()
        pos = jnp.array([0.3, -0.6, 0.74])
        pos = pos / jnp.linalg.norm(pos)
        p = _cw_params()
        d_near = cw_delay_from_array(toa, pos, jnp.float64(0.3), p, earth_term_only=True)
        d_far = cw_delay_from_array(toa, pos, jnp.float64(3.0), p, earth_term_only=True)
        assert jnp.allclose(d_near, d_far, rtol=1e-12, atol=1e-30)

    def test_full_term_depends_on_distance(self):
        """Sanity: the full (default) delay *does* depend on PX."""
        toa = _toa()
        pos = jnp.array([0.3, -0.6, 0.74])
        pos = pos / jnp.linalg.norm(pos)
        p = _cw_params()
        d_near = cw_delay_from_array(toa, pos, jnp.float64(0.3), p)
        d_far = cw_delay_from_array(toa, pos, jnp.float64(3.0), p)
        assert not jnp.allclose(d_near, d_far, rtol=1e-3)

