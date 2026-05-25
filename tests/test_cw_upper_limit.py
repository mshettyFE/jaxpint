"""Tests for the analytic CW strain upper-limit helpers."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr

from jaxpint.pta.cw_upper_limit import (
    quadratic_coeffs,
    h0_95_closed_form,
    h0_to_distance,
)
from jaxpint.pta.params import GlobalParams
from jaxpint.pta.signals.cw import CWInjector, log10_strain_from_binary
from tests.helpers import make_toa_data


def _truncated_normal_cdf(h, mu, sigma):
    """CDF at h of N(mu, sigma^2) truncated to [0, inf)."""
    lo = ndtr(-mu / sigma)
    return (ndtr((h - mu) / sigma) - lo) / (1.0 - lo)


class TestQuadraticCoeffs:
    def test_recovers_known_coeffs(self):
        X_true, Y_true, L0 = 3.7, 2.1, -12.0

        def logL(A):
            return L0 + A * X_true - 0.5 * A**2 * Y_true

        X, Y = quadratic_coeffs(logL)
        assert jnp.allclose(X, X_true, rtol=1e-10)
        assert jnp.allclose(Y, Y_true, rtol=1e-10)

    def test_independent_of_expansion_point(self):
        X_true, Y_true = -1.3, 0.8

        def logL(A):
            return A * X_true - 0.5 * A**2 * Y_true

        X0, Y0 = quadratic_coeffs(logL, amp=0.0)
        X5, Y5 = quadratic_coeffs(logL, amp=5.0)
        # X = dlogL/dA = X_true - A*Y_true, so X depends on the point; Y does not.
        assert jnp.allclose(Y0, Y5, rtol=1e-10)
        assert jnp.allclose(X5, X_true - 5.0 * Y_true, rtol=1e-10)


class TestClosedFormUL:
    def test_zero_matched_filter_gives_1p96_sigma(self):
        # X = 0 -> mu = 0 -> half-normal -> h0_95 = 1.96 sigma
        Y = 4.0
        sigma = 1.0 / np.sqrt(Y)
        h0 = h0_95_closed_form(jnp.float64(0.0), jnp.float64(Y))
        assert jnp.allclose(h0, 1.959964 * sigma, rtol=1e-4)

    @pytest.mark.parametrize("X,Y", [(0.0, 4.0), (5.0, 2.0), (-3.0, 1.5)])
    def test_cdf_at_ul_is_level(self, X, Y):
        h0 = h0_95_closed_form(jnp.float64(X), jnp.float64(Y))
        mu, sigma = X / Y, 1.0 / np.sqrt(Y)
        assert jnp.allclose(_truncated_normal_cdf(h0, mu, sigma), 0.95, atol=1e-6)


class TestH0ToDistance:
    def test_round_trip(self):
        log10_mc, log10_fgw = 9.0, float(np.log10(27e-9))
        log10_dist = np.log10(150.0)  # 150 Mpc
        h0 = 10.0 ** log10_strain_from_binary(log10_mc, log10_dist, log10_fgw)
        dist = h0_to_distance(h0, log10_mc, log10_fgw)
        assert jnp.allclose(dist, 150.0, rtol=1e-8)

    def test_smaller_h0_means_larger_distance(self):
        log10_mc, log10_fgw = 9.0, float(np.log10(27e-9))
        d_loud = h0_to_distance(jnp.float64(1e-14), log10_mc, log10_fgw)
        d_quiet = h0_to_distance(jnp.float64(1e-15), log10_mc, log10_fgw)
        assert d_quiet > d_loud


class TestCWInjectorLinearMode:
    """CWInjector(linear_amplitude=True, earth_term_only=True): the template the
    analytic UL relies on — residual linear in h0, no pulsar-distance dependence."""

    def _setup(self):
        positions = jnp.array([[0.3, -0.6, 0.74], [1.0, 0.0, 0.0]])
        positions = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
        inj = CWInjector(
            positions, earth_term_only=True, linear_amplitude=True,
            initial_values={"cos_gwtheta": 0.3, "gwphi": 1.7, "log10_fgw": -8.0,
                            "cos_inc": 0.4, "psi": 0.6, "phase0": 0.9},
        )
        gp = inj.register_params(GlobalParams.empty())
        t = np.array([59000.0, 59300.0, 59600.0, 59900.0, 60200.0])
        toa = make_toa_data(t_mjd=t)
        return inj, gp, toa

    def test_registers_seven_params_with_linear_amp(self):
        inj, gp, _ = self._setup()
        assert gp.n_params == 7
        assert "cw0_h0" in gp.names          # linear amplitude param
        assert "cw0_log10_h" not in gp.names  # not the log param

    def test_delay_linear_in_amplitude(self):
        inj, gp, toa = self._setup()
        # pulsar_params unused in earth_term_only mode (no PX lookup).
        d1 = inj.delay(0, toa, None, gp.with_value("cw0_h0", 1.0))
        d3 = inj.delay(0, toa, None, gp.with_value("cw0_h0", 3.0))
        assert jnp.allclose(d3, 3.0 * d1, rtol=1e-10)

    def test_zero_amplitude_zero_delay(self):
        inj, gp, toa = self._setup()
        d0 = inj.delay(0, toa, None, gp.with_value("cw0_h0", 0.0))
        assert jnp.allclose(d0, 0.0, atol=1e-30)
