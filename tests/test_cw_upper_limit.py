"""Tests for the analytic CW strain upper-limit helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import ndtr

from jaxpint.frequentist.detection import fstat
from jaxpint.pta.cw_upper_limit import h0_95_closed_form, h0_95_marginalized, h0_to_distance
from jaxpint.pta.extraction import basis_quadratics, orientation_coeffs, quadratic_coeffs
from jaxpint.types import GlobalParams
from jaxpint.pta.signals.cw import (
    CWInjector,
    log10_strain_from_binary,
    fdot,
    evolution_ok,
)
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
            positions,
            earth_term_only=True,
            linear_amplitude=True,
            initial_values={
                "cos_gwtheta": 0.3,
                "gwphi": 1.7,
                "log10_fgw": -8.0,
                "cos_inc": 0.4,
                "psi": 0.6,
                "phase0": 0.9,
            },
        )
        gp = inj.register_params(GlobalParams.empty())
        t = np.array([59000.0, 59300.0, 59600.0, 59900.0, 60200.0])
        toa = make_toa_data(t_mjd=t)
        return inj, gp, toa

    def test_registers_seven_params_with_linear_amp(self):
        inj, gp, _ = self._setup()
        assert gp.n_params == 7
        assert "cw0_h0" in gp.names  # linear amplitude param
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


class TestH0Marginalized:
    """h0_95_marginalized: mixture-CDF quantile over an orientation grid."""

    @pytest.mark.parametrize("X,Y", [(0.0, 4.0), (5.0, 2.0), (-3.0, 1.5)])
    def test_single_orientation_reduces_to_closed_form(self, X, Y):
        # A one-point grid is one truncated Gaussian -> must match the closed form.
        marg = h0_95_marginalized(jnp.array([X]), jnp.array([Y]))
        closed = h0_95_closed_form(jnp.float64(X), jnp.float64(Y))
        assert jnp.allclose(marg, closed, rtol=1e-6)

    def test_identical_orientations_reduce_to_closed_form(self):
        # N copies of the same component is the same posterior -> same quantile.
        X, Y = 2.0, 1.3
        marg = h0_95_marginalized(jnp.full((7,), X), jnp.full((7,), Y))
        closed = h0_95_closed_form(jnp.float64(X), jnp.float64(Y))
        assert jnp.allclose(marg, closed, rtol=1e-6)

    def test_expected_mode_between_component_limits(self):
        # X=0 mixture of two sensitivities: the marginal 95% sits between the
        # best (large Y) and worst (small Y) single-orientation limits.
        Y_big, Y_small = 9.0, 1.0
        marg = h0_95_marginalized(jnp.array([0.0, 0.0]), jnp.array([Y_big, Y_small]))
        best = h0_95_closed_form(jnp.float64(0.0), jnp.float64(Y_big))
        worst = h0_95_closed_form(jnp.float64(0.0), jnp.float64(Y_small))
        assert best < marg < worst

    def test_vmaps_over_pixels(self):
        Xs = jnp.zeros((3, 5))
        Ys = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (3, 5))) + 0.5
        out = jax.vmap(h0_95_marginalized)(Xs, Ys)
        assert out.shape == (3,) and jnp.all(out > 0)


class TestOrientationCoeffs:
    def test_shape_and_finite(self):
        c = orientation_coeffs(jnp.float64(0.3), jnp.float64(0.6), jnp.float64(0.9))
        assert c.shape == (4,) and jnp.all(jnp.isfinite(c))


class TestFstat:
    def test_matches_maximized_loglike(self):
        # 2F = b^T M^-1 b = 2 * max_A (A.b - 0.5 A.M.A).
        key = jax.random.PRNGKey(3)
        A = jax.random.normal(key, (4, 4))
        M = A @ A.T + jnp.eye(4)  # SPD
        b = jax.random.normal(jax.random.PRNGKey(4), (4,))
        A_hat = jnp.linalg.solve(M, b)
        loglike_max = A_hat @ b - 0.5 * A_hat @ M @ A_hat
        assert jnp.allclose(fstat(M, b), 2.0 * loglike_max, rtol=1e-10)


class TestBasisReductionConsistency:
    """The crucial validation: orientation_coeffs + basis_quadratics must
    reproduce the real CWInjector residual structure. If orientation_coeffs were
    derived wrong (sign/normalization vs cw.py), held-out reconstruction breaks.

    Uses 2 pulsars (so the 4 basis waveforms {f_p S, f_p C, f_c S, f_c C} are
    full rank; one pulsar collapses them to rank 2)."""

    def _toy_pta_logL(self):
        positions = jnp.array([[0.3, -0.6, 0.74], [-0.5, 0.2, 0.84]])
        positions = positions / jnp.linalg.norm(positions, axis=1, keepdims=True)
        inj = CWInjector(
            positions,
            earth_term_only=True,
            linear_amplitude=True,
            initial_values={"cos_gwtheta": 0.3, "gwphi": 1.7, "log10_fgw": -8.0},
        )
        gp = inj.register_params(GlobalParams.empty())
        t = np.linspace(58000.0, 60500.0, 24)
        toas = [make_toa_data(t_mjd=t), make_toa_data(t_mjd=t + 7.0)]
        # Fixed pseudo-data + inverse-variance (the quadratic logL: X=(d|s), Y=(s|s)).
        key = jax.random.PRNGKey(7)
        data = jax.random.normal(key, (2 * t.size,)) * 1e-7
        invvar = jnp.full((2 * t.size,), 1.0 / (1e-7) ** 2)

        def logL(amp, ci, psi, ph):
            g = (
                gp.with_value("cw0_h0", amp)
                .with_value("cw0_cos_inc", ci)
                .with_value("cw0_psi", psi)
                .with_value("cw0_phase0", ph)
            )
            s = jnp.concatenate([inj.delay(i, toas[i], None, g) for i in range(2)])
            return jnp.sum(data * s * invvar) - 0.5 * jnp.sum(s * s * invvar)

        return logL

    def test_reconstructs_held_out_orientations(self):
        logL = self._toy_pta_logL()
        M, b = basis_quadratics(logL)

        # Held-out orientations not in the extraction set.
        key = jax.random.PRNGKey(123)
        k1, k2, k3 = jax.random.split(key, 3)
        cis = jax.random.uniform(k1, (6,), minval=-1.0, maxval=1.0)
        psis = jax.random.uniform(k2, (6,), minval=0.0, maxval=float(np.pi))
        phs = jax.random.uniform(k3, (6,), minval=0.0, maxval=2.0 * float(np.pi))
        for ci, psi, ph in zip(cis, psis, phs):
            c = orientation_coeffs(ci, psi, ph)
            X_recon, Y_recon = c @ b, c @ M @ c
            X_direct, Y_direct = quadratic_coeffs(lambda a: logL(a, ci, psi, ph))
            assert jnp.allclose(X_recon, X_direct, rtol=1e-6, atol=1e-8)
            assert jnp.allclose(Y_recon, Y_direct, rtol=1e-6, atol=1e-8)

    def test_gram_matrix_is_psd(self):
        logL = self._toy_pta_logL()
        M, _ = basis_quadratics(logL)
        # M = (basis|basis) is symmetric and positive *definite* (hence full
        # rank) with 2 distinct pulsars -- assert the full-rank claim, not just
        # PSD: the smallest eigenvalue is strictly positive.
        assert jnp.allclose(M, M.T, atol=1e-10)
        eigs = jnp.linalg.eigvalsh(M)
        assert jnp.all(eigs > 1e-8 * eigs[-1])
        assert jnp.linalg.matrix_rank(M) == M.shape[0]


# ----------------------------------------------- source frequency evolution
def test_fdot_scalings():
    f0 = fdot(1e9, 27e-9)
    assert np.isclose(fdot(2e9, 27e-9), f0 * 2 ** (5 / 3))  # ∝ Mc^(5/3)
    assert np.isclose(fdot(1e9, 54e-9), f0 * 2 ** (11 / 3))  # ∝ f^(11/3)


def test_evolution_ok_flags_and_keys():
    # low mass / low freq stays monochromatic; high mass / high freq drifts out
    t_span = 15 * 365.25 * 86400.0  # 15 yr
    lo = evolution_ok(1e8, 27e-9, t_span)
    hi = evolution_ok(1e11, 1e-7, t_span)
    assert lo["earth_ok"] and not hi["earth_ok"]
    assert lo["drift_cycles"] < hi["drift_cycles"]
    assert set(lo) == {"earth_ok", "coherent_ok", "drift_cycles", "psr_dff"}
