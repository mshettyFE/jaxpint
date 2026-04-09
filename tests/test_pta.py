"""Tests for the PTA likelihood module (jaxpint.pta).

All tests use synthetic data — no PINT or Discovery dependency.
Analytic reference values are derived from:

- Antenna patterns: Ellis, Siemens & Creighton (2012), ApJ 756, 175.
- CW timing delay: Sesana & Vecchio (2010), PRD 81, 104008;
  Ellis (2013), CQG 30, 224004.
- Power-law PSD: Arzoumanian et al. (2016), ApJ 821, 13;
  Phinney (2001), astro-ph/0108028.
- Hellings-Downs ORF: Hellings & Downs (1983), ApJL 265, L39.
- Fourier basis: Lentati et al. (2013), PRD 87, 104021.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.pta.params import GlobalParams
from jaxpint.pta.signals.cw import fplus_fcross, cw_delay, CWInjector, CW_PARAM_DEFAULTS
from jaxpint.pta.signals.gwb import (
    powerlaw_psd, fourier_basis, gwb_covariance, CURNInjector, CURN_PARAM_DEFAULTS, FYR,
)
from jaxpint.pta.signals.orf import hd_orf, dipole_orf
from jaxpint.pta.likelihood import PTAConfig
from jaxpint.pta.fisher import flatten_params, unflatten_params
from jaxpint.types import ParameterVector

from tests.helpers import make_toa_data, make_params


# ---------------------------------------------------------------------------
# Helpers local to this test module
# ---------------------------------------------------------------------------


def _make_cw_global_params(prefix="cw0_", **overrides):
    """Build a GlobalParams with one CW source."""
    gp = GlobalParams.empty()
    inj = CWInjector(
        jnp.array([[1.0, 0.0, 0.0]]),
        prefix=prefix,
        initial_values=overrides or None,
    )
    return inj.register_params(gp)


def _make_simple_toa_data(n_toas=10):
    """Minimal TOAData spanning ~1 year from MJD 59000."""
    return make_toa_data(
        n_toas=n_toas,
        tdb_int=59000.0,
        tdb_frac=None,  # linspace 0.1–0.9
        error=1e-6,
        freq=1400.0,
    )


def _make_pulsar_params_with_px(px_value=0.5):
    """ParameterVector with a PX (parallax) parameter."""
    return make_params(
        names=["F0", "PX"],
        values=[200.0, px_value],
        units=("Hz", "mas"),
    )


# ===================================================================
# TestGlobalParams
# ===================================================================


class TestGlobalParams:
    def test_empty(self):
        gp = GlobalParams.empty()
        assert gp.n_params == 0
        assert gp.names == ()

    def test_add_params(self):
        gp = GlobalParams.empty().add_params(["a", "b", "c"], [1.0, 2.0, 3.0])
        assert gp.n_params == 3
        assert gp.names == ("a", "b", "c")
        assert jnp.isclose(gp.param_value("b"), 2.0)

    def test_incremental_build(self):
        gp = GlobalParams.empty()
        gp = gp.add_params(["x"], [10.0])
        gp = gp.add_params(["y", "z"], [20.0, 30.0])
        assert gp.n_params == 3
        assert jnp.isclose(gp.param_value("x"), 10.0)
        assert jnp.isclose(gp.param_value("z"), 30.0)

    def test_with_value(self):
        gp = GlobalParams.empty().add_params(["a", "b"], [1.0, 2.0])
        gp2 = gp.with_value("a", 99.0)
        # New instance has updated value
        assert jnp.isclose(gp2.param_value("a"), 99.0)
        # Original unchanged
        assert jnp.isclose(gp.param_value("a"), 1.0)

    def test_duplicate_raises(self):
        gp = GlobalParams.empty().add_params(["a", "b"], [1.0, 2.0])
        with pytest.raises(ValueError, match="already registered"):
            gp.add_params(["b", "c"], [3.0, 4.0])

    def test_length_mismatch_raises(self):
        gp = GlobalParams.empty()
        with pytest.raises(ValueError, match="same length"):
            gp.add_params(["a", "b"], [1.0])


# ===================================================================
# TestFplusFcross
# ===================================================================


class TestFplusFcross:
    def test_gw_at_z_pole(self):
        """GW at north pole (gwtheta=0), pulsar on x-axis.

        From Ellis et al. (2012) Eqs. 1--3:
        sin_theta=0, cos_theta=1, omhat=(0,0,-1), denom=1.
        m·pos=0, n·pos=-1 → fplus=-0.5, fcross=0.
        """
        pos = jnp.array([1.0, 0.0, 0.0])
        fp, fc = fplus_fcross(pos, jnp.float64(0.0), jnp.float64(0.0))
        assert jnp.isclose(fp, -0.5, rtol=1e-12)
        assert jnp.isclose(fc, 0.0, atol=1e-15)

    def test_known_geometry(self):
        """GW at (gwtheta=pi/4, gwphi=0), pulsar at z-axis.

        Hand-computed from Ellis et al. (2012) Eqs. 1--3.
        """
        pos = jnp.array([0.0, 0.0, 1.0])
        gwtheta = jnp.float64(jnp.pi / 4)
        gwphi = jnp.float64(0.0)

        fp, fc = fplus_fcross(pos, gwtheta, gwphi)

        # Hand calculation:
        # sin_phi=0, cos_phi=1, sin_theta=sqrt(2)/2, cos_theta=sqrt(2)/2
        # m_dot_pos = 0*0 - 1*0 = 0
        # n_dot_pos = -cos_theta*1*0 - cos_theta*0*0 + sin_theta*1 = sin_theta
        # omhat_dot_pos = -sin_theta*1*0 - sin_theta*0*0 - cos_theta*1 = -cos_theta
        # denom = 1 - cos_theta
        # fplus = 0.5*(0 - sin_theta^2) / (1 - cos_theta)
        # fcross = 0
        st = np.sqrt(2) / 2
        ct = np.sqrt(2) / 2
        expected_fp = 0.5 * (0 - st**2) / (1 - ct)
        expected_fc = 0.0

        assert jnp.isclose(fp, expected_fp, rtol=1e-12)
        assert jnp.isclose(fc, expected_fc, atol=1e-15)

    def test_jit(self):
        pos = jnp.array([0.0, 1.0, 0.0])
        fp, fc = jax.jit(fplus_fcross)(
            pos, jnp.float64(jnp.pi / 3), jnp.float64(1.0)
        )
        assert jnp.isfinite(fp)
        assert jnp.isfinite(fc)

    def test_grad(self):
        pos = jnp.array([0.0, 1.0, 0.0])
        grad_fp = jax.grad(
            lambda th: fplus_fcross(pos, th, jnp.float64(1.0))[0]
        )(jnp.float64(jnp.pi / 3))
        assert jnp.isfinite(grad_fp)
        assert not jnp.isclose(grad_fp, 0.0)


# ===================================================================
# TestCWDelay
# ===================================================================


class TestCWDelay:
    def test_zero_strain(self):
        """Near-zero strain → near-zero delay."""
        toa_data = _make_simple_toa_data(10)
        pos = jnp.array([0.0, 0.0, 1.0])
        gp = _make_cw_global_params(log10_h=-300.0)
        delay = cw_delay(toa_data, pos, jnp.float64(1.0), gp)
        assert jnp.allclose(delay, 0.0, atol=1e-30)

    def test_periodicity(self):
        """Earth-term dominated: delay should be periodic with period 1/f0."""
        log10_fgw = -8.0
        f0 = 10**log10_fgw
        period_days = 1.0 / f0 / 86400.0

        # Two TOA sets offset by exactly one CW period
        t1 = np.array([59000.0, 59100.0, 59200.0])
        t2 = t1 + period_days
        toa1 = make_toa_data(t_mjd=t1)
        toa2 = make_toa_data(t_mjd=t2)

        pos = jnp.array([0.0, 0.0, 1.0])
        # Large distance → pulsar term oscillates much faster, averages out
        gp = _make_cw_global_params(log10_h=-14.0, log10_fgw=log10_fgw)
        d1 = cw_delay(toa1, pos, jnp.float64(1000.0), gp)
        d2 = cw_delay(toa2, pos, jnp.float64(1000.0), gp)

        # Earth term is exactly periodic; pulsar term adds a fixed offset
        # so d1 and d2 should be very close (exact for Earth-term only)
        assert jnp.allclose(d1, d2, rtol=1e-3)

    def test_grad_wrt_distance(self):
        """Gradient w.r.t. pulsar distance should be finite and non-zero."""
        toa_data = _make_simple_toa_data(10)
        pos = jnp.array([0.0, 0.0, 1.0])
        gp = _make_cw_global_params(log10_h=-14.0)

        def scalar_delay(dist):
            return jnp.sum(cw_delay(toa_data, pos, dist, gp))

        grad_d = jax.grad(scalar_delay)(jnp.float64(1.0))
        assert jnp.isfinite(grad_d)
        assert not jnp.isclose(grad_d, 0.0)

    def test_jit(self):
        toa_data = _make_simple_toa_data(10)
        pos = jnp.array([0.0, 0.0, 1.0])
        gp = _make_cw_global_params(log10_h=-14.0)

        eager = cw_delay(toa_data, pos, jnp.float64(1.0), gp)
        jitted = jax.jit(cw_delay, static_argnums=(4,))(
            toa_data, pos, jnp.float64(1.0), gp
        )
        assert jnp.allclose(eager, jitted)


# ===================================================================
# TestPowerlawPSD
# ===================================================================


class TestPowerlawPSD:
    def test_analytic_value(self):
        """At f=fyr, A=1 (log10_A=0), gamma=0: S = 1/(12*pi^2) * fyr^(-3).

        From Arzoumanian et al. (2016) Eq. 1 / Phinney (2001).
        """
        f = jnp.array([FYR])
        psd = powerlaw_psd(f, jnp.float64(0.0), jnp.float64(0.0))
        expected = 1.0 / (12.0 * jnp.pi**2) * FYR ** (-3)
        assert jnp.isclose(psd[0], expected, rtol=1e-10)

    def test_spectral_slope(self):
        """For gamma=3, PSD(2f)/PSD(f) = (1/2)^3 = 0.125."""
        f = jnp.array([1e-8])
        f2 = jnp.array([2e-8])
        gamma = jnp.float64(3.0)
        log10_A = jnp.float64(-15.0)
        ratio = powerlaw_psd(f2, log10_A, gamma) / powerlaw_psd(f, log10_A, gamma)
        assert jnp.isclose(ratio[0], 0.125, rtol=1e-10)


# ===================================================================
# TestFourierBasis
# ===================================================================


class TestFourierBasis:
    def test_shape(self):
        toas = jnp.linspace(0, 1e8, 20)
        F, freqs = fourier_basis(toas, 5, 1e8)
        assert F.shape == (20, 10)
        assert freqs.shape == (5,)

    def test_orthogonality(self):
        """F^T F should be approximately diagonal for evenly spaced TOAs."""
        n = 200
        T = 1e8
        toas = jnp.linspace(0, T, n)
        F, _ = fourier_basis(toas, 3, T)
        FtF = F.T @ F
        # Normalise to correlation matrix
        diag = jnp.diag(FtF)
        corr = FtF / jnp.sqrt(jnp.outer(diag, diag))
        off_diag = corr - jnp.eye(corr.shape[0])
        assert jnp.all(jnp.abs(off_diag) < 0.1)

    def test_frequencies(self):
        T = 3e8
        F, freqs = fourier_basis(jnp.linspace(0, T, 10), 4, T)
        assert jnp.isclose(freqs[0], 1.0 / T, rtol=1e-12)
        assert jnp.isclose(freqs[-1], 4.0 / T, rtol=1e-12)


# ===================================================================
# TestORF
# ===================================================================


class TestORF:
    def test_hd_self_correlation(self):
        """Self-correlation: HD(0) = 0.5.  Hellings & Downs (1983) Eq. 2."""
        pos = jnp.array([1.0, 0.0, 0.0])
        assert jnp.isclose(hd_orf(pos, pos), 0.5, atol=1e-6)

    def test_hd_orthogonal(self):
        """90° separation from Hellings & Downs (1983) Eq. 2.

        x = (1 - cos(90°))/2 = 0.5
        HD = 1.5*0.5*ln(0.5) - 0.25*0.5 + 0.5 ≈ -0.1449.
        """
        pos1 = jnp.array([1.0, 0.0, 0.0])
        pos2 = jnp.array([0.0, 1.0, 0.0])
        val = hd_orf(pos1, pos2)
        expected = 1.5 * 0.5 * np.log(0.5) - 0.25 * 0.5 + 0.5
        assert jnp.isclose(val, expected, rtol=1e-10)

    def test_dipole(self):
        pos1 = jnp.array([1.0, 0.0, 0.0])
        pos2 = jnp.array([0.0, 1.0, 0.0])
        assert jnp.isclose(dipole_orf(pos1, pos2), 0.0, atol=1e-15)


# ===================================================================
# TestCWInjector
# ===================================================================


class TestCWInjector:
    def test_register_params(self):
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        inj = CWInjector(positions, prefix="cw0_")
        gp = inj.register_params(GlobalParams.empty())
        assert gp.n_params == 7
        assert all(n.startswith("cw0_") for n in gp.names)

    def test_unknown_param_raises(self):
        positions = jnp.array([[1.0, 0.0, 0.0]])
        with pytest.raises(ValueError, match="Unknown CW parameters"):
            CWInjector(positions, initial_values={"bad_param": 1.0})

    def test_delay_returns_array(self):
        positions = jnp.array([[0.0, 0.0, 1.0]])
        inj = CWInjector(positions, prefix="cw0_", initial_values={"log10_h": -14.0})
        gp = inj.register_params(GlobalParams.empty())
        toa_data = _make_simple_toa_data(10)
        pp = _make_pulsar_params_with_px()

        delay = inj.delay(0, toa_data, pp, gp)
        assert delay.shape == (10,)
        assert jnp.all(jnp.isfinite(delay))

    def test_covariance_returns_none(self):
        positions = jnp.array([[1.0, 0.0, 0.0]])
        inj = CWInjector(positions)
        assert inj.covariance(0, None, None, None) is None


# ===================================================================
# TestCURNInjector
# ===================================================================


class TestCURNInjector:
    def test_register_params(self):
        inj = CURNInjector(n_components=14, T_span=1e8, prefix="gwb_")
        gp = inj.register_params(GlobalParams.empty())
        assert gp.n_params == 2
        assert "gwb_log10_A" in gp.names
        assert "gwb_gamma" in gp.names

    def test_unknown_param_raises(self):
        with pytest.raises(ValueError, match="Unknown CURN parameters"):
            CURNInjector(14, 1e8, initial_values={"bad": 1.0})

    def test_delay_returns_none(self):
        inj = CURNInjector(14, 1e8)
        assert inj.delay(0, None, None, None) is None

    def test_covariance_returns_tuple(self):
        inj = CURNInjector(n_components=5, T_span=1e8)
        gp = inj.register_params(GlobalParams.empty())
        toa_data = _make_simple_toa_data(10)
        pp = _make_pulsar_params_with_px()

        result = inj.covariance(0, toa_data, pp, gp)
        assert result is not None
        U, Phi = result
        assert U.shape == (10, 10)  # n_toas x 2*n_components
        assert Phi.shape == (10,)
        assert jnp.all(jnp.isfinite(U))
        assert jnp.all(Phi > 0)


# ===================================================================
# TestPTAConfig
# ===================================================================


class TestPTAConfig:
    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="Mismatched pulsar counts"):
            PTAConfig(
                toa_data_list=(None, None),
                timing_models=(None,),
                noise_models=(None, None),
                signal_injectors=(),
            )

    def test_n_pulsars(self):
        config = PTAConfig(
            toa_data_list=(None, None, None),
            timing_models=(None, None, None),
            noise_models=(None, None, None),
            signal_injectors=(),
        )
        assert config.n_pulsars == 3


# ===================================================================
# TestFlattenUnflatten
# ===================================================================


class TestFlattenUnflatten:
    def _make_test_data(self):
        gp = GlobalParams.empty().add_params(["a", "b"], [1.0, 2.0])
        pp0 = make_params(["x", "y", "z"], [10.0, 20.0, 30.0])
        pp1 = make_params(["u", "v"], [40.0, 50.0])
        return gp, (pp0, pp1)

    def test_round_trip(self):
        gp, pp = self._make_test_data()
        flat = flatten_params(gp, pp)
        gp2, pp2 = unflatten_params(flat, gp, pp)

        assert jnp.allclose(gp2.values, gp.values)
        assert gp2.names == gp.names
        for orig, recovered in zip(pp, pp2):
            assert jnp.allclose(recovered.values, orig.values)
            assert recovered.names == orig.names
            assert recovered.frozen_mask == orig.frozen_mask

    def test_layout_order(self):
        gp, pp = self._make_test_data()
        flat = flatten_params(gp, pp)
        expected = jnp.array([1.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0])
        assert jnp.allclose(flat, expected)

    def test_jax_differentiable(self):
        """Gradient through flatten → unflatten → param_value."""
        gp, pp = self._make_test_data()
        flat = flatten_params(gp, pp)

        def f(flat_params):
            gp2, pp2 = unflatten_params(flat_params, gp, pp)
            return gp2.param_value("a") + pp2[0].param_value("x")

        grad = jax.grad(f)(flat)
        # d/d(flat[0]) = 1 (param "a"), d/d(flat[2]) = 1 (param "x")
        expected = jnp.array([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(grad, expected)
