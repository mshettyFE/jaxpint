"""Tests for jaxpint.spin: Spindown phase component."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest



from jaxpint.constants import SECS_PER_DAY
from jaxpint.dual_float import DualFloat
from jaxpint.phase.spin import Spindown
from tests.helpers import make_gbt_toa_data, make_spindown_params


# ===========================================================================
# Construction tests
# ===========================================================================

class TestConstruction:
    def test_f0_only(self):
        s = Spindown(spin_param_names=("F0",))
        assert s.spin_param_names == ("F0",)

    def test_f0_f1_f2(self):
        s = Spindown(spin_param_names=("F0", "F1", "F2"))
        assert len(s.spin_param_names) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            Spindown(spin_param_names=())

    def test_missing_f0_raises(self):
        with pytest.raises(ValueError, match="F0"):
            Spindown(spin_param_names=("F1",))

    def test_custom_pepoch_name(self):
        s = Spindown(spin_param_names=("F0",), pepoch_name="MYEPOCH")
        assert s.pepoch_name == "MYEPOCH"


# ===========================================================================
# Pytree tests
# ===========================================================================

class TestPytree:
    def test_zero_dynamic_leaves(self):
        s = Spindown(spin_param_names=("F0", "F1"))
        leaves, _ = jax.tree.flatten(s)
        assert len(leaves) == 0

    def test_pytree_roundtrip(self):
        s = Spindown(spin_param_names=("F0", "F1"))
        leaves, treedef = jax.tree.flatten(s)
        s2 = jax.tree.unflatten(treedef, leaves)
        assert s2.spin_param_names == s.spin_param_names
        assert s2.pepoch_name == s.pepoch_name


# ===========================================================================
# Phase computation tests
# ===========================================================================

class TestSpindownPhase:
    @pytest.mark.parametrize("coeffs, dt_sec, expected_fn", [
        pytest.param(
            {"f0": 100.0}, 1.0,
            lambda dt, c: c["f0"] * dt,
            id="f0_only",
        ),
        pytest.param(
            {"f0": 200.0, "f1": -1e-10}, 1000.0,
            lambda dt, c: c["f0"] * dt + c["f1"] * dt**2 / 2.0,
            id="f0_f1_quadratic",
        ),
        pytest.param(
            {"f0": 200.0, "f1": -1e-10, "f2": 1e-20}, 1000.0,
            lambda dt, c: c["f0"] * dt + c["f1"] * dt**2 / 2.0 + c["f2"] * dt**3 / 6.0,
            id="f0_f1_f2_cubic",
        ),
    ])
    def test_polynomial_phase(self, coeffs, dt_sec, expected_fn):
        """Phase matches analytic Taylor expansion."""
        spin_names = tuple(coeffs.keys())
        spindown = Spindown(spin_param_names=tuple(n.upper() for n in spin_names))
        params = make_spindown_params(**coeffs, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=dt_sec / 86400.0
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        expected = expected_fn(dt_sec, coeffs)
        assert isinstance(result, DualFloat)
        assert jnp.isclose(result.total, expected, rtol=1e-12)

    def test_delay_subtracted(self):
        """Delay reduces effective dt."""
        spindown = Spindown(spin_param_names=("F0",))
        f0 = 100.0
        params = make_spindown_params(f0=f0, pepoch_int=59000.0, pepoch_frac=0.0)
        # TOA at 2 seconds after PEPOCH
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=2.0 / 86400.0
        )
        delay = jnp.array([0.5])  # 0.5 seconds delay

        result = spindown(toa_data, params, delay)
        expected = f0 * (2.0 - 0.5)  # F0 * (dt - delay)
        assert jnp.isclose(result.total, expected, rtol=1e-12)

    def test_multiple_toas(self):
        """Vectorised over multiple TOAs."""
        spindown = Spindown(spin_param_names=("F0",))
        f0 = 100.0
        params = make_spindown_params(f0=f0, pepoch_int=59000.0, pepoch_frac=0.0)
        n = 5
        fracs = jnp.arange(1, n + 1) / 86400.0  # 1, 2, 3, 4, 5 seconds
        toa_data = make_gbt_toa_data(n_toas=n, tdb_int=59000.0, tdb_frac=fracs)
        delay = jnp.zeros(n)

        result = spindown(toa_data, params, delay)
        expected = f0 * jnp.arange(1.0, n + 1)
        assert jnp.allclose(result.total, expected, rtol=1e-12)

    def test_phase_result_normalization(self):
        """Fractional part is in [-0.5, 0.5)."""
        spindown = Spindown(spin_param_names=("F0",))
        params = make_spindown_params(f0=100.0, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=1.0 / 86400.0
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        assert jnp.all(result.frac >= -0.5)
        assert jnp.all(result.frac < 0.5)

    def test_zero_dt_gives_zero_phase(self):
        """TOA at PEPOCH with no delay -> zero phase."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = make_spindown_params(f0=200.0, f1=-1e-15, pepoch_int=59000.0, pepoch_frac=0.5)
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.5)
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        assert jnp.isclose(result.total, 0.0, atol=1e-15)


# ===========================================================================
# Precision tests
# ===========================================================================

class TestPrecision:
    def test_dt_precision_30yr_baseline(self):
        """Over a 30-year baseline, dt should preserve sub-nanosecond precision."""
        spindown = Spindown(spin_param_names=("F0",))
        pepoch_int = 50000.0
        pepoch_frac = 0.0
        # ~30 years later
        toa_int = 61000.0
        # Add a tiny fractional offset: 1e-14 days ~ 0.86 ps
        tiny_frac = 1e-14
        toa_frac = tiny_frac

        params = make_spindown_params(f0=1.0, pepoch_int=pepoch_int, pepoch_frac=pepoch_frac)
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=toa_int, tdb_frac=toa_frac)
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        # Expected: F0 * ((61000 - 50000) + 1e-14) * 86400
        dt_expected = (toa_int - pepoch_int + tiny_frac) * 86400.0
        expected_phase = 1.0 * dt_expected
        assert jnp.isclose(result.total, expected_phase, rtol=1e-12)

    def test_realistic_msp_30yr_high_order(self):
        """End-to-end Spindown precision check at NANOGrav-realistic scale.

        Exercises the same Horner precision regime as
        test_high_order_matches_longdouble in test_utils.py, but through
        the full Spindown.__call__ path. Guards against integration-level
        regressions of the KBN compensation.
        """
        raw_f = [600.0, -1.0e-15, 1.0e-25]  # F0, F1, F2 (helper supports up to F2)
        pepoch_int = 50000.0
        pepoch_frac = 0.0
        # 30-year baseline with day count % 7 != 0 to stress non-trivial dividers
        toa_int = 60957.0
        toa_frac = 0.314

        spindown = Spindown(spin_param_names=("F0", "F1", "F2"))
        params = make_spindown_params(
            f0=raw_f[0], f1=raw_f[1], f2=raw_f[2],
            pepoch_int=pepoch_int, pepoch_frac=pepoch_frac,
        )
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=toa_int, tdb_frac=toa_frac,
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)

        # Longdouble reference: sum_k F_k * x^{k+1} / (k+1)!
        x = (np.longdouble(toa_int - pepoch_int) * np.longdouble(SECS_PER_DAY)
             + np.longdouble(toa_frac - pepoch_frac) * np.longdouble(SECS_PER_DAY))
        ld_expected = np.longdouble(0)
        for k, f in enumerate(raw_f):
            ld_expected += np.longdouble(f) * x ** (k + 1) / np.longdouble(math.factorial(k + 1))

        actual = np.longdouble(float(result.int[0])) + np.longdouble(float(result.frac[0]))
        # Expect ~1e-7 cycles (~0.2 ns at 600 Hz), same regime as the unit
        # test. Tolerance 1e-6 to be robust against longdouble platform variation.
        assert abs(float(actual - ld_expected)) < 1e-6


# ===========================================================================
# JIT tests
# ===========================================================================

class TestJIT:
    def test_jit_call(self):
        """Spindown.__call__ works under jax.jit."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = make_spindown_params(f0=200.0, f1=-1e-15)
        toa_data = make_gbt_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        jitted = jax.jit(spindown)
        result = jitted(toa_data, params, delay)
        assert isinstance(result, DualFloat)
        assert result.int.shape == (toa_data.n_toas,)

    def test_jit_same_trace(self):
        """Same spin_param_names does not retrace."""
        spindown = Spindown(spin_param_names=("F0",))
        params = make_spindown_params(f0=100.0)
        toa_data = make_gbt_toa_data(n_toas=3)
        delay = jnp.zeros(3)

        jitted = jax.jit(spindown)
        r1 = jitted(toa_data, params, delay)

        # Change param value but not structure -> no retrace
        params2 = params.with_value("F0", 200.0)
        r2 = jitted(toa_data, params2, delay)

        assert not jnp.array_equal(r1.total, r2.total)


# ===========================================================================
# Gradient tests
# ===========================================================================

class TestGrad:
    @pytest.mark.parametrize("spin_names, coeffs, param_name, dt_secs, expected_grad_fn, rtol", [
        pytest.param(
            ("F0",), {"f0": 100.0}, "F0",
            jnp.array([1.0, 2.0, 3.0]),
            lambda dt: jnp.sum(dt),
            1e-10,
            id="grad_wrt_f0",
        ),
        pytest.param(
            ("F0", "F1"), {"f0": 100.0, "f1": -1e-15}, "F1",
            jnp.array([100.0, 200.0, 300.0]),
            lambda dt: jnp.sum(dt**2 / 2.0),
            1e-8,
            id="grad_wrt_f1",
        ),
    ])
    def test_grad_wrt_param(self, spin_names, coeffs, param_name, dt_secs, expected_grad_fn, rtol):
        """d(sum(phase))/d(param) matches analytic expectation."""
        spindown = Spindown(spin_param_names=spin_names)
        params = make_spindown_params(**coeffs, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = make_gbt_toa_data(
            n_toas=len(dt_secs), tdb_int=59000.0, tdb_frac=dt_secs / 86400.0
        )
        delay = jnp.zeros(len(dt_secs))

        def loss(p):
            return spindown(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        idx = params.param_index(param_name)
        assert jnp.isclose(grads.values[idx], expected_grad_fn(dt_secs), rtol=rtol)

    def test_grad_finite(self):
        """All gradients are finite."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = make_spindown_params(f0=200.0, f1=-1e-15)
        toa_data = make_gbt_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        def loss(p):
            return spindown(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))


# ===========================================================================
# change_pepoch tests
# ===========================================================================

class TestChangePepoch:
    def test_frequency_invariant(self):
        """Instantaneous frequency at any time is invariant under PEPOCH change.

        Spindown phase is relative to PEPOCH (constant term = 0), so the
        absolute phase changes when PEPOCH moves.  But the frequency
        (d(phase)/dt) at any given time must be the same.
        """
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = make_spindown_params(f0=200.0, f1=-1e-14, pepoch_int=59000.0, pepoch_frac=0.0)

        # Shift PEPOCH by 50 days
        new_params = spindown.change_pepoch(params, 59050.0, 0.0)

        # Frequency at a test time (59100 days) should be the same
        # freq(t) = F0 + F1 * (t - PEPOCH) in seconds
        t_sec_before = (59100.0 - 59000.0) * 86400.0
        freq_before = params.param_value("F0") + params.param_value("F1") * t_sec_before

        t_sec_after = (59100.0 - 59050.0) * 86400.0
        freq_after = new_params.param_value("F0") + new_params.param_value("F1") * t_sec_after

        assert jnp.isclose(freq_before, freq_after, rtol=1e-12)

    def test_roundtrip(self):
        """Shift forward then back recovers original F-values."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = make_spindown_params(f0=200.0, f1=-1e-14, pepoch_int=59000.0, pepoch_frac=0.0)

        shifted = spindown.change_pepoch(params, 59100.0, 0.0)
        restored = spindown.change_pepoch(shifted, 59000.0, 0.0)

        assert jnp.isclose(
            restored.param_value("F0"), params.param_value("F0"), rtol=1e-10
        )
        assert jnp.isclose(
            restored.param_value("F1"), params.param_value("F1"), rtol=1e-10
        )
