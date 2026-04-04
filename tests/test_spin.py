"""Tests for jaxpint.spin: Spindown phase component."""

import jax
import jax.numpy as jnp
import pytest



from jaxpint.phase_result import PhaseResult
from jaxpint.phase.spin import Spindown
from tests.helpers import make_toa_data, make_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toa_data(n_toas=5, tdb_int=59000.0, tdb_frac=None):
    return make_toa_data(n_toas, tdb_int=tdb_int, tdb_frac=tdb_frac,
                         obs_names=("GBT",), planet_positions=None)


def _make_params(f0=200.0, f1=None, f2=None, pepoch_int=59000.0, pepoch_frac=0.0):
    names = ["F0"]
    values = [f0]
    components = ["Spindown"]
    units = ["Hz"]

    if f1 is not None:
        names += ["F1"]; values += [f1]
        components += ["Spindown"]; units += ["Hz/s"]
    if f2 is not None:
        names += ["F2"]; values += [f2]
        components += ["Spindown"]; units += ["Hz/s"]

    names += ["PEPOCH"]; values += [pepoch_frac]
    components += ["Spindown"]; units += ["day"]

    return make_params(names, values, units=tuple(units),
                       components=tuple(components),
                       epoch_int_values={"PEPOCH": pepoch_int})


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
    def test_f0_only_known_dt(self):
        """F0=100 Hz, dt=1 second -> phase=100 cycles."""
        spindown = Spindown(spin_param_names=("F0",))
        # PEPOCH at 59000.0, TOA at 59000.0 + 1/86400 (= 1 second later)
        params = _make_params(f0=100.0, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=1.0 / 86400.0)
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        assert isinstance(result, PhaseResult)
        assert jnp.isclose(result.total, 100.0, rtol=1e-12)

    def test_f0_f1_quadratic(self):
        """Phase = F0*dt + F1*dt^2/2."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        f0, f1 = 200.0, -1e-10
        dt_sec = 1000.0  # 1000 seconds
        params = _make_params(f0=f0, f1=f1, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=dt_sec / 86400.0
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        expected = f0 * dt_sec + f1 * dt_sec**2 / 2.0
        assert jnp.isclose(result.total, expected, rtol=1e-12)

    def test_f0_f1_f2_cubic(self):
        """Phase = F0*dt + F1*dt^2/2 + F2*dt^3/6."""
        spindown = Spindown(spin_param_names=("F0", "F1", "F2"))
        f0, f1, f2 = 200.0, -1e-10, 1e-20
        dt_sec = 1000.0
        params = _make_params(
            f0=f0, f1=f1, f2=f2, pepoch_int=59000.0, pepoch_frac=0.0
        )
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=dt_sec / 86400.0
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        expected = f0 * dt_sec + f1 * dt_sec**2 / 2.0 + f2 * dt_sec**3 / 6.0
        assert jnp.isclose(result.total, expected, rtol=1e-12)

    def test_delay_subtracted(self):
        """Delay reduces effective dt."""
        spindown = Spindown(spin_param_names=("F0",))
        f0 = 100.0
        params = _make_params(f0=f0, pepoch_int=59000.0, pepoch_frac=0.0)
        # TOA at 2 seconds after PEPOCH
        toa_data = _make_toa_data(
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
        params = _make_params(f0=f0, pepoch_int=59000.0, pepoch_frac=0.0)
        n = 5
        fracs = jnp.arange(1, n + 1) / 86400.0  # 1, 2, 3, 4, 5 seconds
        toa_data = _make_toa_data(n_toas=n, tdb_int=59000.0, tdb_frac=fracs)
        delay = jnp.zeros(n)

        result = spindown(toa_data, params, delay)
        expected = f0 * jnp.arange(1.0, n + 1)
        assert jnp.allclose(result.total, expected, rtol=1e-12)

    def test_phase_result_normalization(self):
        """Fractional part is in [-0.5, 0.5)."""
        spindown = Spindown(spin_param_names=("F0",))
        params = _make_params(f0=100.0, pepoch_int=59000.0, pepoch_frac=0.0)
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=59000.0, tdb_frac=1.0 / 86400.0
        )
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        assert jnp.all(result.frac >= -0.5)
        assert jnp.all(result.frac < 0.5)

    def test_zero_dt_gives_zero_phase(self):
        """TOA at PEPOCH with no delay -> zero phase."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = _make_params(f0=200.0, f1=-1e-15, pepoch_int=59000.0, pepoch_frac=0.5)
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.5)
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

        params = _make_params(f0=1.0, pepoch_int=pepoch_int, pepoch_frac=pepoch_frac)
        toa_data = _make_toa_data(n_toas=1, tdb_int=toa_int, tdb_frac=toa_frac)
        delay = jnp.zeros(1)

        result = spindown(toa_data, params, delay)
        # Expected: F0 * ((61000 - 50000) + 1e-14) * 86400
        dt_expected = (toa_int - pepoch_int + tiny_frac) * 86400.0
        expected_phase = 1.0 * dt_expected
        assert jnp.isclose(result.total, expected_phase, rtol=1e-12)


# ===========================================================================
# JIT tests
# ===========================================================================

class TestJIT:
    def test_jit_call(self):
        """Spindown.__call__ works under jax.jit."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = _make_params(f0=200.0, f1=-1e-15)
        toa_data = _make_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        jitted = jax.jit(spindown)
        result = jitted(toa_data, params, delay)
        assert isinstance(result, PhaseResult)
        assert result.int.shape == (toa_data.n_toas,)

    def test_jit_same_trace(self):
        """Same spin_param_names does not retrace."""
        spindown = Spindown(spin_param_names=("F0",))
        params = _make_params(f0=100.0)
        toa_data = _make_toa_data(n_toas=3)
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
    def test_grad_wrt_f0(self):
        """d(sum(phase))/dF0 ~ sum(dt)."""
        spindown = Spindown(spin_param_names=("F0",))
        params = _make_params(f0=100.0, pepoch_int=59000.0, pepoch_frac=0.0)
        dt_secs = jnp.array([1.0, 2.0, 3.0])
        toa_data = _make_toa_data(
            n_toas=3, tdb_int=59000.0, tdb_frac=dt_secs / 86400.0
        )
        delay = jnp.zeros(3)

        def loss(p):
            return spindown(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        # d(phase)/dF0 = dt for each TOA, so d(sum)/dF0 = sum(dt)
        f0_idx = params.param_index("F0")
        expected_grad = jnp.sum(dt_secs)
        assert jnp.isclose(grads.values[f0_idx], expected_grad, rtol=1e-10)

    def test_grad_wrt_f1(self):
        """d(sum(phase))/dF1 ~ sum(dt^2 / 2)."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = _make_params(f0=100.0, f1=-1e-15, pepoch_int=59000.0, pepoch_frac=0.0)
        dt_secs = jnp.array([100.0, 200.0, 300.0])
        toa_data = _make_toa_data(
            n_toas=3, tdb_int=59000.0, tdb_frac=dt_secs / 86400.0
        )
        delay = jnp.zeros(3)

        def loss(p):
            return spindown(toa_data, p, delay).total.sum()

        grads = jax.grad(loss)(params)
        f1_idx = params.param_index("F1")
        expected_grad = jnp.sum(dt_secs**2 / 2.0)
        assert jnp.isclose(grads.values[f1_idx], expected_grad, rtol=1e-8)

    def test_grad_finite(self):
        """All gradients are finite."""
        spindown = Spindown(spin_param_names=("F0", "F1"))
        params = _make_params(f0=200.0, f1=-1e-15)
        toa_data = _make_toa_data()
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
        params = _make_params(f0=200.0, f1=-1e-14, pepoch_int=59000.0, pepoch_frac=0.0)

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
        params = _make_params(f0=200.0, f1=-1e-14, pepoch_int=59000.0, pepoch_frac=0.0)

        shifted = spindown.change_pepoch(params, 59100.0, 0.0)
        restored = spindown.change_pepoch(shifted, 59000.0, 0.0)

        assert jnp.isclose(
            restored.param_value("F0"), params.param_value("F0"), rtol=1e-10
        )
        assert jnp.isclose(
            restored.param_value("F1"), params.param_value("F1"), rtol=1e-10
        )
