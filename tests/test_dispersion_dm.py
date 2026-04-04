"""Tests for jaxpint.dispersion_dm: DispersionDM delay component."""

import jax
import jax.numpy as jnp
import pytest


from jaxpint.constants import DMCONST
from jaxpint.delay.dispersion_dm import DispersionDM
from tests.helpers import make_gbt_toa_data, make_dispersion_dm_params


# ===========================================================================
# Construction tests
# ===========================================================================

class TestConstruction:
    def test_dm_only(self):
        d = DispersionDM(dm_param_names=("DM",))
        assert d.dm_param_names == ("DM",)
        assert d.dmepoch_name == "DMEPOCH"

    def test_dm_dm1_dm2(self):
        d = DispersionDM(dm_param_names=("DM", "DM1", "DM2"))
        assert len(d.dm_param_names) == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            DispersionDM(dm_param_names=())

    def test_missing_dm_raises(self):
        with pytest.raises(ValueError, match="DM"):
            DispersionDM(dm_param_names=("DM1",))

    def test_custom_epoch_name(self):
        d = DispersionDM(dm_param_names=("DM",), dmepoch_name="MYEPOCH")
        assert d.dmepoch_name == "MYEPOCH"


# ===========================================================================
# Pytree tests
# ===========================================================================

class TestPytree:
    def test_zero_dynamic_leaves(self):
        d = DispersionDM(dm_param_names=("DM", "DM1"))
        leaves, _ = jax.tree.flatten(d)
        assert len(leaves) == 0

    def test_pytree_roundtrip(self):
        d = DispersionDM(dm_param_names=("DM", "DM1"))
        leaves, treedef = jax.tree.flatten(d)
        d2 = jax.tree.unflatten(treedef, leaves)
        assert d2.dm_param_names == d.dm_param_names
        assert d2.dmepoch_name == d.dmepoch_name


# ===========================================================================
# Delay computation tests
# ===========================================================================

class TestDispersionDelay:
    @pytest.mark.parametrize("dm_names, coeffs, dt_yr, dm_expected_fn", [
        pytest.param(
            ("DM",), {"dm": 15.0}, 0.0,
            lambda c, dt: c["dm"],
            id="constant_dm",
        ),
        pytest.param(
            ("DM", "DM1"), {"dm": 15.0, "dm1": 0.1}, 1.0,
            lambda c, dt: c["dm"] + c["dm1"] * dt,
            id="dm_dm1_linear",
        ),
        pytest.param(
            ("DM", "DM1", "DM2"), {"dm": 15.0, "dm1": 0.1, "dm2": 0.02}, 2.0,
            lambda c, dt: c["dm"] + c["dm1"] * dt + c["dm2"] * dt**2 / 2.0,
            id="dm_dm1_dm2_quadratic",
        ),
    ])
    def test_polynomial_dm(self, dm_names, coeffs, dt_yr, dm_expected_fn):
        """DM Taylor expansion gives expected delay."""
        disp = DispersionDM(dm_param_names=dm_names)
        freq = 1400.0
        dt_days = dt_yr * 365.25
        params = make_dispersion_dm_params(**coeffs, dmepoch_int=59000.0, dmepoch_frac=0.0)
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=59000.0 + dt_days, tdb_frac=0.0, freq=freq
        )
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        expected = dm_expected_fn(coeffs, dt_yr) * DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_frequency_dependence(self):
        """Delay scales as 1/freq^2."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = make_dispersion_dm_params(dm=15.0)

        freq_lo, freq_hi = 800.0, 1400.0
        toa_lo = make_gbt_toa_data(n_toas=1, freq=freq_lo)
        toa_hi = make_gbt_toa_data(n_toas=1, freq=freq_hi)
        delay = jnp.zeros(1)

        d_lo = disp(toa_lo, params, delay)
        d_hi = disp(toa_hi, params, delay)

        ratio = d_lo[0] / d_hi[0]
        expected_ratio = (freq_hi / freq_lo) ** 2
        assert jnp.isclose(ratio, expected_ratio, rtol=1e-12)

    def test_multiple_toas(self):
        """Vectorised over multiple TOAs with different frequencies."""
        disp = DispersionDM(dm_param_names=("DM",))
        dm = 15.0
        freqs = jnp.array([800.0, 1000.0, 1400.0, 2000.0])
        params = make_dispersion_dm_params(dm=dm)
        toa_data = make_gbt_toa_data(n_toas=4, freq=freqs)
        delay = jnp.zeros(4)

        result = disp(toa_data, params, delay)
        expected = dm * DMCONST / freqs ** 2
        assert jnp.allclose(result, expected, rtol=1e-12)

    def test_zero_dt_gives_base_dm(self):
        """TOA at DMEPOCH with DM1 -> DM(t) = DM."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        dm, dm1 = 20.0, 1.0
        freq = 1400.0
        params = make_dispersion_dm_params(dm=dm, dm1=dm1, dmepoch_int=59000.0, dmepoch_frac=0.5)
        toa_data = make_gbt_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.5, freq=freq)
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        expected = dm * DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_acc_delay_ignored(self):
        """Accumulated delay does not affect dispersion."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = make_dispersion_dm_params(dm=15.0)
        toa_data = make_gbt_toa_data(n_toas=1, freq=1400.0)

        result_no_delay = disp(toa_data, params, jnp.zeros(1))
        result_with_delay = disp(toa_data, params, jnp.array([0.5]))
        assert jnp.isclose(result_no_delay[0], result_with_delay[0])


# ===========================================================================
# Precision tests
# ===========================================================================

class TestPrecision:
    def test_dt_precision_large_baseline(self):
        """Over a 20-year baseline, dt_yr should be precise."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        dm, dm1 = 15.0, 0.001  # small DM1
        freq = 1400.0

        dmepoch_int = 50000.0
        toa_int = 57305.0  # ~20 years later
        tiny_frac = 1e-12  # ~0.086 ns in days

        params = make_dispersion_dm_params(
            dm=dm, dm1=dm1,
            dmepoch_int=dmepoch_int, dmepoch_frac=0.0,
        )
        toa_data = make_gbt_toa_data(
            n_toas=1, tdb_int=toa_int, tdb_frac=tiny_frac, freq=freq
        )
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        dt_yr = (toa_int - dmepoch_int + tiny_frac) / 365.25
        dm_expected = dm + dm1 * dt_yr
        expected = dm_expected * DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)


# ===========================================================================
# JIT tests
# ===========================================================================

class TestJIT:
    def test_jit_call(self):
        """DispersionDM.__call__ works under jax.jit."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        params = make_dispersion_dm_params(dm=15.0, dm1=0.1)
        toa_data = make_gbt_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        jitted = jax.jit(disp)
        result = jitted(toa_data, params, delay)
        assert result.shape == (toa_data.n_toas,)

    def test_jit_same_trace(self):
        """Same dm_param_names does not retrace."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = make_dispersion_dm_params(dm=15.0)
        toa_data = make_gbt_toa_data(n_toas=3)
        delay = jnp.zeros(3)

        jitted = jax.jit(disp)
        r1 = jitted(toa_data, params, delay)

        params2 = params.with_value("DM", 30.0)
        r2 = jitted(toa_data, params2, delay)

        assert not jnp.array_equal(r1, r2)


# ===========================================================================
# Gradient tests
# ===========================================================================

class TestGrad:
    @pytest.mark.parametrize("dm_names, coeffs, param_name, freqs, dt_days, expected_grad_fn, rtol", [
        pytest.param(
            ("DM",), {"dm": 15.0}, "DM",
            jnp.array([800.0, 1400.0, 2000.0]),
            jnp.zeros(3),
            lambda f, dt: jnp.sum(DMCONST / f ** 2),
            1e-10,
            id="grad_wrt_dm",
        ),
        pytest.param(
            ("DM", "DM1"), {"dm": 15.0, "dm1": 0.1}, "DM1",
            jnp.full(3, 1400.0),
            jnp.array([365.25, 730.5, 1095.75]),
            lambda f, dt: jnp.sum((dt / 365.25) * DMCONST / f ** 2),
            1e-8,
            id="grad_wrt_dm1",
        ),
    ])
    def test_grad_wrt_param(self, dm_names, coeffs, param_name, freqs, dt_days, expected_grad_fn, rtol):
        """d(sum(delay))/d(param) matches analytic expectation."""
        disp = DispersionDM(dm_param_names=dm_names)
        params = make_dispersion_dm_params(**coeffs, dmepoch_int=59000.0, dmepoch_frac=0.0)
        toa_data = make_gbt_toa_data(
            n_toas=len(freqs), tdb_int=59000.0 + dt_days, tdb_frac=0.0, freq=freqs
        )
        delay = jnp.zeros(len(freqs))

        def loss(p):
            return disp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        idx = params.param_index(param_name)
        assert jnp.isclose(grads.values[idx], expected_grad_fn(freqs, dt_days), rtol=rtol)

    def test_grad_finite(self):
        """All gradients are finite."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        params = make_dispersion_dm_params(dm=15.0, dm1=0.1)
        toa_data = make_gbt_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        def loss(p):
            return disp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values))


# ===========================================================================
# Cross-validation against PINT (oracle tests)
# ===========================================================================

class TestPINTOracle:
    """Compare JaxPINT dispersion delay against PINT's implementation."""

    @pytest.fixture
    def pint_setup(self):
        """Build a PINT model with DM and compute its dispersion delay.

        Both PINT and JaxPINT use barycentric (Doppler-corrected) frequency
        for dispersion calculations.  We compare against PINT's
        ``dispersion_type_delay`` which internally calls
        ``barycentric_radio_freq``.
        """
        from io import StringIO
        import astropy.units as u
        import numpy as np
        from pint.models import get_model
        from pint.simulation import make_fake_toas_uniform
        from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params

        par = """
PSR           J1234+5678
RAJ           12:34:56.789
DECJ          +56:07:08.12
F0            100.0
F1            -1e-15
PEPOCH        55000
DM            15.0
DM1           0.1
DMEPOCH       55000
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  N
PLANET_SHAPIRO       N
"""
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            startMJD=54500, endMJD=55500,
            ntoas=20, model=model, freq=1400.0,
            add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        # Compute PINT's dispersion delay (uses barycentric frequency internally)
        dm_comp = model.components["DispersionDM"]
        pint_delay = np.array(
            dm_comp.dispersion_type_delay(toas).to("s").value,
            dtype=np.float64,
        )

        # Convert to JaxPINT types
        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model)

        return toa_data, params, pint_delay

    def test_matches_pint(self, pint_setup):
        """JaxPINT dispersion delay matches PINT within float64 tolerance."""
        toa_data, params, pint_delay = pint_setup

        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        jax_delay = disp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.allclose(
            jax_delay, jnp.asarray(pint_delay), rtol=1e-10, atol=1e-15,
        )
