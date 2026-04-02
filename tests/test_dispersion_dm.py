"""Tests for jaxpint.dispersion_dm: DispersionDM delay component."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from jaxpint.types import TOAData, ParameterVector
from jaxpint.dispersion_dm import DispersionDM, _DMCONST


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_toa_data(n_toas=5, tdb_int=59000.0, tdb_frac=None, freq=1400.0):
    """Minimal TOAData with controllable TDB times and frequencies."""
    if tdb_frac is None:
        tdb_frac = jnp.linspace(0.1, 0.9, n_toas)
    else:
        tdb_frac = jnp.broadcast_to(jnp.asarray(tdb_frac), (n_toas,))

    tdb_int_arr = jnp.full(n_toas, tdb_int)
    freq_arr = jnp.broadcast_to(jnp.asarray(freq), (n_toas,))

    return TOAData(
        mjd_int=tdb_int_arr,
        mjd_frac=tdb_frac,
        tdb_int=tdb_int_arr,
        tdb_frac=tdb_frac,
        error=jnp.ones(n_toas) * 1e-6,
        freq=freq_arr,
        delta_pulse_number=jnp.zeros(n_toas),
        ssb_obs_pos=jnp.zeros((n_toas, 3)),
        ssb_obs_vel=jnp.zeros((n_toas, 3)),
        obs_sun_pos=jnp.zeros((n_toas, 3)),
        obs_indices=jnp.zeros(n_toas, dtype=jnp.int32),
        flag_masks={},
        planet_positions=None,
        dm_values=None,
        dm_errors=None,
        n_toas=n_toas,
        obs_names=("GBT",),
    )


def _make_params(dm=15.0, dm1=None, dm2=None,
                 dmepoch_int=59000.0, dmepoch_frac=0.0):
    """Minimal ParameterVector with DM params and DMEPOCH."""
    names = ["DM"]
    values = [dm]
    components = ["DispersionDM"]

    if dm1 is not None:
        names.append("DM1")
        values.append(dm1)
        components.append("DispersionDM")
    if dm2 is not None:
        names.append("DM2")
        values.append(dm2)
        components.append("DispersionDM")

    names.append("DMEPOCH")
    values.append(dmepoch_frac)
    components.append("DispersionDM")

    n = len(names)
    names = tuple(names)
    return ParameterVector(
        values=jnp.array(values),
        frozen_mask=(False,) * n,
        names=names,
        units=("pc cm^-3",) + ("pc cm^-3/yr",) * (n - 2) + ("day",),
        components=tuple(components),
        _name_to_index={name: i for i, name in enumerate(names)},
        bounds=((None, None),) * n,
        epoch_int_values={"DMEPOCH": dmepoch_int},
    )


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
    def test_constant_dm_known_value(self):
        """DM=15, freq=1400 MHz -> known delay."""
        disp = DispersionDM(dm_param_names=("DM",))
        dm = 15.0
        freq = 1400.0
        params = _make_params(dm=dm)
        toa_data = _make_toa_data(n_toas=1, freq=freq)
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        expected = dm * _DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_dm_dm1_linear(self):
        """DM(t) = DM + DM1 * dt_yr -> linear increase."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        dm, dm1 = 15.0, 0.1  # pc/cm^3 and pc/cm^3/yr
        freq = 1400.0
        # 1 Julian year after DMEPOCH
        dt_days = 365.25
        params = _make_params(dm=dm, dm1=dm1, dmepoch_int=59000.0, dmepoch_frac=0.0)
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=59000.0 + dt_days, tdb_frac=0.0, freq=freq
        )
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        # dt_yr = 1.0, so DM(t) = 15.0 + 0.1 * 1.0 = 15.1
        dm_expected = dm + dm1 * 1.0
        expected = dm_expected * _DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_dm_dm1_dm2_quadratic(self):
        """DM(t) = DM + DM1*dt + DM2*dt^2/2."""
        disp = DispersionDM(dm_param_names=("DM", "DM1", "DM2"))
        dm, dm1, dm2 = 15.0, 0.1, 0.02
        freq = 1400.0
        dt_yr = 2.0
        dt_days = dt_yr * 365.25
        params = _make_params(
            dm=dm, dm1=dm1, dm2=dm2,
            dmepoch_int=59000.0, dmepoch_frac=0.0,
        )
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=59000.0 + dt_days, tdb_frac=0.0, freq=freq
        )
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        dm_expected = dm + dm1 * dt_yr + dm2 * dt_yr ** 2 / 2.0
        expected = dm_expected * _DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_frequency_dependence(self):
        """Delay scales as 1/freq^2."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = _make_params(dm=15.0)

        freq_lo, freq_hi = 800.0, 1400.0
        toa_lo = _make_toa_data(n_toas=1, freq=freq_lo)
        toa_hi = _make_toa_data(n_toas=1, freq=freq_hi)
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
        params = _make_params(dm=dm)
        toa_data = _make_toa_data(n_toas=4, freq=freqs)
        delay = jnp.zeros(4)

        result = disp(toa_data, params, delay)
        expected = dm * _DMCONST / freqs ** 2
        assert jnp.allclose(result, expected, rtol=1e-12)

    def test_zero_dt_gives_base_dm(self):
        """TOA at DMEPOCH with DM1 -> DM(t) = DM."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        dm, dm1 = 20.0, 1.0
        freq = 1400.0
        params = _make_params(dm=dm, dm1=dm1, dmepoch_int=59000.0, dmepoch_frac=0.5)
        toa_data = _make_toa_data(n_toas=1, tdb_int=59000.0, tdb_frac=0.5, freq=freq)
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        expected = dm * _DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)

    def test_acc_delay_ignored(self):
        """Accumulated delay does not affect dispersion."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = _make_params(dm=15.0)
        toa_data = _make_toa_data(n_toas=1, freq=1400.0)

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

        params = _make_params(
            dm=dm, dm1=dm1,
            dmepoch_int=dmepoch_int, dmepoch_frac=0.0,
        )
        toa_data = _make_toa_data(
            n_toas=1, tdb_int=toa_int, tdb_frac=tiny_frac, freq=freq
        )
        delay = jnp.zeros(1)

        result = disp(toa_data, params, delay)
        dt_yr = (toa_int - dmepoch_int + tiny_frac) / 365.25
        dm_expected = dm + dm1 * dt_yr
        expected = dm_expected * _DMCONST / freq ** 2
        assert jnp.isclose(result[0], expected, rtol=1e-12)


# ===========================================================================
# JIT tests
# ===========================================================================

class TestJIT:
    def test_jit_call(self):
        """DispersionDM.__call__ works under jax.jit."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        params = _make_params(dm=15.0, dm1=0.1)
        toa_data = _make_toa_data()
        delay = jnp.zeros(toa_data.n_toas)

        jitted = jax.jit(disp)
        result = jitted(toa_data, params, delay)
        assert result.shape == (toa_data.n_toas,)

    def test_jit_same_trace(self):
        """Same dm_param_names does not retrace."""
        disp = DispersionDM(dm_param_names=("DM",))
        params = _make_params(dm=15.0)
        toa_data = _make_toa_data(n_toas=3)
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
    def test_grad_wrt_dm(self):
        """d(sum(delay))/dDM = sum(DMCONST / freq^2)."""
        disp = DispersionDM(dm_param_names=("DM",))
        freqs = jnp.array([800.0, 1400.0, 2000.0])
        params = _make_params(dm=15.0)
        toa_data = _make_toa_data(n_toas=3, freq=freqs)
        delay = jnp.zeros(3)

        def loss(p):
            return disp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        dm_idx = params.param_index("DM")
        expected_grad = jnp.sum(_DMCONST / freqs ** 2)
        assert jnp.isclose(grads.values[dm_idx], expected_grad, rtol=1e-10)

    def test_grad_wrt_dm1(self):
        """d(sum(delay))/dDM1 = sum(dt_yr * DMCONST / freq^2)."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        freq = 1400.0
        dt_days = jnp.array([365.25, 730.5, 1095.75])  # 1, 2, 3 years
        params = _make_params(dm=15.0, dm1=0.1, dmepoch_int=59000.0, dmepoch_frac=0.0)
        toa_data = _make_toa_data(
            n_toas=3, tdb_int=59000.0 + dt_days, tdb_frac=0.0, freq=freq
        )
        delay = jnp.zeros(3)

        def loss(p):
            return disp(toa_data, p, delay).sum()

        grads = jax.grad(loss)(params)
        dm1_idx = params.param_index("DM1")
        dt_yr = dt_days / 365.25
        expected_grad = jnp.sum(dt_yr * _DMCONST / freq ** 2)
        assert jnp.isclose(grads.values[dm1_idx], expected_grad, rtol=1e-8)

    def test_grad_finite(self):
        """All gradients are finite."""
        disp = DispersionDM(dm_param_names=("DM", "DM1"))
        params = _make_params(dm=15.0, dm1=0.1)
        toa_data = _make_toa_data()
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

        PINT's ``constant_dispersion_delay`` uses the barycentric
        (Doppler-corrected) frequency, whereas JaxPINT currently uses
        the topocentric frequency stored in ``TOAData.freq``.  To get
        an apples-to-apples comparison we compute PINT's delay using the
        topocentric frequency directly via ``dispersion_time_delay``.
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

        # Compute PINT's DM value and delay using topocentric frequency
        dm_comp = model.components["DispersionDM"]
        dm_values = dm_comp.base_dm(toas)
        topo_freq = toas.table["freq"].quantity
        pint_delay = np.array(
            dm_comp.dispersion_time_delay(dm_values, topo_freq).to("s").value,
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
