"""Tests for the troposphere delay component."""

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytest.importorskip("pint")  # optional dependency; skip module if absent
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params
from jaxpint.constants import NIELL_LAT_BREAKS
from tests.helpers import make_toa_data, make_params
from jaxpint.delay.troposphere import (
    TroposphereDelay,
    _herring_map,
    _herring_map_scalar,
    _interp_lat,
    _pressure_from_height_km,
    _year_fraction,
    _zenith_hydrostatic_delay,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PAR = """\
PSR           J1744-1134
RAJ           17:44:29.407
DECJ          -11:34:54.681
F0            245.4261197
F1            -5.381e-16
PEPOCH        55000
DM            3.138
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  Y
"""


@pytest.fixture
def tropo_setup():
    """PINT model with troposphere correction enabled."""
    model = get_model(StringIO(_PAR))
    toas = make_fake_toas_uniform(
        startMJD=54500, endMJD=55500,
        ntoas=50, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()

    tropo_comp = model.components["TroposphereDelay"]
    pint_delay = np.array(
        tropo_comp.troposphere_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model).params

    return toa_data, params, pint_delay


# ---------------------------------------------------------------------------
# Oracle tests: match PINT
# ---------------------------------------------------------------------------

class TestMatchesPINT:
    """Troposphere delay matches PINT output."""

    @pytest.mark.slow
    def test_matches_pint(self, tropo_setup):
        toa_data, params, pint_delay = tropo_setup

        comp = TroposphereDelay()
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    @pytest.mark.slow
    def test_nonzero(self, tropo_setup):
        """Troposphere delay should be non-trivially nonzero."""
        toa_data, params, _ = tropo_setup

        comp = TroposphereDelay()
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-12

    @pytest.mark.slow
    def test_correct_troposphere_n(self):
        """When CORRECT_TROPOSPHERE N, no troposphere data should be populated."""
        par = _PAR.replace("CORRECT_TROPOSPHERE  Y", "CORRECT_TROPOSPHERE  N")
        model = get_model(StringIO(par))
        toas = make_fake_toas_uniform(
            startMJD=54500, endMJD=55500,
            ntoas=10, model=model, freq=1400.0,
            add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        toa_data = pint_toas_to_jax(toas, model)
        assert toa_data.tropo_alt is None

        comp = TroposphereDelay()
        jax_delay = comp(toa_data, pint_model_to_params(model).params, jnp.zeros(toa_data.n_toas))
        np.testing.assert_array_equal(np.array(jax_delay), 0.0)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Unit tests for pure helper functions."""

    def test_herring_map_zenith(self):
        """At zenith (sin_alt=1), mapping should be 1.0."""
        sin_alt = jnp.ones(1)
        a = jnp.array([1.2769934e-3])
        b = jnp.array([2.9153695e-3])
        c = jnp.array([62.610505e-3])

        result = _herring_map(sin_alt, a, b, c)
        np.testing.assert_allclose(float(result[0]), 1.0, atol=1e-14)

    def test_herring_map_scalar_zenith(self):
        """Scalar version at zenith should also be 1.0."""
        sin_alt = jnp.ones(1)
        result = _herring_map_scalar(sin_alt, 2.53e-5, 5.49e-3, 1.14e-3)
        np.testing.assert_allclose(float(result[0]), 1.0, atol=1e-14)

    def test_herring_map_low_elevation(self):
        """At low elevation, mapping should be large (>> 1)."""
        sin_alt = jnp.array([jnp.sin(jnp.radians(5.0))])
        a = jnp.array([1.2769934e-3])
        b = jnp.array([2.9153695e-3])
        c = jnp.array([62.610505e-3])

        result = _herring_map(sin_alt, a, b, c)
        assert float(result[0]) > 5.0

    def test_pressure_sea_level(self):
        """Pressure at sea level should be ~101.325 kPa."""
        p = _pressure_from_height_km(jnp.array([0.0]))
        np.testing.assert_allclose(float(p[0]), 101.325, rtol=1e-6)

    def test_pressure_decreases_with_height(self):
        """Pressure should decrease with increasing height."""
        heights = jnp.array([0.0, 1.0, 2.0, 5.0])
        p = _pressure_from_height_km(heights)
        for i in range(1, len(p)):
            assert p[i] < p[i - 1]

    def test_zenith_delay_order_of_magnitude(self):
        """Zenith delay at sea level should be ~7-8 ns."""
        lat = jnp.array([jnp.radians(45.0)])
        H_km = jnp.array([0.0])
        delay = _zenith_hydrostatic_delay(lat, H_km)
        # ~101.325 / (43.921 * c) ~ 7.7e-9 seconds
        assert 5e-9 < float(delay[0]) < 1e-8

    def test_year_fraction_range(self):
        """Year fraction should be in [0, 1)."""
        mjd = jnp.array([55000.0, 55182.0, 55365.0])
        lat = jnp.array([0.5, -0.5, 0.5])
        yf = _year_fraction(mjd, lat)
        assert jnp.all(yf >= 0.0)
        assert jnp.all(yf < 1.0)

    def test_year_fraction_southern_offset(self):
        """Southern hemisphere should have 0.5 offset."""
        mjd = jnp.array([55000.0])
        yf_north = _year_fraction(mjd, jnp.array([0.5]))
        yf_south = _year_fraction(mjd, jnp.array([-0.5]))
        diff = float(jnp.abs(yf_north[0] - yf_south[0]))
        np.testing.assert_allclose(diff, 0.5, atol=0.01)

    def test_interp_lat_at_breakpoints(self):
        """Interpolation at exact breakpoints should return the coefficient value."""
        coeff = jnp.arange(7, dtype=jnp.float64)
        for i in range(1, 6):  # skip endpoints where searchsorted edge cases occur
            lat = NIELL_LAT_BREAKS[i]
            result = _interp_lat(jnp.array([lat]), coeff)
            np.testing.assert_allclose(float(result[0]), float(i), atol=1e-12)

    def test_interp_lat_midpoint(self):
        """Interpolation at midpoint should return average of neighbors."""
        coeff = jnp.arange(7, dtype=jnp.float64)
        mid = (NIELL_LAT_BREAKS[2] + NIELL_LAT_BREAKS[3]) / 2.0
        result = _interp_lat(jnp.array([mid]), coeff)
        np.testing.assert_allclose(float(result[0]), 2.5, atol=1e-10)


# ---------------------------------------------------------------------------
# Monotonicity tests
# ---------------------------------------------------------------------------

class TestMonotonicity:
    """Physical sanity checks: delay should vary monotonically."""

    @pytest.mark.slow
    def test_delay_decreases_with_altitude(self, tropo_setup):
        """Troposphere delay should decrease as target rises higher."""
        toa_data, params, _ = tropo_setup
        comp = TroposphereDelay()
        jax_delay = np.array(comp(toa_data, params, jnp.zeros(toa_data.n_toas)))

        alts = np.array(toa_data.tropo_alt)
        valid = np.array(toa_data.tropo_alt_valid)

        # Single observatory + source: delay strictly decreases as the target
        # rises (air path shortens). Check every valid TOA, not just endpoints.
        sorted_delays = jax_delay[valid][np.argsort(alts[valid])]
        assert np.all(np.diff(sorted_delays) < 0)


# ---------------------------------------------------------------------------
# JIT tests
# ---------------------------------------------------------------------------

class TestJIT:
    """Component works under JIT compilation."""

    @pytest.mark.slow
    def test_jit_compiles(self, tropo_setup):
        toa_data, params, pint_delay = tropo_setup
        comp = TroposphereDelay()

        @jax.jit
        def compute(td, p):
            return comp(td, p, jnp.zeros(td.n_toas))

        jax_delay = compute(toa_data, params)
        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    @pytest.mark.slow
    def test_jit_consistent(self, tropo_setup):
        """JIT and non-JIT produce identical results."""
        toa_data, params, _ = tropo_setup
        comp = TroposphereDelay()

        eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        @jax.jit
        def compute(td, p):
            return comp(td, p, jnp.zeros(td.n_toas))

        jitted = compute(toa_data, params)
        np.testing.assert_array_equal(np.array(eager), np.array(jitted))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case handling."""

    def test_none_tropo_data_returns_zeros(self):
        """When tropo_alt is None, should return zeros."""
        toa_data = make_toa_data(1, tdb_int=55000.0, tdb_frac=0.0,
                                 obs_names=("gbt",), planet_positions=None)
        params = make_params(("DUMMY",), [0.0], frozen_mask=(True,), units=("s",))

        comp = TroposphereDelay()
        result = comp(toa_data, params, jnp.zeros(1))
        np.testing.assert_array_equal(np.array(result), 0.0)

    def test_invalid_altitudes_zeroed(self):
        """TOAs with tropo_alt_valid=False should have zero delay."""
        toa_data = make_toa_data(
            2, tdb_int=55000.0, tdb_frac=0.0,
            obs_names=("gbt",), planet_positions=None,
            tropo_alt=jnp.array([jnp.radians(45.0), jnp.pi / 2]),
            tropo_alt_valid=jnp.array([True, False]),
            obs_geodetic_lat=jnp.array([jnp.radians(40.0), jnp.radians(40.0)]),
            obs_height_km=jnp.array([1.0, 1.0]),
        )
        params = make_params(("DUMMY",), [0.0], frozen_mask=(True,), units=("s",))

        comp = TroposphereDelay()
        result = comp(toa_data, params, jnp.zeros(2))

        # First TOA should have nonzero delay
        assert float(result[0]) != 0.0
        # Second TOA (invalid) should be zero
        assert float(result[1]) == 0.0
