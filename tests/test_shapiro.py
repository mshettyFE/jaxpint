"""Tests for the solar system Shapiro delay component."""

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params
from jaxpint.delay.shapiro import SolarSystemShapiroDelay, _ss_obj_shapiro_delay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PAR_BASE = """\
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
CORRECT_TROPOSPHERE  N
"""


@pytest.fixture
def sun_only_setup():
    """PINT model with solar Shapiro delay only (PLANET_SHAPIRO N)."""
    par = _PAR_BASE + "PLANET_SHAPIRO  N\n"
    model = get_model(StringIO(par))
    toas = make_fake_toas_uniform(
        startMJD=54500, endMJD=55500,
        ntoas=50, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()

    shapiro_comp = model.components["SolarSystemShapiro"]
    pint_delay = np.array(
        shapiro_comp.solar_system_shapiro_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model).params

    return toa_data, params, pint_delay


@pytest.fixture
def planet_setup():
    """PINT model with planetary Shapiro delay (PLANET_SHAPIRO Y)."""
    par = _PAR_BASE + "PLANET_SHAPIRO  Y\n"
    model = get_model(StringIO(par))
    toas = make_fake_toas_uniform(
        startMJD=54500, endMJD=55500,
        ntoas=50, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels(planets=True)

    shapiro_comp = model.components["SolarSystemShapiro"]
    pint_delay = np.array(
        shapiro_comp.solar_system_shapiro_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model).params

    return toa_data, params, pint_delay


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSunOnly:
    """Solar Shapiro delay (Sun only) matches PINT."""

    @pytest.mark.slow
    def test_matches_pint(self, sun_only_setup):
        toa_data, params, pint_delay = sun_only_setup

        comp = SolarSystemShapiroDelay(planet_shapiro=False)
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    @pytest.mark.slow
    def test_nonzero(self, sun_only_setup):
        """Shapiro delay should be non-trivially nonzero."""
        toa_data, params, _ = sun_only_setup

        comp = SolarSystemShapiroDelay(planet_shapiro=False)
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-10


class TestWithPlanets:
    """Solar Shapiro delay (Sun + planets) matches PINT."""

    @pytest.mark.slow
    def test_matches_pint(self, planet_setup):
        toa_data, params, pint_delay = planet_setup

        comp = SolarSystemShapiroDelay(planet_shapiro=True)
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    @pytest.mark.slow
    def test_planets_add_contribution(self, sun_only_setup, planet_setup):
        """Planet contribution should differ from Sun-only."""
        toa_data_sun, params_sun, _ = sun_only_setup
        toa_data_pl, params_pl, _ = planet_setup

        sun_comp = SolarSystemShapiroDelay(planet_shapiro=False)
        pl_comp = SolarSystemShapiroDelay(planet_shapiro=True)

        sun_delay = sun_comp(toa_data_sun, params_sun, jnp.zeros(toa_data_sun.n_toas))
        pl_delay = pl_comp(toa_data_pl, params_pl, jnp.zeros(toa_data_pl.n_toas))

        # The two should not be identical (planets add a correction).
        assert not jnp.allclose(sun_delay, pl_delay)


class TestAutodiff:
    """Shapiro delay is differentiable w.r.t. sky position."""

    @pytest.mark.slow
    def test_grad_raj_finite(self, sun_only_setup):
        toa_data, params, _ = sun_only_setup
        comp = SolarSystemShapiroDelay(planet_shapiro=False)

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_fn = jax.grad(total_delay)
        grad_vals = grad_fn(params)

        raj_idx = params.param_index("RAJ")
        assert jnp.isfinite(grad_vals.values[raj_idx])
        assert grad_vals.values[raj_idx] != 0.0

    @pytest.mark.slow
    def test_grad_decj_finite(self, sun_only_setup):
        toa_data, params, _ = sun_only_setup
        comp = SolarSystemShapiroDelay(planet_shapiro=False)

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_fn = jax.grad(total_delay)
        grad_vals = grad_fn(params)

        decj_idx = params.param_index("DECJ")
        assert jnp.isfinite(grad_vals.values[decj_idx])
        assert grad_vals.values[decj_idx] != 0.0


class TestHelper:
    """Unit tests for _ss_obj_shapiro_delay."""

    def test_known_value(self):
        """Check against a hand-computed value."""
        # Sun at 1 AU along x-axis, pulsar along x-axis.
        au_km = 149597870.7
        obj_pos = jnp.array([[au_km, 0.0, 0.0]])
        psr_dir = jnp.array([[1.0, 0.0, 0.0]])
        T_sun = 4.92549094830932e-6

        delay = _ss_obj_shapiro_delay(obj_pos, psr_dir, T_sun)

        # r = AU, rcostheta = AU, arg = (AU - AU) / AU = 0 -> clamped to 1e-100
        # delay = -2 * T_sun * log(1e-100) ≈ -2 * 4.925e-6 * (-230.26) ≈ 2.268e-3
        assert jnp.isfinite(delay[0])
        assert delay[0] > 0  # positive delay when pulsar is behind the Sun
        assert jnp.isclose(delay[0], 2.2682724e-3, rtol=1e-6)

    def test_perpendicular_direction(self):
        """Pulsar perpendicular to Sun direction: moderate delay."""
        au_km = 149597870.7
        obj_pos = jnp.array([[au_km, 0.0, 0.0]])
        psr_dir = jnp.array([[0.0, 1.0, 0.0]])
        T_sun = 4.92549094830932e-6

        delay = _ss_obj_shapiro_delay(obj_pos, psr_dir, T_sun)

        # r = AU, rcostheta = 0, arg = AU / AU = 1.0, log(1) = 0
        expected = -2.0 * T_sun * jnp.log(1.0)
        np.testing.assert_allclose(float(delay[0]), float(expected), atol=1e-20)

    def test_zero_position_guarded(self):
        """Zero position (TZR TOA) should not produce NaN."""
        obj_pos = jnp.array([[0.0, 0.0, 0.0]])
        psr_dir = jnp.array([[1.0, 0.0, 0.0]])

        delay = _ss_obj_shapiro_delay(obj_pos, psr_dir, 4.92549094830932e-6)

        assert jnp.isfinite(delay[0])
