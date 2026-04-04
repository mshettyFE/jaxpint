"""Tests for the solar wind dispersion delay component."""

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.solar_wind import (
    SolarWindDispersion,
    _solar_wind_geometry_swm0,
    _solar_wind_geometry_swm1,
    _sun_angle_and_distance,
)


# ---------------------------------------------------------------------------
# Par file templates
# ---------------------------------------------------------------------------

_PAR_SWM0 = """\
PSR           J1744-1134
RAJ           17:44:29.407
DECJ          -11:34:54.681
F0            245.4261197
F1            -5.381e-16
PEPOCH        55000
DM            3.138
NE_SW         5.0
SWM           0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  N
"""

_PAR_SWM1 = """\
PSR           J1744-1134
RAJ           17:44:29.407
DECJ          -11:34:54.681
F0            245.4261197
F1            -5.381e-16
PEPOCH        55000
DM            3.138
NE_SW         5.0
SWM           1
SWP           2.0
EPHEM         DE421
CLK           UTC(NIST)
UNITS         TDB
CORRECT_TROPOSPHERE  N
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_setup(par_str):
    """Build PINT model + JaxPINT data from a par string."""
    model = get_model(StringIO(par_str))
    toas = make_fake_toas_uniform(
        startMJD=54500, endMJD=55500,
        ntoas=50, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()

    sw_comp = model.components["SolarWindDispersion"]
    pint_delay = np.array(
        sw_comp.solar_wind_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model)
    jax_model, _ = build_timing_model(model)

    # Extract the JaxPINT solar wind component
    jax_sw = [
        c for c in jax_model.delay_components
        if isinstance(c, SolarWindDispersion)
    ]
    assert len(jax_sw) == 1

    return toa_data, params, pint_delay, model, jax_sw[0]


@pytest.fixture
def swm0_setup():
    """PINT model with SWM=0 solar wind."""
    return _make_setup(_PAR_SWM0)


@pytest.fixture
def swm1_setup():
    """PINT model with SWM=1 solar wind."""
    return _make_setup(_PAR_SWM1)


# ---------------------------------------------------------------------------
# Tests: SWM=0
# ---------------------------------------------------------------------------


class TestSWM0:
    """Solar wind delay with SWM=0 (Edwards et al.) matches PINT."""

    def test_matches_pint(self, swm0_setup):
        toa_data, params, pint_delay, _, comp = swm0_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_nonzero(self, swm0_setup):
        toa_data, params, _, _, comp = swm0_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-10

    def test_jit_compatible(self, swm0_setup):
        toa_data, params, _, _, comp = swm0_setup
        eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jitted = jax.jit(comp)(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)


# ---------------------------------------------------------------------------
# Tests: SWM=1
# ---------------------------------------------------------------------------


class TestSWM1:
    """Solar wind delay with SWM=1 (Hazboun et al.) matches PINT."""

    def test_matches_pint(self, swm1_setup):
        toa_data, params, pint_delay, _, comp = swm1_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_nonzero(self, swm1_setup):
        toa_data, params, _, _, comp = swm1_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-10

    def test_jit_compatible(self, swm1_setup):
        toa_data, params, _, _, comp = swm1_setup
        eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jitted = jax.jit(comp)(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-14)


# ---------------------------------------------------------------------------
# Tests: SWM=0 and SWM=1 agree for p=2
# ---------------------------------------------------------------------------


class TestSWM0vsSWM1:
    """SWM=0 and SWM=1 with p=2 should produce similar (but not identical) results.

    They use different integration approaches, but for p=2 the physics
    is the same.  The geometry factors should agree to high precision.
    """

    def test_geometry_agreement(self):
        """Geometry factors at typical elongation angles should agree."""
        theta = jnp.array([0.5, 1.0, 1.5, 2.0, 2.5])
        r_km = jnp.full_like(theta, 149597870.7)  # 1 AU
        p = jnp.float64(2.0)

        g0 = _solar_wind_geometry_swm0(theta, r_km)
        g1 = _solar_wind_geometry_swm1(theta, r_km, p)

        np.testing.assert_allclose(np.array(g0), np.array(g1), rtol=1e-6)


# ---------------------------------------------------------------------------
# Tests: Autodiff
# ---------------------------------------------------------------------------


class TestAutodiff:
    """Solar wind delay is differentiable w.r.t. NE_SW and SWP."""

    def test_grad_ne_sw_swm0(self, swm0_setup):
        toa_data, params, _, _, comp = swm0_setup

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_vals = jax.grad(total_delay)(params)

        ne_sw_idx = params.param_index("NE_SW")
        assert jnp.isfinite(grad_vals.values[ne_sw_idx])
        assert grad_vals.values[ne_sw_idx] != 0.0

    def test_grad_ne_sw_swm1(self, swm1_setup):
        toa_data, params, _, _, comp = swm1_setup

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_vals = jax.grad(total_delay)(params)

        ne_sw_idx = params.param_index("NE_SW")
        assert jnp.isfinite(grad_vals.values[ne_sw_idx])
        assert grad_vals.values[ne_sw_idx] != 0.0

    def test_grad_swp(self, swm1_setup):
        toa_data, params, _, _, comp = swm1_setup

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_vals = jax.grad(total_delay)(params)

        swp_idx = params.param_index("SWP")
        assert jnp.isfinite(grad_vals.values[swp_idx])


# ---------------------------------------------------------------------------
# Tests: Bridge integration
# ---------------------------------------------------------------------------


class TestBridge:
    """The bridge correctly creates a SolarWindDispersion from a PINT model."""

    def test_bridge_swm0(self, swm0_setup):
        _, _, _, _, comp = swm0_setup
        assert comp.swm == 0
        assert comp.swp_name is None

    def test_bridge_swm1(self, swm1_setup):
        _, _, _, _, comp = swm1_setup
        assert comp.swm == 1
        assert comp.swp_name == "SWP"


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Component initialization validates arguments."""

    def test_requires_ne_sw(self):
        with pytest.raises(ValueError, match="at least one NE_SW term"):
            SolarWindDispersion(ne_sw_param_names=())

    def test_first_must_be_ne_sw(self):
        with pytest.raises(ValueError, match="First NE_SW term"):
            SolarWindDispersion(ne_sw_param_names=("NE_SW1",))

    def test_invalid_swm(self):
        with pytest.raises(ValueError, match="SWM must be 0 or 1"):
            SolarWindDispersion(ne_sw_param_names=("NE_SW",), swm=2)

    def test_swm1_requires_swp(self):
        with pytest.raises(ValueError, match="swp_name"):
            SolarWindDispersion(ne_sw_param_names=("NE_SW",), swm=1)
