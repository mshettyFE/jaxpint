"""Tests for the solar wind dispersion delay component."""

import functools
from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytest.importorskip("pint")  # optional dependency; skip module if absent
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

import equinox as eqx

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.solar_wind import (
    SolarWindDispersion,
    _solar_wind_geometry_swm0,
    _solar_wind_geometry_swm1,
    _sun_angle_and_distance,
)
from tests.helpers import make_params, make_toa_data


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
    """Build PINT model + JaxPINT data from a par string.
    """
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
    params = pint_model_to_params(model).params
    jax_model, _ = build_timing_model(model)

    # Extract the JaxPINT solar wind component
    jax_sw = [
        c for c in jax_model.delay_components
        if isinstance(c, SolarWindDispersion)
    ]
    assert len(jax_sw) == 1

    return toa_data, params, pint_delay, model, jax_sw[0]


_make_setup = functools.lru_cache(maxsize=None)(_make_setup)


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
        assert grad_vals.values[swp_idx] != 0.0


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
# Tests: model.compute_dm parity (solar-wind DM reaches the total DM)
# ---------------------------------------------------------------------------


class TestComputeDM:
    """Solar-wind DM flows into ``TimingModel.compute_dm`` and matches PINT.

    ``SolarWindDispersion`` is a :class:`DispersionDelayComponent`, so its DM must
    be summed into the model's total DM (used for wideband fitting), matching
    PINT's ``total_dm`` (which includes the solar wind via ``dm_value_funcs``).
    """

    @pytest.mark.parametrize("par_str", [_PAR_SWM0, _PAR_SWM1])
    def test_compute_dm_matches_pint_total_dm(self, par_str):
        model = get_model(StringIO(par_str))
        toas = make_fake_toas_uniform(
            startMJD=54500, endMJD=55500,
            ntoas=50, model=model, freq=1400.0,
            add_noise=False,
        )
        toas.compute_TDBs()
        toas.compute_posvels()

        toa_data = pint_toas_to_jax(toas, model)
        params = pint_model_to_params(model).params
        jax_model, _ = build_timing_model(model)

        jax_dm = np.array(jax_model.compute_dm(toa_data, params))
        pint_dm = np.array(model.total_dm(toas).to("pc/cm^3").value)

        np.testing.assert_allclose(jax_dm, pint_dm, rtol=1e-10, atol=1e-12)

        # The solar wind must actually contribute: total DM is not just the flat
        # constant DM (3.138) but carries a time/geometry-dependent piece.
        assert np.max(np.abs(jax_dm - 3.138)) > 1e-6


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


# ---------------------------------------------------------------------------
# Barycentric (obs_sun_pos == 0) guard
# ---------------------------------------------------------------------------


class TestBarycentricGuard:
    """A barycentric TOA row (obs_sun_pos == 0, e.g. the TZRSITE @ TZR row the
    bridge zeroes deliberately) formerly hit 0/0 = nan in
    ``_sun_angle_and_distance`` and poisoned the whole delay/phase chain
    whenever NE_SW != 0.  astrometry.py and shapiro.py guard the same case;
    solar wind now does too: such rows contribute exactly zero DM."""

    def _setup(self, swm=0):
        # Row 0: barycentric (zeros).  Row 1: a realistic observer-Sun vector
        # (~1 AU along x).  make_toa_data defaults obs_sun_pos to all-zeros;
        # patch in the mixed geometry.
        toa_data = make_toa_data(t_mjd=[55000.3, 55001.3])
        obs_sun = jnp.array([[0.0, 0.0, 0.0], [1.496e8, 0.0, 0.0]])
        toa_data = eqx.tree_at(lambda t: t.obs_sun_pos, toa_data, obs_sun)

        names = ["RAJ", "DECJ", "NE_SW", "PEPOCH"]
        values = [1.0, 0.3, 8.0, 0.0]
        units = ["rad", "rad", "cm^-3", "day"]
        if swm == 1:
            names.append("SWP")
            values.append(2.0)
            units.append("")
        params = make_params(
            names=tuple(names), values=tuple(values), units=tuple(units),
            epoch_int_values={"PEPOCH": 55000.0},
        )
        comp = SolarWindDispersion(
            ne_sw_param_names=("NE_SW",),
            swepoch_name="PEPOCH",
            swm=swm,
            swp_name="SWP" if swm == 1 else None,
        )
        return toa_data, params, comp

    @pytest.mark.parametrize("swm", [0, 1])
    def test_zero_obs_sun_row_gives_zero_dm(self, swm):
        toa_data, params, comp = self._setup(swm=swm)
        dm = comp.compute_dm(toa_data, params, jnp.zeros(2))
        assert jnp.all(jnp.isfinite(dm)), f"non-finite DM: {dm}"
        assert float(dm[0]) == 0.0  # barycentric row: no line of sight
        assert float(dm[1]) != 0.0  # real row unaffected by the guard

    @pytest.mark.parametrize("swm", [0, 1])
    def test_grads_finite_with_zero_obs_sun_row(self, swm):
        toa_data, params, comp = self._setup(swm=swm)

        def loss(p):
            return comp.compute_dm(toa_data, p, jnp.zeros(2)).sum()

        grads = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(grads.values)), (
            f"non-finite gradients: "
            f"{[n for n, g in zip(params.names, grads.values) if not jnp.isfinite(g)]}"
        )

    def test_sun_angle_helper_finite_at_zero(self):
        toa_data, params, comp = self._setup()
        psr_dir = jnp.tile(jnp.array([[0.5, 0.5, 0.7071]]), (2, 1))
        theta, r_km = _sun_angle_and_distance(toa_data, psr_dir)
        assert jnp.all(jnp.isfinite(theta))
        assert float(r_km[0]) == 0.0  # true r is preserved, not the dummy
