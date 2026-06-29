"""Tests for the piecewise solar wind dispersion delay component (SWX)."""

from io import StringIO

import jax
import jax.numpy as jnp
import numpy as np
import pytest
pytest.importorskip("pint")  # optional dependency; skip module if absent
from pint.models import get_model
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import pint_toas_to_jax, pint_model_to_params, build_timing_model
from jaxpint.delay.solar_wind_x import SolarWindDispersionX


# ---------------------------------------------------------------------------
# Par file templates
# ---------------------------------------------------------------------------

_PAR_HEADER = """\
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

_PAR_SWX_SINGLE = _PAR_HEADER + """\
SWXDM_0001    0.01
SWXP_0001     2
SWXR1_0001    54000
SWXR2_0001    56000
"""

_PAR_SWX_MULTI = _PAR_HEADER + """\
SWXDM_0001    0.01
SWXP_0001     2
SWXR1_0001    54000
SWXR2_0001    54800
SWXDM_0002    0.005
SWXP_0002     2.5
SWXR1_0002    54800
SWXR2_0002    55600
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_setup(par_str):
    """Build PINT model + JaxPINT data + PINT reference delay."""
    model = get_model(StringIO(par_str))
    toas = make_fake_toas_uniform(
        startMJD=54200, endMJD=55400,
        ntoas=60, model=model, freq=1400.0,
        add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()

    swx_comp = model.components["SolarWindDispersionX"]
    pint_delay = np.array(
        swx_comp.swx_delay(toas).to("s").value,
        dtype=np.float64,
    )

    toa_data = pint_toas_to_jax(toas, model)
    params = pint_model_to_params(model).params
    jax_model, _ = build_timing_model(model)

    jax_swx = [
        c for c in jax_model.delay_components
        if isinstance(c, SolarWindDispersionX)
    ]
    assert len(jax_swx) == 1

    return toa_data, params, pint_delay, model, jax_swx[0]


def _toas_outside(model, startMJD, endMJD, n=10):
    """JaxPINT TOAData spanning [startMJD, endMJD] (to probe zero-delay
    behaviour for TOAs that fall outside every SWX bin)."""
    toas = make_fake_toas_uniform(
        startMJD=startMJD, endMJD=endMJD, ntoas=n, model=model,
        freq=1400.0, add_noise=False,
    )
    toas.compute_TDBs()
    toas.compute_posvels()
    return pint_toas_to_jax(toas, model)


@pytest.fixture
def single_setup():
    """PINT model with a single SWX segment (p=2)."""
    return _make_setup(_PAR_SWX_SINGLE)


@pytest.fixture
def multi_setup():
    """PINT model with two SWX segments (p=2 and p=2.5)."""
    return _make_setup(_PAR_SWX_MULTI)


# ---------------------------------------------------------------------------
# Tests: Single segment matches PINT
# ---------------------------------------------------------------------------


class TestSingleSegment:
    """Single SWX segment delay matches PINT."""

    def test_matches_pint(self, single_setup):
        toa_data, params, pint_delay, _, comp = single_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-10, atol=1e-15,
        )

    def test_nonzero(self, single_setup):
        toa_data, params, _, _, comp = single_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        assert jnp.max(jnp.abs(jax_delay)) > 1e-10


# ---------------------------------------------------------------------------
# Tests: Multiple segments match PINT
# ---------------------------------------------------------------------------


class TestMultipleSegments:
    """Two SWX segments with different power-law indices match PINT.

    PINT uses hypergeometric functions for the geometry integral while
    JaxPINT uses Gauss-Legendre quadrature.  For non-integer p values
    (p=2.5 here) these give slightly different results (~1e-6 relative),
    so we relax the tolerance compared to the single-segment p=2 case.
    """

    def test_matches_pint(self, multi_setup):
        toa_data, params, pint_delay, _, comp = multi_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay, rtol=1e-5, atol=1e-15,
        )

    def test_nonzero(self, multi_setup):
        toa_data, params, _, _, comp = multi_setup
        jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))

        assert jnp.all(jnp.isfinite(jax_delay))
        # At least some TOAs should have non-zero delay (those inside bins)
        assert jnp.max(jnp.abs(jax_delay)) > 1e-10


# ---------------------------------------------------------------------------
# Tests: JIT compatibility
# ---------------------------------------------------------------------------


class TestJIT:
    """SWX component works under jax.jit."""

    def test_jit_single_segment(self, single_setup):
        toa_data, params, _, _, comp = single_setup
        eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jitted = jax.jit(comp)(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-13)

    def test_jit_multi_segment(self, multi_setup):
        toa_data, params, _, _, comp = multi_setup
        eager = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
        jitted = jax.jit(comp)(toa_data, params, jnp.zeros(toa_data.n_toas))

        np.testing.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-13)


# ---------------------------------------------------------------------------
# Tests: Autodiff
# ---------------------------------------------------------------------------


class TestAutodiff:
    """SWX delay is differentiable w.r.t. SWXDM and SWXP."""

    def test_grad_swxdm(self, single_setup):
        toa_data, params, _, _, comp = single_setup

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_vals = jax.grad(total_delay)(params)

        swxdm_idx = params.param_index("SWXDM_0001")
        assert jnp.isfinite(grad_vals.values[swxdm_idx])
        assert grad_vals.values[swxdm_idx] != 0.0

    def test_grad_swxp(self, single_setup):
        toa_data, params, _, _, comp = single_setup

        def total_delay(p):
            return jnp.sum(comp(toa_data, p, jnp.zeros(toa_data.n_toas)))

        grad_vals = jax.grad(total_delay)(params)

        swxp_idx = params.param_index("SWXP_0001")
        assert jnp.isfinite(grad_vals.values[swxp_idx])
        assert grad_vals.values[swxp_idx] != 0.0


# ---------------------------------------------------------------------------
# Tests: Bridge integration
# ---------------------------------------------------------------------------


class TestBridge:
    """The bridge correctly creates a SolarWindDispersionX from a PINT model."""

    def test_bridge_creates_component(self, single_setup):
        _, _, _, _, comp = single_setup
        assert isinstance(comp, SolarWindDispersionX)
        assert comp.n_bins == 1
        assert comp.swxdm_names == ("SWXDM_0001",)
        assert comp.swxp_names == ("SWXP_0001",)
        assert comp.swxr1_names == ("SWXR1_0001",)
        assert comp.swxr2_names == ("SWXR2_0001",)
        assert comp.theta0 > 0  # should be a positive angle

    def test_bridge_multi_segment(self, multi_setup):
        _, _, _, _, comp = multi_setup
        assert comp.n_bins == 2
        assert comp.swxdm_names == ("SWXDM_0001", "SWXDM_0002")
        assert comp.swxp_names == ("SWXP_0001", "SWXP_0002")


# ---------------------------------------------------------------------------
# Tests: Validation
# ---------------------------------------------------------------------------


class TestValidation:
    """Component initialization validates arguments."""

    def test_requires_at_least_one_segment(self):
        with pytest.raises(ValueError, match="at least one segment"):
            SolarWindDispersionX(
                n_bins=0,
                swxdm_names=(),
                swxp_names=(),
                swxr1_names=(),
                swxr2_names=(),
                theta0=0.1,
            )

    def test_mismatched_names(self):
        with pytest.raises(ValueError, match="does not match n_bins"):
            SolarWindDispersionX(
                n_bins=2,
                swxdm_names=("SWXDM_0001",),
                swxp_names=("SWXP_0001", "SWXP_0002"),
                swxr1_names=("SWXR1_0001", "SWXR1_0002"),
                swxr2_names=("SWXR2_0001", "SWXR2_0002"),
                theta0=0.1,
            )


# ---------------------------------------------------------------------------
# Tests: TOAs outside all bins
# ---------------------------------------------------------------------------


class TestTOAsOutsideBins:
    """TOAs outside all SWX bins get zero delay."""

    def test_zero_outside_bins(self, single_setup):
        """TOAs after the single [54000, 56000] segment get exactly zero delay."""
        _, params, _, model, comp = single_setup

        out_data = _toas_outside(model, 56100, 56500)
        out_mjd = np.array(out_data.mjd_int + out_data.mjd_frac)
        assert np.all(out_mjd > 56000), "probe TOAs must be outside the segment"

        jax_delay = np.array(comp(out_data, params, jnp.zeros(out_data.n_toas)))
        np.testing.assert_array_equal(jax_delay, 0.0)

    def test_zero_outside_all_segments(self, multi_setup):
        """In-bin TOAs are nonzero; TOAs past the last segment are exactly zero.

        The two segments are contiguous ([54000, 54800] and [54800, 55600]),
        so there is no inter-bin gap; the in-range TOAs all fall in a bin.  We
        separately generate probe TOAs after 55600 to exercise the
        zero-outside-all-bins path.
        """
        toa_data, params, _, model, comp = multi_setup

        in_delay = np.array(comp(toa_data, params, jnp.zeros(toa_data.n_toas)))
        toa_mjd = np.array(toa_data.mjd_int + toa_data.mjd_frac)
        in_any_bin = (toa_mjd >= 54000) & (toa_mjd <= 55600)
        assert np.all(in_delay[in_any_bin] != 0.0)

        out_data = _toas_outside(model, 55700, 56100)
        out_delay = np.array(comp(out_data, params, jnp.zeros(out_data.n_toas)))
        np.testing.assert_array_equal(out_delay, 0.0)
