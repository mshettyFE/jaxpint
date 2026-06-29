"""Tests for jaxpint.simulation (zero_residuals and apply_delay_to_toas)."""

from __future__ import annotations

import io

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pint")  # optional dependency; skip module if absent
import pint.models as models
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
)
from jaxpint.fitters import compute_time_residuals
from jaxpint.simulation import apply_delay_to_toas, zero_residuals
from tests.helpers import make_toa_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SYNTH_PAR = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
START         53000 1
FINISH        55000 1
PEPOCH        54000
F0            100.0 1
F1            -1e-15 1
DM            15.0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
"""


@pytest.fixture(scope="module")
def pint_model():
    return models.get_model(io.StringIO(_SYNTH_PAR))


@pytest.fixture(scope="module")
def pint_toas_zeroed(pint_model):
    """Create fake TOAs with PINT (no noise, residuals zeroed by PINT)."""
    toas_lo = make_fake_toas_uniform(
        53000, 55000, 30, pint_model,
        error=10 * u.us, add_noise=False, freq=1400 * u.MHz,
    )
    toas_hi = make_fake_toas_uniform(
        53000, 55000, 30, pint_model,
        error=10 * u.us, add_noise=False, freq=2000 * u.MHz,
    )
    toas_lo.merge(toas_hi)
    return toas_lo


@pytest.fixture(scope="module")
def pint_toas_raw(pint_model):
    """Create raw TOAs (NOT zeroed) for testing zero_residuals from scratch."""
    import pint.toa

    times = np.linspace(53000, 55000, 30, dtype=np.longdouble) * u.d
    ts = pint.toa.get_TOAs_array(
        times, obs="GBT", freqs=1400 * u.MHz, errors=10 * u.us,
        ephem="DE421", include_bipm=True, bipm_version="BIPM2019",
    )
    return ts


@pytest.fixture(scope="module")
def jax_objects_zeroed(pint_model, pint_toas_zeroed):
    """Convert PINT-zeroed TOAs to JaxPINT objects."""
    toa_data = pint_toas_to_jax(pint_toas_zeroed, model=pint_model)
    params = pint_model_to_params(pint_model).params
    jax_model, _noise = build_timing_model(pint_model)
    return jax_model, toa_data, params


@pytest.fixture(scope="module")
def jax_objects_raw(pint_model, pint_toas_raw):
    """Convert raw (unzeroed) PINT TOAs to JaxPINT objects."""
    toa_data = pint_toas_to_jax(pint_toas_raw, model=pint_model)
    params = pint_model_to_params(pint_model).params
    jax_model, _noise = build_timing_model(pint_model)
    return jax_model, toa_data, params


# ---------------------------------------------------------------------------
# apply_delay_to_toas
# ---------------------------------------------------------------------------


class TestApplyDelayToToas:
    """Tests for apply_delay_to_toas."""

    def test_zero_delay_is_identity(self):
        td = make_toa_data(n_toas=5)
        delays = jnp.zeros(5)
        result = apply_delay_to_toas(td, delays)

        np.testing.assert_array_equal(result.mjd_int, td.mjd_int)
        np.testing.assert_array_equal(result.mjd_frac, td.mjd_frac)
        np.testing.assert_array_equal(result.tdb_int, td.tdb_int)
        np.testing.assert_array_equal(result.tdb_frac, td.tdb_frac)

    def test_small_delay_preserves_precision(self):
        td = make_toa_data(n_toas=3, tdb_frac=0.5)
        # 1 microsecond delay
        delays = jnp.array([1e-6, 1e-6, 1e-6])
        result = apply_delay_to_toas(td, delays)

        expected_frac = 0.5 + 1e-6 / 86400.0
        np.testing.assert_allclose(
            result.tdb_frac, expected_frac, atol=1e-18,
        )
        np.testing.assert_array_equal(result.tdb_int, td.tdb_int)

    def test_negative_delay(self):
        td = make_toa_data(n_toas=2, tdb_frac=0.5)
        delays = jnp.array([-1e-6, -1e-6])
        result = apply_delay_to_toas(td, delays)

        expected_frac = 0.5 - 1e-6 / 86400.0
        np.testing.assert_allclose(result.tdb_frac, expected_frac, atol=1e-18)

    def test_overflow_renormalization(self):
        """A delay that pushes frac >= 1 should carry into int."""
        td = make_toa_data(n_toas=1, tdb_frac=0.9999)
        # 100 seconds pushes frac past 1.0
        delays = jnp.array([100.0])
        result = apply_delay_to_toas(td, delays)

        total_before = float(td.tdb_int[0] + td.tdb_frac[0])
        total_after = float(result.tdb_int[0] + result.tdb_frac[0])
        expected_total = total_before + 100.0 / 86400.0

        np.testing.assert_allclose(total_after, expected_total, rtol=1e-12)
        assert float(result.tdb_frac[0]) >= 0.0
        assert float(result.tdb_frac[0]) < 1.0

    def test_underflow_renormalization(self):
        """A large negative delay that pushes frac < 0 should borrow from int."""
        td = make_toa_data(n_toas=1, tdb_frac=0.0001)
        delays = jnp.array([-100.0])
        result = apply_delay_to_toas(td, delays)

        total_before = float(td.tdb_int[0] + td.tdb_frac[0])
        total_after = float(result.tdb_int[0] + result.tdb_frac[0])
        expected_total = total_before - 100.0 / 86400.0

        np.testing.assert_allclose(total_after, expected_total, rtol=1e-12)
        assert float(result.tdb_frac[0]) >= 0.0
        assert float(result.tdb_frac[0]) < 1.0

    def test_mjd_and_tdb_both_updated(self):
        td = make_toa_data(n_toas=3)
        delays = jnp.array([1e-3, -1e-3, 2e-3])
        result = apply_delay_to_toas(td, delays)

        # Both mjd and tdb should change by the same amount
        mjd_delta = (result.mjd_int - td.mjd_int) + (result.mjd_frac - td.mjd_frac)
        tdb_delta = (result.tdb_int - td.tdb_int) + (result.tdb_frac - td.tdb_frac)
        np.testing.assert_allclose(mjd_delta, tdb_delta, atol=1e-18)

    def test_other_fields_unchanged(self):
        td = make_toa_data(n_toas=3)
        delays = jnp.array([1e-3, -1e-3, 2e-3])
        result = apply_delay_to_toas(td, delays)

        np.testing.assert_array_equal(result.error, td.error)
        np.testing.assert_array_equal(result.freq, td.freq)
        np.testing.assert_array_equal(result.ssb_obs_pos, td.ssb_obs_pos)
        np.testing.assert_array_equal(result.delta_pulse_number, td.delta_pulse_number)


# ---------------------------------------------------------------------------
# zero_residuals
# ---------------------------------------------------------------------------


class TestZeroResiduals:
    """Tests for zero_residuals."""

    @pytest.mark.slow
    def test_converges_from_raw(self, jax_objects_raw):
        """Starting from unzeroed TOAs, residuals should converge to < 1 ns."""
        model, toa_data, params = jax_objects_raw
        tol = 1e-9  # 1 nanosecond

        # Raw TOAs have large residuals (milliseconds)
        raw_resids = compute_time_residuals(model, toa_data, params)
        assert float(jnp.max(jnp.abs(raw_resids))) > 1e-4  # > 0.1 ms

        result = zero_residuals(model, toa_data, params, tolerance=tol)
        resids = compute_time_residuals(model, result, params)
        max_resid = float(jnp.max(jnp.abs(resids)))

        assert max_resid < tol, f"max |residual| = {max_resid:.3e} s > {tol:.3e} s"

    @pytest.mark.slow
    def test_already_zeroed_is_noop(self, jax_objects_zeroed):
        """If residuals are already below tolerance, TOAs should not change."""
        model, toa_data, params = jax_objects_zeroed
        tol = 1e-9

        zeroed = zero_residuals(model, toa_data, params, tolerance=tol)
        zeroed_again = zero_residuals(model, zeroed, params, tolerance=tol)

        np.testing.assert_allclose(
            zeroed.tdb_frac, zeroed_again.tdb_frac, atol=1e-15,
        )
        np.testing.assert_array_equal(zeroed.tdb_int, zeroed_again.tdb_int)

    @pytest.mark.slow
    def test_vs_pint(self, jax_objects_raw):
        """JaxPINT zero_residuals should produce sub-nanosecond residuals."""
        model, toa_data, params = jax_objects_raw
        tol = 1e-9

        jax_zeroed = zero_residuals(model, toa_data, params, tolerance=tol)
        jax_resids = compute_time_residuals(model, jax_zeroed, params)

        assert float(jnp.max(jnp.abs(jax_resids))) < tol

    @pytest.mark.slow
    def test_raises_on_nonconvergence(self, jax_objects_raw):
        """Should raise RuntimeError if maxiter is too small."""
        model, toa_data, params = jax_objects_raw

        with pytest.raises(RuntimeError, match="did not converge"):
            zero_residuals(
                model, toa_data, params,
                maxiter=0, tolerance=1e-20,
            )
