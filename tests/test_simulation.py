"""Tests for jaxpint.simulation (zero_residuals and apply_delay_to_toas).
"""

from __future__ import annotations

import io

import jax.numpy as jnp
import numpy as np
import pytest

import jaxpint.par as jpar
from jaxpint import build_model
from jaxpint.fitters import compute_time_residuals
from jaxpint.simulation import (
    apply_delay_to_toas,
    make_fake_toas_uniform,
    make_uniform_toa_data,
    zero_residuals,
)
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
def par_result():
    return jpar.get_model(io.StringIO(_SYNTH_PAR))


@pytest.fixture(scope="module")
def jax_objects_zeroed(par_result):
    """Model + TOAs whose residuals were already zeroed by the generator.

    Two interleaved frequencies (the old version merged separate 1400 and
    2000 MHz sets; cycling covers the same multi-frequency ground).
    """
    toa_data = make_fake_toas_uniform(
        53000.0, 55000.0, 60, par_result,
        obs="gbt", freq_mhz=[1400.0, 2000.0], error_us=10.0,
    )
    model, _noise = build_model(par_result, toa_data)
    return model, toa_data, par_result.params


@pytest.fixture(scope="module")
def jax_objects_raw(par_result):
    """Model + raw grid TOAs (NOT zeroed), for testing zero_residuals itself.

    make_uniform_toa_data is the scaffolding-only builder: real GBT clock
    corrections and posvels, but timestamps that do not realize any model --
    so the fixture's residuals start out large, which test_converges_from_raw
    asserts before zeroing.
    """
    toa_data = make_uniform_toa_data(
        53000.0, 55000.0, 30, par_result, obs="gbt", freq_mhz=1400.0, error_us=10.0
    )
    model, _noise = build_model(par_result, toa_data)
    return model, toa_data, par_result.params


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
    def test_zeroed_to_sub_ns(self, jax_objects_raw):
        """zero_residuals drives a raw grid below 1 ns.

        (Previously named test_vs_pint, but no PINT quantity was ever
        compared -- the assertion has always been JaxPINT-side only.)
        """
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
