"""Tests for the bridge layer (PINT <-> JaxPINT conversion)."""

from __future__ import annotations

import copy

import jax.numpy as jnp
import numpy as np
import pytest

import pint.models as models
import pint.toa as toa
from pint.config import examplefile
from pint.models.parameter import AngleParameter, MJDParameter

from jaxpint.bridge import (
    params_to_pint_model,
    pint_model_to_params,
    pint_toas_to_jax,
)


@pytest.fixture
def ngc6440e():
    """Load the NGC6440E test pulsar (simple, isolated)."""
    model = models.get_model(examplefile("NGC6440E.par"))
    toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
    return model, toas


@pytest.fixture
def b1855():
    """Load B1855+09 (binary, with JUMPs/EFAC/EQUAD)."""
    model = models.get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
    toas = toa.get_TOAs(examplefile("B1855+09_NANOGrav_9yv1.tim"), ephem="DE421")
    return model, toas


# ===================================================================
# pint_toas_to_jax
# ===================================================================


class TestPintToasToJax:
    """Tests for TOA conversion."""

    def test_basic_fields(self, ngc6440e):
        model, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        assert td.n_toas == toas.ntoas
        assert td.mjd_int.shape == (td.n_toas,)
        assert td.mjd_frac.shape == (td.n_toas,)
        assert td.tdb_int.shape == (td.n_toas,)
        assert td.tdb_frac.shape == (td.n_toas,)
        assert td.error.shape == (td.n_toas,)
        assert td.freq.shape == (td.n_toas,)
        assert td.ssb_obs_pos.shape == (td.n_toas, 3)
        assert td.ssb_obs_vel.shape == (td.n_toas, 3)
        assert td.obs_sun_pos.shape == (td.n_toas, 3)
        assert td.obs_indices.shape == (td.n_toas,)

    def test_dtypes(self, ngc6440e):
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        assert td.mjd_int.dtype == jnp.float64
        assert td.tdb_frac.dtype == jnp.float64
        assert td.error.dtype == jnp.float64
        assert td.obs_indices.dtype == jnp.int32

    def test_mjd_int_frac_reconstruction(self, ngc6440e):
        """Verify MJD int+frac reconstructs to within ~1 ns of the original."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        # Reconstruct and compare against PINT's tdbld (longdouble)
        tdb_reconstructed = np.asarray(td.tdb_int) + np.asarray(td.tdb_frac)
        tdb_original = np.asarray(toas.table["tdbld"], dtype=np.float64)

        # float64 MJDs near 53000-54000 have ~10 ps precision,
        # so int+frac should be exact to float64
        np.testing.assert_allclose(
            tdb_reconstructed,
            tdb_original,
            atol=1e-14,  # ~1 ps in days
            rtol=0,
        )

    def test_mjd_frac_in_range(self, ngc6440e):
        """Fractional day should be in [0, 1)."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        assert jnp.all(td.mjd_frac >= 0.0)
        assert jnp.all(td.mjd_frac < 1.0)
        assert jnp.all(td.tdb_frac >= 0.0)
        assert jnp.all(td.tdb_frac < 1.0)

    def test_mjd_int_is_integer(self, ngc6440e):
        """Integer day should actually be an integer."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        np.testing.assert_array_equal(
            np.asarray(td.mjd_int), np.floor(np.asarray(td.mjd_int))
        )
        np.testing.assert_array_equal(
            np.asarray(td.tdb_int), np.floor(np.asarray(td.tdb_int))
        )

    def test_error_positive(self, ngc6440e):
        """TOA errors should be positive (in seconds)."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)
        assert jnp.all(td.error > 0)

    def test_freq_positive(self, ngc6440e):
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)
        assert jnp.all(td.freq > 0)

    def test_freq_is_barycentric(self, ngc6440e):
        """When model is provided, freq should be barycentric (Doppler-corrected)."""
        import astropy.units as u

        model, toas = ngc6440e
        td = pint_toas_to_jax(toas, model)

        expected = np.asarray(
            model.barycentric_radio_freq(toas).to(u.MHz).value,
            dtype=np.float64,
        )
        np.testing.assert_allclose(np.asarray(td.freq), expected, rtol=1e-14)

    def test_observatory_names(self, ngc6440e):
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)

        assert isinstance(td.obs_names, tuple)
        assert len(td.obs_names) > 0
        # Every index should be in range
        assert jnp.all(td.obs_indices >= 0)
        assert jnp.all(td.obs_indices < len(td.obs_names))

    def test_no_model_gives_empty_masks(self, ngc6440e):
        """Without a model, flag_masks should be empty."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas, model=None)
        assert td.flag_masks == {}

    def test_flag_masks_with_model(self, b1855):
        """With a model that has mask params, flag_masks should be populated."""
        model, toas = b1855
        td = pint_toas_to_jax(toas, model=model)

        # B1855+09 has JUMP/EFAC/EQUAD parameters
        assert len(td.flag_masks) > 0
        for name, mask in td.flag_masks.items():
            assert mask.shape == (td.n_toas,)
            assert mask.dtype == jnp.bool_

    def test_flag_masks_match_pint(self, b1855):
        """Verify flag masks match PINT's select_toa_mask for each param."""
        model, toas = b1855
        td = pint_toas_to_jax(toas, model=model)

        from pint.models.parameter import maskParameter

        for pname in model.params:
            param = getattr(model, pname)
            if isinstance(param, maskParameter) and pname in td.flag_masks:
                pint_idx = param.select_toa_mask(toas)
                expected = np.zeros(td.n_toas, dtype=bool)
                if len(pint_idx) > 0:
                    expected[pint_idx] = True
                np.testing.assert_array_equal(
                    np.asarray(td.flag_masks[pname]), expected
                )

    def test_no_planets_by_default(self, ngc6440e):
        """NGC6440E doesn't compute planet positions by default."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)
        # planet_positions is None unless planets were computed
        if td.planet_positions is not None:
            assert len(td.planet_positions) == 0

    def test_no_wideband_dm(self, ngc6440e):
        """NGC6440E is narrowband — no DM values."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas)
        assert td.dm_values is None
        assert td.dm_errors is None

    def test_auto_compute_tdb(self):
        """If TDBs not computed, pint_toas_to_jax should compute them."""
        toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
        # Remove TDB columns to force recomputation
        if "tdbld" in toas.table.colnames:
            toas.table.remove_column("tdbld")
            toas.table.remove_column("tdb")
        td = pint_toas_to_jax(toas)
        assert td.tdb_int.shape == (td.n_toas,)


# ===================================================================
# pint_model_to_params
# ===================================================================


class TestPintModelToParams:
    """Tests for parameter extraction."""

    def test_basic_structure(self, ngc6440e):
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        assert pv.n_params > 0
        assert pv.values.shape == (pv.n_params,)
        assert len(pv.names) == pv.n_params
        assert len(pv.units) == pv.n_params
        assert len(pv.frozen_mask) == pv.n_params
        assert len(pv.units) == pv.n_params

    def test_expected_params_present(self, ngc6440e):
        """NGC6440E should have F0, F1, RAJ, DECJ, DM, PEPOCH, POSEPOCH."""
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        for expected in ("F0", "F1", "RAJ", "DECJ", "DM"):
            assert expected in pv.names, f"{expected} missing from ParameterVector"

    def test_epoch_split(self, ngc6440e):
        """PEPOCH should be split into epoch_int_values + fractional in values."""
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        assert "PEPOCH" in pv.epoch_int_values
        pepoch_int = pv.epoch_int_values["PEPOCH"]
        pepoch_frac = float(pv.param_value("PEPOCH"))

        # Reconstruct and compare to original
        original = float(model.PEPOCH.value)
        reconstructed = pepoch_int + pepoch_frac
        assert abs(reconstructed - original) < 1e-12  # sub-ns

    def test_angle_in_radians(self, ngc6440e):
        """RAJ and DECJ should be stored in radians."""
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        raj_idx = pv.param_index("RAJ")
        decj_idx = pv.param_index("DECJ")

        assert pv.units[raj_idx] == "rad"
        assert pv.units[decj_idx] == "rad"

        # Check the value matches PINT's quantity converted to radians
        import astropy.units as u

        expected_raj_rad = model.RAJ.quantity.to(u.rad).value
        np.testing.assert_allclose(
            float(pv.values[raj_idx]), expected_raj_rad, rtol=1e-15
        )

    def test_frozen_mask(self, ngc6440e):
        """Check frozen status matches PINT model."""
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        for i, name in enumerate(pv.names):
            param = getattr(model, name)
            assert pv.frozen_mask[i] == param.frozen, (
                f"{name}: expected frozen={param.frozen}, got {pv.frozen_mask[i]}"
            )

    def test_no_string_or_bool_params(self, ngc6440e):
        """String and bool parameters should not appear."""
        from pint.models.parameter import boolParameter, intParameter, strParameter

        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        for name in pv.names:
            param = getattr(model, name)
            assert not isinstance(param, (strParameter, boolParameter, intParameter))

    def test_name_to_index_consistent(self, ngc6440e):
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        for i, name in enumerate(pv.names):
            assert pv.param_index(name) == i


# ===================================================================
# params_to_pint_model (round-trip)
# ===================================================================


class TestParamsToPintModel:
    """Tests for writing parameters back to PINT."""

    def test_roundtrip_float_params(self, ngc6440e):
        """Float parameters should survive the round-trip exactly."""
        model, _ = ngc6440e
        original_model = copy.deepcopy(model)
        pv = pint_model_to_params(model)
        params_to_pint_model(pv, model)

        for name in pv.names:
            param = getattr(model, name)
            original = getattr(original_model, name)
            if not isinstance(param, (MJDParameter, AngleParameter)):
                assert param.value == pytest.approx(original.value, rel=1e-15), (
                    f"{name}: {param.value} != {original.value}"
                )

    def test_roundtrip_angle_params(self, ngc6440e):
        """Angle params should survive radians -> native -> radians."""
        model, _ = ngc6440e
        original_model = copy.deepcopy(model)
        pv = pint_model_to_params(model)
        params_to_pint_model(pv, model)

        import astropy.units as u

        for name in ("RAJ", "DECJ"):
            if name not in pv.names:
                continue
            original_rad = original_model.__getattr__(name).quantity.to(u.rad).value
            restored_rad = model.__getattr__(name).quantity.to(u.rad).value
            np.testing.assert_allclose(restored_rad, original_rad, atol=1e-15)

    def test_roundtrip_epoch_params(self, ngc6440e):
        """Epoch params (PEPOCH) should round-trip within float64 precision."""
        model, _ = ngc6440e
        original_model = copy.deepcopy(model)
        pv = pint_model_to_params(model)
        params_to_pint_model(pv, model)

        for name in pv.epoch_int_values:
            original_val = float(getattr(original_model, name).value)
            restored_val = float(getattr(model, name).value)
            # float64 MJD precision is ~10 ps near MJD 53750
            assert abs(restored_val - original_val) < 1e-12, (
                f"{name}: {restored_val} != {original_val}"
            )

    def test_roundtrip_with_mask_params(self, b1855):
        """Model with mask parameters should round-trip correctly."""
        model, _ = b1855
        original_model = copy.deepcopy(model)
        pv = pint_model_to_params(model)
        params_to_pint_model(pv, model)

        for name in pv.names:
            param = getattr(model, name)
            original = getattr(original_model, name)
            if not isinstance(param, (MJDParameter, AngleParameter)):
                assert param.value == pytest.approx(original.value, rel=1e-12), (
                    f"{name}: {param.value} != {original.value}"
                )

    def test_modified_values_propagate(self, ngc6440e):
        """If we change a value in ParameterVector, it should propagate back."""
        model, _ = ngc6440e
        pv = pint_model_to_params(model)

        # Perturb F0 slightly
        f0_idx = pv.param_index("F0")
        original_f0 = float(pv.values[f0_idx])
        perturbed_f0 = original_f0 + 1e-6
        pv = pv.with_value("F0", perturbed_f0)

        params_to_pint_model(pv, model)
        assert model.F0.value == pytest.approx(perturbed_f0, rel=1e-15)
