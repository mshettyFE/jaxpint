"""Tests for absolute phase (TZR TDB extraction in the bridge layer)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("pint")  # optional dependency; skip module if absent
import pint.models as models
import pint.toa as toa
from pint.config import examplefile

from jaxpint.bridge import extract_tzr_toa, pint_toas_to_jax


@pytest.fixture(scope="module")
def ngc6440e():
    """Load NGC6440E — has TZRMJD in par file."""
    model = models.get_model(examplefile("NGC6440E.par"))
    toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
    return model, toas


class TestExtractTzrTdb:
    """Tests for extract_tzr_tdb and its integration into pint_toas_to_jax."""

    @pytest.mark.slow
    def test_explicit_absphase(self, ngc6440e):
        """Model with TZRMJD in par file produces TZR TDB values."""
        model, toas = ngc6440e
        td = pint_toas_to_jax(toas, model=model)

        assert td.tzr_tdb_int is not None
        assert td.tzr_tdb_frac is not None

    @pytest.mark.slow
    def test_auto_generated_absphase(self, ngc6440e):
        """Model without explicit AbsPhase auto-generates one."""
        model, toas = ngc6440e
        # Remove AbsPhase to force auto-generation
        model_no_abs = copy.deepcopy(model)
        model_no_abs.remove_component("AbsPhase")
        assert "AbsPhase" not in model_no_abs.components

        td = pint_toas_to_jax(toas, model=model_no_abs)

        assert td.tzr_tdb_int is not None
        assert td.tzr_tdb_frac is not None

    @pytest.mark.slow
    def test_tzr_tdb_matches_pint(self, ngc6440e):
        """Extracted TZR TDB matches PINT's TZR TOA TDB within float64."""
        model, toas = ngc6440e
        tzr_info = extract_tzr_toa(model, toas)
        tdb_int, tdb_frac = tzr_info["tdb_int"], tzr_info["tdb_frac"]

        # Get PINT's TZR TOA TDB for comparison
        tz_toas = model.components["AbsPhase"].get_TZR_toa(toas)
        pint_tdb = np.asarray(tz_toas.table["tdbld"], dtype=np.float64)[0]
        jaxpint_tdb = tdb_int + tdb_frac

        # ~1 ns precision in days is ~1e-14
        assert abs(jaxpint_tdb - pint_tdb) < 1e-13

    @pytest.mark.slow
    def test_no_model_gives_none(self, ngc6440e):
        """pint_toas_to_jax without model leaves TZR fields as None."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas, model=None)

        assert td.tzr_tdb_int is None
        assert td.tzr_tdb_frac is None

    @pytest.mark.slow
    def test_tzr_tdb_frac_in_unit_interval(self, ngc6440e):
        """Fractional day should be in [0, 1)."""
        model, toas = ngc6440e
        tzr_info = extract_tzr_toa(model, toas)

        assert 0.0 <= tzr_info["tdb_frac"] < 1.0

    @pytest.mark.slow
    def test_tzr_tdb_int_is_integer(self, ngc6440e):
        """Integer day should be a whole number."""
        model, toas = ngc6440e
        tzr_info = extract_tzr_toa(model, toas)

        assert tzr_info["tdb_int"] == int(tzr_info["tdb_int"])

    @pytest.mark.slow
    def test_tzr_freq_extracted(self, ngc6440e):
        """TZR frequency is extracted."""
        model, toas = ngc6440e
        tzr_info = extract_tzr_toa(model, toas)

        assert tzr_info["freq"] is not None
        assert tzr_info["freq"] > 0

    @pytest.mark.slow
    def test_tzr_ssb_obs_pos_extracted(self, ngc6440e):
        """TZR SSB observer position is extracted."""
        model, toas = ngc6440e
        tzr_info = extract_tzr_toa(model, toas)

        assert tzr_info["ssb_obs_pos"] is not None
        assert tzr_info["ssb_obs_pos"].shape == (3,)
