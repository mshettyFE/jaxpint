"""Tests for absolute phase (TZR TDB extraction in the bridge layer)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import pint.models as models
import pint.toa as toa
from pint.config import examplefile

from jaxpint.bridge import extract_tzr_tdb, pint_toas_to_jax


@pytest.fixture
def ngc6440e():
    """Load NGC6440E — has TZRMJD in par file."""
    model = models.get_model(examplefile("NGC6440E.par"))
    toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
    return model, toas


class TestExtractTzrTdb:
    """Tests for extract_tzr_tdb and its integration into pint_toas_to_jax."""

    def test_explicit_absphase(self, ngc6440e):
        """Model with TZRMJD in par file produces TZR TDB values."""
        model, toas = ngc6440e
        td = pint_toas_to_jax(toas, model=model)

        assert td.tzr_tdb_int is not None
        assert td.tzr_tdb_frac is not None

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

    def test_tzr_tdb_matches_pint(self, ngc6440e):
        """Extracted TZR TDB matches PINT's TZR TOA TDB within float64."""
        model, toas = ngc6440e
        tdb_int, tdb_frac = extract_tzr_tdb(model, toas)

        # Get PINT's TZR TOA TDB for comparison
        tz_toas = model.components["AbsPhase"].get_TZR_toa(toas)
        pint_tdb = np.asarray(tz_toas.table["tdbld"], dtype=np.float64)[0]
        jaxpint_tdb = tdb_int + tdb_frac

        # ~1 ns precision in days is ~1e-14
        assert abs(jaxpint_tdb - pint_tdb) < 1e-10

    def test_no_model_gives_none(self, ngc6440e):
        """pint_toas_to_jax without model leaves TZR fields as None."""
        _, toas = ngc6440e
        td = pint_toas_to_jax(toas, model=None)

        assert td.tzr_tdb_int is None
        assert td.tzr_tdb_frac is None

    def test_tzr_tdb_frac_in_unit_interval(self, ngc6440e):
        """Fractional day should be in [0, 1)."""
        model, toas = ngc6440e
        tdb_int, tdb_frac = extract_tzr_tdb(model, toas)

        assert 0.0 <= tdb_frac < 1.0

    def test_tzr_tdb_int_is_integer(self, ngc6440e):
        """Integer day should be a whole number."""
        model, toas = ngc6440e
        tdb_int, tdb_frac = extract_tzr_tdb(model, toas)

        assert tdb_int == int(tdb_int)
