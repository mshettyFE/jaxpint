"""Tests for the native clock-correction chain (jaxpint.clock.correction).

* PINT-free unit tests of the chain's gating logic (synthetic RawTOAs).
* Differential parity vs PINT's ``apply_clock_corrections`` (the ``clkcorr``
  flag), with PINT pointed at our pinned snapshot via ``$PINT_CLOCK_OVERRIDE``
  so both sides read byte-identical clock files.
"""

from __future__ import annotations


import numpy as np
import pytest

from jaxpint.clock import correct
from jaxpint.tim.raw_toa import RawTOA


def _toa(mjd, obs, flags=None):
    mi = float(int(mjd))
    return RawTOA(
        mjd_int=mi,
        mjd_frac=mjd - mi,
        error_s=1e-6,
        freq_mhz=1400.0,
        obs=obs,
        flags=flags or {},
    )


# --------------------------------------------------------------------------- units


def test_barycenter_only_time_flag():
    """Barycenter gates off gps+bipm (tdb) and has no site file -> clkcorr == -to."""
    toas = [_toa(55000.3, "@", {"to": "1.25"}), _toa(55001.0, "@")]
    out = correct(toas)
    assert out.clkcorr_seconds[0] == pytest.approx(1.25)
    assert out.clkcorr_seconds[1] == pytest.approx(0.0)


def test_time_flag_included():
    """A -to flag is added on top of the (real) clock terms."""
    base = correct([_toa(55000.3, "gbt")]).clkcorr_seconds[0]
    withto = correct([_toa(55000.3, "gbt", {"to": "2.0"})]).clkcorr_seconds[0]
    assert withto - base == pytest.approx(2.0, abs=1e-9)


def test_chime_has_gps_bipm_no_site():
    """chime: empty site clock_file but apply_gps2utc + utc -> gps + bipm only."""
    out = correct([_toa(58800.5, "chime")])
    # Non-zero (gps ~ns + bipm ~27us), and within a few tens of us.
    c = out.clkcorr_seconds[0]
    assert c != 0.0
    assert abs(c) < 1e-3


def test_include_bipm_toggle():
    on = correct([_toa(58800.5, "gbt")], include_bipm=True).clkcorr_seconds[0]
    off = correct([_toa(58800.5, "gbt")], include_bipm=False).clkcorr_seconds[0]
    # BIPM term is ~27 us; turning it off changes the result measurably.
    assert abs(on - off) > 1e-6


# --------------------------------------------------------------------- CLK/CLOCK
#
# The par's CLK line must select the clock realization, the way EPHEM selects
# the ephemeris.  It was previously ignored in favour of a hardcoded
# "BIPM2023" -- and no par file in the local corpus asks for BIPM2023, so every
# file was silently overridden.  The TT(TAI) case is the costly one: those pars
# opt out of the BIPM term entirely, and applying it anyway shifts TDB by
# ~27 us with ~1 us of variation across the span.
# Mirrors PINT's derivation in pint/toa.py:196-223.


@pytest.mark.parametrize(
    "clk,expected",
    [
        ("TT(TAI)", (False, None)),      # explicit opt-out of BIPM
        ("UNCORR", (False, None)),       # uncorrected
        ("TT(BIPM)", (True, None)),      # bare -> packaged default
        ("TT(BIPM2019)", (True, "BIPM2019")),
        ("TT(BIPM2015)", (True, "BIPM2015")),
        (" TT(BIPM2019) ", (True, "BIPM2019")),  # real pars carry stray space
        (None, (True, None)),            # no CLK line -> default
    ],
)
def test_resolve_clock_config(clk, expected):
    from jaxpint.clock.correction import resolve_clock_config

    assert resolve_clock_config(clk) == expected


def test_resolve_clock_config_unknown_warns_and_falls_back():
    """UTC(NIST) and friends appear in real pars; warn, don't crash or go silent."""
    from jaxpint.clock.correction import (
        UnsupportedClockRealization,
        resolve_clock_config,
    )

    with pytest.warns(UnsupportedClockRealization, match="UTC\\(NIST\\)"):
        assert resolve_clock_config("UTC(NIST)") == (True, None)


def test_resolve_clock_config_explicit_args_win():
    """An explicit kwarg must still override the file (loader API contract)."""
    from jaxpint.clock.correction import resolve_clock_config

    # Naming a version implies applying it, even against a TT(TAI) par --
    # otherwise the version is silently accepted and then ignored.
    assert resolve_clock_config("TT(TAI)", bipm_version="BIPM2019") == (
        True,
        "BIPM2019",
    )
    # ... but an explicit include_bipm=False still wins over that inference.
    assert resolve_clock_config(
        "TT(BIPM2019)", include_bipm=False, bipm_version="BIPM2019"
    ) == (False, "BIPM2019")
    assert resolve_clock_config("TT(BIPM2015)", include_bipm=False) == (
        False,
        "BIPM2015",
    )


def test_clk_tt_tai_changes_correction():
    """End-to-end: the derived config must actually reach ``correct``."""
    from jaxpint.clock.correction import resolve_clock_config

    inc_bipm, vers = resolve_clock_config("TT(BIPM2019)")
    on = correct([_toa(58800.5, "gbt")], include_bipm=inc_bipm, bipm_version=vers)
    inc_tai, vers_tai = resolve_clock_config("TT(TAI)")
    off = correct([_toa(58800.5, "gbt")], include_bipm=inc_tai, bipm_version=vers_tai)
    # ~27 us apart -- the whole BIPM term.
    assert abs(on.clkcorr_seconds[0] - off.clkcorr_seconds[0]) > 1e-6


# --------------------------------------------------------------------------- parity

_CORPUS = [
    "B1855+09_NANOGrav_dfg+12.tim",      # ao  (TEMPO .dat)
    "J0740+6620.FCP+21.wb.tim",          # gbt + chime
    "J1614-2230_NANOGrav_12yv3.wb.tim",  # gbt
]


@pytest.mark.slow
@pytest.mark.parametrize("timname", _CORPUS)
def test_clkcorr_parity_vs_pint(timname, _pinned_clock):
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.tim import read_tim

    try:
        path = examplefile(timname)
        toas = pt.get_TOAs(
            path, include_bipm=True, bipm_version="BIPM2023", planets=False
        )
    except OSError as exc:  # missing example file / clock or ephemeris download
        pytest.skip(f"PINT could not load {timname}: {exc}")

    pint_clk = np.array(toas.get_flag_value("clkcorr", 0.0, float)[0])
    pint_mjd = np.array(toas.get_mjds().value)  # corrected UTC MJD (float)

    out = correct(read_tim(path).toas, include_bipm=True, bipm_version="BIPM2023")
    our_mjd = out.mjd_int + out.mjd_frac

    assert len(out.clkcorr_seconds) == len(pint_clk)
    # PINT sorts TOAs by MJD; align both by their corrected MJD before comparing.
    po, jo = np.argsort(pint_mjd), np.argsort(our_mjd)
    assert np.allclose(out.clkcorr_seconds[jo], pint_clk[po], atol=1e-9, rtol=0.0), (
        timname,
        float(np.max(np.abs(out.clkcorr_seconds[jo] - pint_clk[po]))),
    )
    # corrected MJD agreement (~ns)
    assert np.allclose(our_mjd[jo], pint_mjd[po], atol=1e-9, rtol=0.0)


@pytest.mark.slow
def test_include_bipm_false_parity(_pinned_clock):
    import pint.toa as pt
    from pint.config import examplefile

    from jaxpint.tim import read_tim

    try:
        path = examplefile("J1614-2230_NANOGrav_12yv3.wb.tim")
        toas = pt.get_TOAs(path, include_bipm=False, planets=False)
    except OSError as exc:  # missing example file / clock or ephemeris download
        pytest.skip(f"PINT could not load: {exc}")

    pint_clk = np.array(toas.get_flag_value("clkcorr", 0.0, float)[0])
    pint_mjd = np.array(toas.get_mjds().value)
    out = correct(read_tim(path).toas, include_bipm=False)
    our_mjd = out.mjd_int + out.mjd_frac
    po, jo = np.argsort(pint_mjd), np.argsort(our_mjd)
    assert np.allclose(out.clkcorr_seconds[jo], pint_clk[po], atol=1e-9, rtol=0.0)
