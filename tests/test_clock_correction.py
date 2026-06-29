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
