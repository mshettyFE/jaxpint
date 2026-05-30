"""PINT-free unit tests for the clock-file readers (jaxpint.clock.clockfile)."""

from __future__ import annotations

import numpy as np
import pytest

from jaxpint.clock.clockfile import (
    ClockCorrectionOutOfRange,
    read_tempo2_clock_file,
    read_tempo_clock_file,
)


def _write(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text)
    return p


# --------------------------------------------------------------------------- TEMPO2


def test_tempo2_basic_seconds_to_us(tmp_path):
    # header + 3 data rows (values in SECONDS) + a comment line
    p = _write(
        tmp_path,
        "# UTC(GPS) UTC\n"
        "50000.0 1.0e-6\n"
        "# a comment in the middle\n"
        "50010.0 2.0e-6\n"
        "50020.0 3.0e-6\n",
        "x.clk",
    )
    cf = read_tempo2_clock_file(p)
    assert np.allclose(cf.mjd, [50000, 50010, 50020])
    assert np.allclose(cf.clock_us, [1.0, 2.0, 3.0])  # seconds -> us
    # linear interp at a midpoint
    assert cf.evaluate(50005.0) == pytest.approx(1.5)


def test_tempo2_bogus_last_and_leading_zero(tmp_path):
    p = _write(
        tmp_path,
        "# UTC(GPS) UTC\n"
        "0.0 9.9e-6\n"        # leading mjd==0 -> dropped
        "50000.0 1.0e-6\n"
        "50010.0 2.0e-6\n"
        "99999.0 7.7e-6\n",   # bogus last -> dropped
        "x.clk",
    )
    cf = read_tempo2_clock_file(p, bogus_last_correction=True)
    assert np.allclose(cf.mjd, [50000, 50010])


def test_tempo2_end_clamp_and_out_of_range_warns(tmp_path):
    p = _write(tmp_path, "# A B\n50000.0 1.0e-6\n50010.0 2.0e-6\n", "x.clk")
    cf = read_tempo2_clock_file(p)  # valid_beyond_ends=False
    # past-the-end warns and points the user at refreshing the snapshot
    with pytest.warns(ClockCorrectionOutOfRange, match="update_clocks"):
        val = cf.evaluate(50020.0)
    assert val == pytest.approx(2.0)  # np.interp clamps to last value
    # before-the-start warns too, but without the (irrelevant) update prompt
    with pytest.warns(ClockCorrectionOutOfRange) as rec:
        cf.evaluate(49990.0)
    assert "update_clocks" not in str(rec[-1].message)
    # valid_beyond_ends -> no warning
    cf2 = read_tempo2_clock_file(p, valid_beyond_ends=True)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert cf2.evaluate(50020.0) == pytest.approx(2.0)


def test_tempo2_bad_header_raises(tmp_path):
    p = _write(tmp_path, "50000.0 1.0e-6\n50010.0 2.0e-6\n", "x.clk")
    with pytest.raises(ValueError, match="Header line"):
        read_tempo2_clock_file(p)


# --------------------------------------------------------------------------- TEMPO


def _tempo_line(mjd, c1, c2):
    # fixed columns: mjd=l[:9], c1=l[9:21], c2=l[21:33], site=l[34]
    return f"{mjd:>9}{c1:>12}{c2:>12} g\n"


def test_tempo_basic_and_818_subtraction(tmp_path):
    text = (
        "   MJD       EECO-REF    NIST-REF NS     DATE\n"
        "=========    ========    ========\n"
        + _tempo_line("50000.00", "0.0", "2.5")        # 2.5 - 0.0 = 2.5 us
        + _tempo_line("50010.00", "818.800", "3.0")    # c1>800 -> c1-=818.8 => 0.0; 3.0-0.0=3.0
    )
    p = _write(tmp_path, text, "t.dat")
    cf = read_tempo_clock_file(p)
    assert np.allclose(cf.mjd, [50000, 50010])
    assert np.allclose(cf.clock_us, [2.5, 3.0])


def test_tempo_range_skip(tmp_path):
    text = (
        _tempo_line("38000.00", "0.0", "1.0")   # < 39000 (and !=0) -> skipped
        + _tempo_line("50000.00", "0.0", "2.0")
        + _tempo_line("99999999.0", "0.0", "9.0")  # > 100000 -> skipped
    )
    p = _write(tmp_path, text, "t.dat")
    cf = read_tempo_clock_file(p)
    assert np.allclose(cf.mjd, [50000])


def test_tempo_missing_clkcorr_defaults_zero(tmp_path):
    # only one clkcorr present -> the other defaults to 0.0
    line = f"{'50000.00':>9}{'4.0':>12}{'':>12} g\n"
    p = _write(tmp_path, line, "t.dat")
    cf = read_tempo_clock_file(p)
    # c2 missing -> 0.0; value = 0.0 - 4.0 = -4.0
    assert np.allclose(cf.clock_us, [-4.0])
