"""Tests for the native ``.tim`` parser (:func:`jaxpint.tim.read_tim`).

Two suites:

* A PINT-free unit suite exercising the Tempo2 line parser and the command
  state machine (TIME/PHASE/JUMP/EFAC/EQUAD/filters/INCLUDE).
* A differential-parity suite vs PINT's ``read_toa_file`` over the Tempo2-format example corpus.
"""

from __future__ import annotations

import math

import pytest

from jaxpint.tim import read_tim
from jaxpint.tim.timfile import _classify_line, _parse_tempo2_line


def _write(tmp_path, text, name="t.tim"):
    p = tmp_path / name
    p.write_text(text)
    return p


# ---------------------------------------------------------------------------
# Line-level parsing (PINT-free)
# ---------------------------------------------------------------------------


def test_parse_tempo2_line_fields():
    mjd_int, mjd_frac, freq, err, obs, flags = _parse_tempo2_line(
        "J1 1400.0 55000.5 1.0 gbt -fe Rcvr_800 -be GUPPI"
    )
    assert (mjd_int, mjd_frac) == (55000.0, 0.5)
    assert freq == 1400.0 and err == 1.0 and obs == "gbt"
    # keys lowercased + dash-stripped; values preserved verbatim
    assert flags == {"fe": "Rcvr_800", "be": "GUPPI"}


def test_parse_tempo2_line_mjd_no_fraction():
    mjd_int, mjd_frac, *_ = _parse_tempo2_line("J1 1400.0 55000 1.0 gbt")
    assert (mjd_int, mjd_frac) == (55000.0, 0.0)


def test_reserved_flag_rejected():
    with pytest.raises(ValueError, match="overwrite"):
        _parse_tempo2_line("J1 1400.0 55000.5 1.0 gbt -freq 999")


def test_odd_flags_rejected():
    with pytest.raises(ValueError, match="pairs"):
        _parse_tempo2_line("J1 1400.0 55000.5 1.0 gbt -fe")


def test_classify_line():
    assert _classify_line("# a comment") == "Comment"
    assert _classify_line("FORMAT 1") == "Command"
    assert _classify_line("   ") == "Blank"
    assert _classify_line("1  1949.6 53478.28 21.7") == "Princeton"
    # short line is Tempo2 only once FORMAT 1 has been seen
    assert _classify_line("J1 1400 55000.5 1 gbt", "Unknown") == "Unknown"
    assert _classify_line("J1 1400 55000.5 1 gbt", "Tempo2") == "Tempo2"


# ---------------------------------------------------------------------------
# Command state machine 
# ---------------------------------------------------------------------------


def test_read_tim_basic(tmp_path):
    p = _write(tmp_path, "FORMAT 1\nJ1 1400.0 55000.5 1.0 gbt\nJ2 1400.0 55001.5 2.0 gbt\n")
    res = read_tim(p)
    assert len(res.toas) == 2
    t0 = res.toas[0]
    assert (t0.mjd_int, t0.mjd_frac) == (55000.0, 0.5)
    assert t0.error_s == pytest.approx(1.0e-6)  # 1 us -> seconds
    assert t0.freq_mhz == 1400.0 and t0.obs == "gbt"


def test_zero_freq_becomes_inf(tmp_path):
    p = _write(tmp_path, "FORMAT 1\nJ1 0.0 55000.5 1.0 @\n")
    assert math.isinf(read_tim(p).toas[0].freq_mhz)


def test_time_and_phase_accumulate(tmp_path):
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "TIME 1.5\n"
        "PHASE 2\n"
        "J1 1400.0 55000.5 1.0 gbt\n",
    )
    f = read_tim(p).toas[0].flags
    # PHASE accumulates via += float(...) exactly as PINT does, so the flag is
    # the float string "2.0" (matching PINT's read_toa_file), not "2".
    assert f["to"] == "1.5" and f["phase"] == "2.0"


def test_jump_block_counter(tmp_path):
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "J0 1400 55000.0 1 gbt\n"      # outside any jump
        "JUMP\n"
        "J1 1400 55001.0 1 gbt\n"      # jump block 1
        "JUMP\n"
        "J2 1400 55002.0 1 gbt\n"      # outside again
        "JUMP\n"
        "J3 1400 55003.0 1 gbt\n"      # jump block 2
        "JUMP\n",
    )
    toas = read_tim(p).toas
    assert "jump" not in toas[0].flags
    assert toas[1].flags["jump"] == "1" and toas[1].flags["tim_jump"] == "1"
    assert "jump" not in toas[2].flags
    assert toas[3].flags["jump"] == "2"


def test_efac_equad_scale_error(tmp_path):
    p = _write(tmp_path, "FORMAT 1\nEFAC 2\nEQUAD 3\nJ1 1400 55000.5 4 gbt\n")
    # hypot(EFAC*err, EQUAD) = hypot(8, 3) us
    expected = math.hypot(2 * 4.0, 3.0) * 1e-6
    assert read_tim(p).toas[0].error_s == pytest.approx(expected)


def test_emax_and_fmin_filter(tmp_path):
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "EMAX 10\n"
        "J1 1400 55000.0 5 gbt\n"     # kept
        "J2 1400 55001.0 50 gbt\n"    # dropped (error 50 > EMAX 10)
        "FMIN 1000\n"
        "J3 800 55002.0 5 gbt\n",     # dropped (freq 800 < FMIN 1000)
    )
    toas = read_tim(p).toas
    assert len(toas) == 1 and toas[0].mjd_int == 55000.0


def test_skip_noskip(tmp_path):
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "SKIP\n"
        "J1 1400 55000.0 1 gbt\n"     # skipped
        "NOSKIP\n"
        "J2 1400 55001.0 1 gbt\n",    # kept
    )
    toas = read_tim(p).toas
    assert len(toas) == 1 and toas[0].mjd_int == 55001.0


def test_include_splices(tmp_path):
    _write(tmp_path, "FORMAT 1\nJI 1400 55005.0 1 gbt\n", name="sub.tim")
    main = _write(
        tmp_path,
        "FORMAT 1\nJ1 1400 55000.0 1 gbt\nINCLUDE sub.tim\nJ2 1400 55001.0 1 gbt\n",
        name="main.tim",
    )
    mjds = [t.mjd_int for t in read_tim(main).toas]
    assert mjds == [55000.0, 55005.0, 55001.0]


def test_princeton_raises(tmp_path):
    p = _write(tmp_path, "1  1949.609 53478.2858714192189 21.71\n")
    with pytest.raises(NotImplementedError, match="Princeton"):
        read_tim(p)


# ---------------------------------------------------------------------------
# Differential parity vs PINT's read_toa_file
# ---------------------------------------------------------------------------

_CORPUS = [
    "B1855+09_NANOGrav_9yv1.tim",
    "B1855+09_NANOGrav_dfg+12.tim",
    "J0740+6620.FCP+21.wb.tim",
    "J1614-2230_NANOGrav_12yv3.wb.tim",
]


@pytest.mark.slow
@pytest.mark.parametrize("timname", _CORPUS)
def test_parity_vs_pint_read_toa_file(timname):
    import astropy.units as u
    from pint.config import examplefile
    from pint.observatory import get_observatory
    from pint.toa import read_toa_file

    try:
        path = examplefile(timname)
        pint_toas, _ = read_toa_file(path)
    except Exception as exc:
        pytest.skip(f"PINT could not read {timname}: {exc}")

    parsed = read_tim(path)
    assert len(parsed.toas) == len(pint_toas), (len(parsed.toas), len(pint_toas))

    for raw, pt in zip(parsed.toas, pint_toas):
        # MJD: compare against PINT's precise jd1/jd2 MJD.
        pint_mjd = (pt.mjd.jd1 - 2400000.5) + pt.mjd.jd2
        assert abs((raw.mjd_int + raw.mjd_frac) - pint_mjd) < 1e-8, (timname, pint_mjd)

        # error (post EFAC/EQUAD), seconds
        assert raw.error_s == pytest.approx(pt.error.to_value(u.s), rel=1e-12)

        # frequency (0 -> inf handled both sides)
        pf = pt.freq.to_value(u.MHz)
        if math.isinf(pf):
            assert math.isinf(raw.freq_mhz)
        else:
            assert raw.freq_mhz == pytest.approx(pf, rel=1e-12)

        # observatory: raw token resolves to PINT's canonical name
        assert get_observatory(raw.obs.upper()).name == pt.obs

        # flags: PINT leaks name/format into flags via **kwargs; exclude those.
        pint_flags = {
            k: v for k, v in dict(pt.flags).items() if k not in ("name", "format")
        }
        assert raw.flags == pint_flags, (timname, raw.flags, pint_flags)
