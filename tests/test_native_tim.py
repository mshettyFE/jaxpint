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
# FORMAT 1 is authoritative (deliberate divergence from PINT)
#
# PINT tests its fixed-column heuristics before the Tempo2 branch, so the state
# set by FORMAT 1 cannot protect a line: any Tempo2 TOA starting with a space
# and carrying a "." in column 42 is claimed by the Parkes branch and dies.
# That makes 64 EPTA DR2 files + 1 IPTA DR1 file (5,857 TOA lines) unreadable.
# ---------------------------------------------------------------------------

# Real EPTA DR2 line: the ".cal" extension lands its dot on column 42.
_EPTA_LINE = (
    " 20120209/75624/J0613-0200-20120209-75624.cal 1396.00000000 "
    "55966.87998535736019079 1.98200 leap -fe unknown -be asterix"
)


def test_format1_beats_parkes_column_heuristic():
    """The bug: this is an ordinary tempo2 TOA, not Parkes."""
    assert _EPTA_LINE[41] == "."  # the trigger, byte-exact
    assert _classify_line(_EPTA_LINE, "Tempo2") == "Tempo2"


def test_format1_file_with_col42_dot_reads(tmp_path):
    """End-to-end: the file parses instead of raising NotImplementedError."""
    p = _write(tmp_path, "FORMAT 1\nMODE 1\n" + _EPTA_LINE + "\n")
    parsed = read_tim(p)
    assert len(parsed.toas) == 1
    assert parsed.toas[0].obs == "leap"
    assert parsed.toas[0].freq_mhz == 1396.0


def test_format1_lowercase_c_is_a_comment():
    """PINT's Princeton regex claims 'c ' before the comment check (it has a
    FIXME on that line), so a lowercase comment parses as a TOA there."""
    assert _classify_line("c lowercase comment", "Tempo2") == "Comment"


def test_legacy_dispatch_unchanged_without_format1():
    """Files that never declare FORMAT 1 must classify exactly as before."""
    parkes = (
        " PUPPI_J2044+28_58852_652 432.3420  58852.7590686063892"
        "    0.00  120.75        @"
    )
    assert _classify_line(parkes, "Unknown") == "Parkes"
    assert _classify_line("1  1949.6 53478.28 21.7", "Unknown") == "Princeton"
    assert _classify_line("@  0.000 54657.911 2788.48", "Unknown") == "Princeton"


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
    toa = read_tim(p).toas[0]
    # PHASE accumulates via += float(...) exactly as PINT does, so the flag is
    # the float string "2.0" (matching PINT's read_toa_file), not "2".
    assert toa.flags["to"] == "1.5" and toa.flags["phase"] == "2.0"
    # ...and the accumulated turns are applied to delta_pulse_number (the field
    # the phase residual actually uses), same sign as the flag.
    assert toa.delta_pulse_number == 2.0


def test_phase_accumulates_into_delta_pulse_number(tmp_path):
    # PHASE applies only to *subsequent* TOAs and accumulates (signed) across
    # commands: first TOA sees 0, then +2, then +2-3 = -1.
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "J1 1400.0 55000.5 1.0 gbt\n"
        "PHASE 2\n"
        "J1 1400.0 55001.5 1.0 gbt\n"
        "PHASE -3\n"
        "J1 1400.0 55002.5 1.0 gbt\n",
    )
    dpn = [t.delta_pulse_number for t in read_tim(p).toas]
    assert dpn == [0.0, 2.0, -1.0]


def test_phase_delta_pulse_number_parity_vs_pint(tmp_path):
    # delta_pulse_number must match PINT's get_TOAs on the same file (the flag
    # is compared by the read_toa_file parity suite; this pins the applied field).
    pytest.importorskip("pint")
    import numpy as np
    from pint.toa import get_TOAs

    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "J1 1400.0 55000.5 1.0 gbt\n"
        "PHASE 2\n"
        "J1 1400.0 55001.5 1.0 gbt\n"
        "PHASE 3\n"
        "J1 1400.0 55002.5 1.0 gbt\n",
    )
    native = np.array([t.delta_pulse_number for t in read_tim(p).toas])
    pint_dpn = np.asarray(get_TOAs(str(p)).table["delta_pulse_number"], dtype=float)
    assert np.array_equal(native, pint_dpn)


def test_padd_folds_into_delta_pulse_number(tmp_path):
    # -padd is a per-TOA (possibly fractional) phase offset; it sums with the
    # accumulated PHASE command into delta_pulse_number, same sign as each.
    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "J1 1400.0 55000.5 1.0 gbt\n"
        "J1 1400.0 55001.5 1.0 gbt -padd 0.25\n"
        "PHASE 2\n"
        "J1 1400.0 55002.5 1.0 gbt -padd -0.5\n",
    )
    dpn = [t.delta_pulse_number for t in read_tim(p).toas]
    assert dpn == [0.0, 0.25, 1.5]  # 0 ; padd ; PHASE(2) + padd(-0.5)


def test_padd_delta_pulse_number_parity_vs_pint(tmp_path):
    pytest.importorskip("pint")
    import numpy as np
    from pint.toa import get_TOAs

    p = _write(
        tmp_path,
        "FORMAT 1\n"
        "J1 1400.0 55000.5 1.0 gbt\n"
        "J1 1400.0 55001.5 1.0 gbt -padd 0.25\n"
        "PHASE 2\n"
        "J1 1400.0 55002.5 1.0 gbt -padd -0.5\n",
    )
    native = np.array([t.delta_pulse_number for t in read_tim(p).toas])
    pint_dpn = np.asarray(get_TOAs(str(p)).table["delta_pulse_number"], dtype=float)
    assert np.array_equal(native, pint_dpn)


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


# ---------------------------------------------------------------------------
# MODE (fit weighting)
#
# MODE 1 = weight by TOA uncertainty, MODE 0 = unweighted (OLS). JaxPINT always
# weights, so MODE 1 is a no-op; it used to warn on every MODE line, which is
# noise (632 of 635 MODE lines in the local corpus are MODE 1). MODE 0 asks for
# the opposite of what we do, so ignoring it silently honours the inverse of
# the request. PINT only warns there -- deliberate divergence, and cheap: no
# file in the local corpus uses MODE 0.
# ---------------------------------------------------------------------------


def test_mode_1_is_silent(tmp_path, caplog):
    p = _write(tmp_path, "FORMAT 1\nMODE 1\nJ1 1400.0 55000.5 1.0 gbt\n")
    with caplog.at_level("WARNING", logger="jaxpint.tim.timfile"):
        parsed = read_tim(p)
    assert len(parsed.toas) == 1
    assert "MODE" not in caplog.text


def test_mode_0_raises(tmp_path):
    p = _write(tmp_path, "FORMAT 1\nMODE 0\nJ1 1400.0 55000.5 1.0 gbt\n")
    with pytest.raises(NotImplementedError, match="MODE 0"):
        read_tim(p)


def test_unrecognized_mode_warns_but_parses(tmp_path, caplog):
    """An odd MODE value is not worth failing over -- weighting is unchanged."""
    p = _write(tmp_path, "FORMAT 1\nMODE 2\nJ1 1400.0 55000.5 1.0 gbt\n")
    with caplog.at_level("WARNING", logger="jaxpint.tim.timfile"):
        parsed = read_tim(p)
    assert len(parsed.toas) == 1
    assert "Unrecognized MODE" in caplog.text
