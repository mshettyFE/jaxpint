"""Tests for the native ``.tim`` parser (:func:`jaxpint.tim.read_tim`).

Two suites:

* A PINT-free unit suite exercising the Tempo2 line parser and the command
  state machine (TIME/PHASE/JUMP/EFAC/EQUAD/filters/INCLUDE).
* A differential-parity suite vs PINT's ``read_toa_file`` over the Tempo2-format example corpus.
"""

from __future__ import annotations

import math
import pathlib

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


def test_itoa_still_raises(tmp_path):
    """Princeton and Parkes are supported now; ITOA is not.

    Successor to ``test_princeton_raises`` (then ``test_parkes_and_itoa_still_
    raise``), narrowed each time a format landed. Kept so the last unsupported
    fixed-column dialect stays pinned rather than silently mis-parsing the day
    someone adds a parser without a dispatch entry.

    ITOA is deliberately last: PINT has no implementation to port (it raises
    ``RuntimeError('not implemented yet')``), and exactly one file in ~4,400
    surveyed .tim files uses it.
    """
    # Real line from NGC6440E.itoa: name cols 1-2, TOA 10-28 (decimal in col
    # 15), error 29-34, freq 35-45, obs 58-59.
    itoa = "1748-202153478.2858714192289 21.71  1949.6090  0.000000  GB"
    p = _write(tmp_path, itoa + "\n")
    with pytest.raises(NotImplementedError, match="ITOA"):
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


# ---------------------------------------------------------------------------
# Princeton (TEMPO fixed-column) format
#
# Added with tempo2 goldens generated FIRST (tools/gen_tempo2_goldens.py), so
# the reader has an independent reference rather than being checked only
# against PINT -- the gap that let the missing FB2+ support survive.
#
# Columns (1-indexed; the TEMPO and tempo2 manuals agree byte-for-byte):
#   1 obs code | 16-24 freq MHz | 25-44 TOA | 45-53 err us | 69-78 DM (optional)
# ---------------------------------------------------------------------------

# Real first line of NGC6440E.tim -- PINT's flagship tutorial dataset, which
# JaxPINT could not read at all before this parser existed.
_PRINCETON_LINE = (
    "1               1949.609 53478.2858714192189    21.71         "
)


def test_parse_princeton_line_fields():
    from jaxpint.tim.timfile import _parse_princeton_line

    mjd_int, mjd_frac, freq, err, obs, flags = _parse_princeton_line(_PRINCETON_LINE)
    assert (mjd_int, mjd_frac) == (53478.0, 0.2858714192189)
    assert freq == 1949.609
    assert err == 21.71                 # microseconds, as written
    assert obs == "1"                   # single-char code; resolved downstream
    assert flags["ddm"] == "0.0"        # DM column absent -> 0.0, as PINT does


def test_princeton_legacy_epoch_offset():
    """Integer MJD < 40000 gets +39126 (TEMPO's old day count; PINT mirrors it).

    It silently rewrites dates, so the guard matters: a 1970s-era TOA must shift
    while a modern one must not.
    """
    from jaxpint.tim.timfile import _parse_princeton_line

    old = "1               1949.609   382.2858714192189    21.71"
    mjd_int, _, _, _, _, _ = _parse_princeton_line(old)
    assert mjd_int == 382 + 39126
    # ...and a modern MJD is untouched
    mjd_int, _, _, _, _, _ = _parse_princeton_line(_PRINCETON_LINE)
    assert mjd_int == 53478.0


def test_princeton_dm_column_is_read_when_present():
    from jaxpint.tim.timfile import _parse_princeton_line

    with_dm = _PRINCETON_LINE.ljust(68) + "  1.25e-03"  # cols 69-78, 10 wide
    *_, flags = _parse_princeton_line(with_dm)
    assert float(flags["ddm"]) == pytest.approx(1.25e-3)


def test_princeton_short_line_rejected():
    from jaxpint.tim.timfile import _parse_princeton_line

    with pytest.raises(ValueError, match="too short"):
        _parse_princeton_line("1  1949.609 53478.28")


def test_read_tim_princeton_file(tmp_path):
    """End-to-end through read_tim, including the classifier dispatch."""
    p = _write(tmp_path, _PRINCETON_LINE + "\n" + _PRINCETON_LINE + "\n")
    parsed = read_tim(p)
    assert len(parsed.toas) == 2
    t = parsed.toas[0]
    assert t.obs == "1"
    assert t.freq_mhz == 1949.609
    assert t.error_s == pytest.approx(21.71e-6)   # us -> s


@pytest.mark.slow
@pytest.mark.parametrize("stem", ["NGC6440E", "piecewise", "slug", "testtimes"])
def test_princeton_parity_vs_pint_read_toa_file(stem):
    """Every Princeton file in the corpus parses identically to PINT.

    Uses ``read_toa_file`` (the raw reader), not ``get_TOAs``: the latter
    applies clock corrections -- ~28 us at GBT -- so comparing against it made
    the site-'1' files look 2.8e-05 s wrong while the barycentre-'@' files
    matched exactly. That asymmetry was the tell.
    """
    pytest.importorskip("pint")
    import astropy.units as u
    from pint.observatory import get_observatory
    from pint.toa import read_toa_file

    data = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
    path = str(data / f"{stem}.tim")
    pint_toas, _ = read_toa_file(path)
    parsed = read_tim(path)
    assert len(parsed.toas) == len(pint_toas)

    for raw, pt in zip(parsed.toas, pint_toas):
        pint_mjd = (pt.mjd.jd1 - 2400000.5) + pt.mjd.jd2
        assert abs((raw.mjd_int + raw.mjd_frac) - pint_mjd) < 1e-8, stem
        assert raw.error_s == pytest.approx(pt.error.to_value(u.s), rel=1e-12)
        # JaxPINT keeps the raw single-char code; canonical resolution is a
        # downstream concern (same convention as the Tempo2 parser).
        assert get_observatory(raw.obs.upper()).name == pt.obs


# ---------------------------------------------------------------------------
# Parkes (fixed-column) format
#
# Columns (1-indexed; TEMPO and tempo2 manuals agree):
#   1 blank | 2-25 name | 26-34 freq MHz | 35-55 TOA (decimal in col 42)
#   56-63 phase offset (fraction of P0) | 64-71 err us | 80 obs code
#
# Two Parkes files exist locally, and they need different references:
#
#   0437.tim     -- PINT cannot read it at all. It slices cols 2-25 into a
#                   "name" flag, and its own FlagDict rejects the whitespace
#                   that field contains:
#                     ValueError: value 'c940121_173431.FF   0  0' for key name
#                     cannot contain whitespace
#                   JaxPINT succeeds because it discards the name field, the
#                   same choice already made in the Tempo2 parser. tempo2 is the
#                   reference instead (tests/test_cross_implementation.py).
#   parkes.toa   -- labels carry no internal whitespace, so PINT reads it and a
#                   direct parse-parity test is possible, as with Princeton.
#
# They are not redundant: 0437 is observatory code "7" with short labels,
# parkes.toa is "@" (barycentre) with labels filling all of cols 2-25.
# ---------------------------------------------------------------------------

# Real first line of TEMPO's 0437 test data -- the only genuine 80-column Parkes
# file found in ~4,400 surveyed .tim files.
_PARKES_LINE = (
    " c940121_173431.FF   0  0 1522.369  49373.2743634850894"
    "    0.00     0.2        7"
)


def test_parse_parkes_line_fields():
    from jaxpint.tim.timfile import _parse_parkes_line

    assert _PARKES_LINE[41] == "."          # decimal must be in column 42
    assert len(_PARKES_LINE) == 80          # obs code lives in column 80

    mjd_int, mjd_frac, freq, err, obs, flags = _parse_parkes_line(_PARKES_LINE)
    assert (mjd_int, mjd_frac) == (49373.0, 0.2743634850894)
    assert freq == 1522.369
    assert err == 0.2                       # microseconds, as written
    assert obs == "7"                       # parkes; resolved downstream
    assert flags == {}                      # Parkes carries no flags


def test_parkes_nonzero_phase_offset_raises():
    """Columns 56-63 shift the TOA by a fraction of P0; neither we nor PINT
    apply it, so honouring the column silently is not an option."""
    from jaxpint.tim.timfile import _parse_parkes_line

    shifted = _PARKES_LINE[:55] + "    0.25" + _PARKES_LINE[63:]
    with pytest.raises(NotImplementedError, match="phase offset"):
        _parse_parkes_line(shifted)


def test_parkes_decimal_column_enforced():
    """The format guarantees the decimal position; a shifted one is corrupt."""
    from jaxpint.tim.timfile import _parse_parkes_line

    bad = _PARKES_LINE[:41] + "X" + _PARKES_LINE[42:]
    with pytest.raises(ValueError, match="column 42"):
        _parse_parkes_line(bad)


def test_parkes_short_line_rejected():
    from jaxpint.tim.timfile import _parse_parkes_line

    with pytest.raises(ValueError, match="too short"):
        _parse_parkes_line(" c940121  1522.369  49373.27")


@pytest.mark.slow
def test_read_tim_parkes_corpus_file():
    """The full 0437 file: 5163 TOAs, read end-to-end through the classifier."""
    data = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
    parsed = read_tim(str(data / "0437.tim"))
    assert len(parsed.toas) == 5163          # matches tempo2's own count
    t = parsed.toas[0]
    assert t.obs == "7"
    assert t.freq_mhz == 1522.369
    assert t.error_s == pytest.approx(0.2e-6)
    assert (t.mjd_int, t.mjd_frac) == (49373.0, 0.2743634850894)


def test_read_tim_parkes_barycentre_file():
    """parkes.toa: 8 TOAs at the barycentre, labels filling cols 2-25.

    Complements 0437.tim, which is observatory "7" with short labels. This one
    exercises the "@" barycentre code and a name field that runs the full width
    of its columns -- the case where an off-by-one in the freq slice would show
    up, since there is no padding between the label and the frequency.
    """
    data = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
    parsed = read_tim(str(data / "parkes.toa"))
    assert len(parsed.toas) == 8
    assert {t.obs for t in parsed.toas} == {"@"}
    t = parsed.toas[0]
    assert t.freq_mhz == 432.3420
    assert t.error_s == pytest.approx(120.75e-6)
    assert (t.mjd_int, t.mjd_frac) == (58852.0, 0.7590686063892)


def test_parkes_parity_vs_pint_read_toa_file():
    """JaxPINT and PINT extract identical fields from parkes.toa.

    ``read_toa_file`` is the raw reader: no clock corrections, no barycentring.
    Comparing against ``get_TOAs`` instead would be comparing timescales rather
    than parsers -- when that mistake was made for Princeton it produced a 28 us
    GBT offset, with the "@"-site files matching exactly, which was the tell.

    ``read_toa_file`` returns ``(toas, commands)``; only the first is wanted.
    """
    pint_toa = pytest.importorskip("pint.toa")

    data = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
    path = str(data / "parkes.toa")
    ours = read_tim(path).toas
    theirs, _commands = pint_toa.read_toa_file(path)

    assert len(ours) == len(theirs) == 8
    for a, b in zip(ours, theirs):
        mjd_diff = abs(
            float(b.mjd.jd1 - 2400000.5 - a.mjd_int) + float(b.mjd.jd2 - a.mjd_frac)
        )
        assert mjd_diff == 0.0
        assert a.freq_mhz == float(b.freq.value)
        assert a.error_s == float(b.error.to("s").value)
        # PINT canonicalizes the site code; we keep the raw token, as for
        # Princeton. Comparing a.obs to b.obs directly would fail on that alone.
        assert (a.obs, b.obs) == ("@", "barycenter")


# ---------------------------------------------------------------------------
# Generated Parkes fixtures
#
# Only two Parkes files exist locally (0437.tim, parkes.toa) and both are
# well-behaved: obs code present, phase offset "    0.00", full 80 columns. A
# whole-filesystem sweep in 2026-07 found no others, and no in-the-wild PTA
# release ships the format -- so the column edge cases cannot be covered by
# vendoring more data, only by constructing it.
#
# Every fixture below is built by _parkes_line from named fields, so a test says
# which column it is probing instead of hiding it in an 80-character literal.
# ---------------------------------------------------------------------------


def _parkes_line(
    name="testTOA_0001",
    freq="1440.000",
    mjd_int="55000",
    mjd_frac="5000000000000",
    phase="    0.00",
    error="    1.50",
    obs="7",
):
    """Assemble one 80-column Parkes line from its fields.

    Columns (1-indexed): 1 blank | 2-25 name | 26-34 freq | 35-41 MJD int |
    42 "." | 43-55 MJD frac | 56-63 phase | 64-71 error | 72-79 blank | 80 obs.

    Fields are placed by width, so passing an over-wide value shifts everything
    after it -- which is what makes the misalignment fixtures below realistic
    rather than hand-corrupted.
    """
    line = (
        " "
        + f"{name:<24}"[:24]
        + f"{freq:>9}"[:9]
        + f"{mjd_int:>7}"[:7]
        + "."
        + f"{mjd_frac:<13}"[:13]
        + f"{phase:>8}"[:8]
        + f"{error:>8}"[:8]
        + " " * 8
        + obs
    )
    assert len(line) == 80, f"fixture built {len(line)} columns, not 80"
    return line


def test_parkes_fixture_builder_matches_the_real_corpus():
    """The builder reproduces a real 0437 line, so fixtures are not fiction.

    Without this, every test below could be self-consistent against a layout
    that no writer actually emits.
    """
    from jaxpint.tim.timfile import _parse_parkes_line

    built = _parkes_line(
        name="c940121_173431.FF   0  0",
        freq="1522.369",
        mjd_int="49373",
        mjd_frac="2743634850894",
        phase="    0.00",
        error="     0.2",
        obs="7",
    )
    assert built == _PARKES_LINE
    assert _parse_parkes_line(built) == _parse_parkes_line(_PARKES_LINE)


def test_parkes_phase_offset_read_across_full_width():
    """A phase offset in column 63 is caught (PINT reads only 56-62 and misses it).

    The regression this pins: "0.000001" right-justified in the 8-column phase
    field reads as "0.00000" under PINT's line[55:62], parses to 0.0, and is
    accepted -- silently shifting the TOA by a fraction of P0, which is exactly
    what raising here is supposed to prevent.
    """
    from jaxpint.tim.timfile import _parse_parkes_line

    line = _parkes_line(phase="0.000001")
    assert line[62] == "1", "the probe digit must land in column 63"
    with pytest.raises(NotImplementedError, match="phase offset"):
        _parse_parkes_line(line)


def test_parkes_phase_offset_zero_variants_accepted():
    """Zero written any of the ways a writer might pad it."""
    from jaxpint.tim.timfile import _parse_parkes_line

    for phase in ("    0.00", "     0.0", "       0", "0.000000", "    0.00"):
        assert _parse_parkes_line(_parkes_line(phase=phase))[0] == 55000.0


def test_parkes_line_missing_observatory_column_defaults_to_barycentre():
    """Writers that stop at column 71 leave no obs code; the parser assumes "@".

    Documented behaviour rather than an accident -- the parser guards on
    len(line) > 79. Pinned because silently defaulting a *site* to the
    barycentre would move every TOA by up to the Roemer delay (~500 s), so if
    this default is ever wrong it should fail loudly here first.
    """
    from jaxpint.tim.timfile import _parse_parkes_line

    truncated = _parkes_line()[:71]
    assert len(truncated) == 71
    assert _parse_parkes_line(truncated)[4] == "@"


def test_parkes_line_one_column_short_of_the_error_field_rejected():
    """70 columns cannot hold the error field (64-71), so it must raise."""
    from jaxpint.tim.timfile import _parse_parkes_line

    with pytest.raises(ValueError, match="too short"):
        _parse_parkes_line(_parkes_line()[:70])


def test_parkes_column_slide_is_always_caught():
    """A field overrunning its columns slides every later field, and must raise.

    This is the realistic corruption mode for a fixed-column format: a writer
    emits one character too many and everything to its right shifts. Nothing
    about the line looks malformed -- it is still 80-ish printable columns -- so
    the only defence is that some field stops parsing.

    Two guards catch it depending on where the slide starts, and the test pins
    both. A slide from the name field lands non-numeric text in the frequency
    columns, so float() rejects it first; a slide starting after the frequency
    reaches the decimal-column check instead. Asserting only the second would
    have failed here, because the frequency guard fires earlier than expected.
    """
    from jaxpint.tim.timfile import _parse_parkes_line

    good = _parkes_line()
    assert good[41] == "."

    # Inserting a character *overwrites nothing* -- it pushes columns right.
    from_name = good[:25] + "x" + good[25:]
    assert from_name[41] != "."
    with pytest.raises(ValueError, match="could not convert string to float"):
        _parse_parkes_line(from_name)

    # Slide starting after the frequency field: freq still parses, so the
    # decimal-column check is what stands between this and a wrong MJD.
    from_mjd = good[:34] + "0" + good[34:]
    assert from_mjd[25:34] == good[25:34] and from_mjd[41] != "."
    with pytest.raises(ValueError, match="decimal point must be in column 42"):
        _parse_parkes_line(from_mjd)


def test_parkes_non_numeric_frequency_rejected():
    """Column 26-34 must parse as a float; a blank or text field is not silently 0."""
    from jaxpint.tim.timfile import _parse_parkes_line

    with pytest.raises(ValueError):
        _parse_parkes_line(_parkes_line(freq="   badval"))


def test_parkes_generated_file_reads_end_to_end(tmp_path):
    """A synthesized multi-TOA file routes through the classifier, not just the parser.

    The parse-level tests above call _parse_parkes_line directly, which bypasses
    dialect detection entirely -- so without this, a classifier change could stop
    recognizing Parkes and every one of them would still pass.
    """
    lines = [
        _parkes_line(
            name=f"gen_{i:04d}",
            freq=f"{1400.0 + i:.3f}",
            mjd_int=f"{55000 + i}",
            error=f"{1.0 + i:.2f}",
            obs="7",
        )
        for i in range(5)
    ]
    p = _write(tmp_path, "\n".join(lines) + "\n", name="generated.tim")
    parsed = read_tim(p)

    assert len(parsed.toas) == 5
    assert [t.mjd_int for t in parsed.toas] == [55000.0 + i for i in range(5)]
    assert parsed.toas[0].freq_mhz == 1400.0
    assert parsed.toas[4].error_s == pytest.approx(5.0e-6)
    assert {t.obs for t in parsed.toas} == {"7"}
