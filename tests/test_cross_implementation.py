"""Cross-implementation checks: JaxPINT vs tempo2 and TEMPO.

Agreeing with PINT is necessary but not sufficient -- PINT is sometimes the
outlier. (Ex: PINT's `.tim` dispatcher lets a fixed-column heuristic override a
declared ``FORMAT 1``, which makes 64 EPTA DR2 files unreadable.) So these
tests compare against tempo2, from two sets of precomputed residuals:

* ``<par>.tempo2_test`` in PINT's test data -- 6 usable pairs, committed
  upstream with no provenance header.
* ``tests/data/tempo2_goldens/<par>.tempo2_golden`` -- 15 pairs generated here
  by ``tools/gen_tempo2_goldens.py`` against tempo2 2021.07.1, each recording
  the tempo2/libstempo versions and a hash of the clock directory.

* ``<par>.tempo_test`` -- 4 pairs from **TEMPO** (tempo1). A separate codebase,
  so these are the only genuinely independent check here; the two sets above
  are both tempo2. Five more exist but are excluded (row-count mismatches, a
  whitened-GLS file, and one real 2.8e-04 disagreement) -- see _TEMPO_CASES.

Residuals are not the only layer checked. ``<par>.parse_golden`` records what
tempo2's *reader* produced -- site arrival times, frequencies, errors, before
any physics -- so a reader bug fails a different test than a model bug, and a
pair whose par does not phase-connect can still cross-check its parser. See the
parse-level section at the bottom of this file.

All inputs and references are vendored under ``tests/data`` (17 MB), so this
suite is self-contained -- it needs neither a PINT source checkout nor tempo2.
Only ``test_jaxpint_matches_pint_where_golden_is_stale`` imports PINT itself.

Three conventions matter and are deliberate:

**Mean subtraction.** Residuals are defined up to an additive constant (the
phase reference). PINT's own tests use ``use_weighted_mean=False``; JaxPINT's
``compute_time_residuals`` applies no mean subtraction.

**Ordering.** libstempo returns TOAs in ``.tim`` file order, and so does
JaxPINT's native loader; the goldens are written in that order too. Do NOT sort
by MJD -- these files are not MJD-ascending, and sorting one side inflates the
apparent difference by ~3 orders (7.5e-09 -> 2.8e-05 when this was got wrong).

**Ephemeris/clock.** The two sets need opposite handling. PINT's goldens were
made under specific settings, so those cases pin ``ephem``/``include_bipm`` per
file to reproduce them. The generated goldens were made by libstempo reading
each par natively, so those cases pass only ``planets=False`` and let ephem and
clock come from the par -- which is what JaxPINT does by default.

Pinning here is not the load-bearing kind seen in ``test_bary_toas.py`` (which
hid the CLK-derivation bug by pinning the value under test on both sides): it
reproduces reference conditions, and the derivation itself is tested in
``test_clock_correction.py``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

# Everything these tests read is vendored under ``tests/data``: the par/tim
# inputs here, the upstream goldens in ``pint_goldens``, ours in
# ``tempo2_goldens``. No PINT source checkout is required.
#
# It has to be vendored rather than fetched from the installed wheel:
# ``pint.config.examplefile`` reads ``pint/data/examples``, which holds 23
# files -- only 5 of the 30 inputs used here. The rest live solely in
# ``PINT/tests/datafile``, which is not packaged, and neither are the goldens.
_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"


def _example(name: str) -> str:
    """Path to a vendored par/tim input."""
    return str(_DATA / name)


# (par, tim, loader kwargs, tolerance in seconds, note)
_TEMPO2_CASES = [
    (
        "B1855+09_NANOGrav_dfg+12_TAI_FB90.par",
        "B1855+09_NANOGrav_dfg+12.tim",
        dict(ephem="DE405", planets=False, include_bipm=False),
        3e-8,  # PINT's own tolerance for this file
        "",
    ),
    (
        "B1953+29_NANOGrav_dfg+12_TAI_FB90.par",
        "B1953+29_NANOGrav_dfg+12.tim",
        dict(ephem="DE405", planets=False, include_bipm=False),
        3e-8,
        "",
    ),
    (
        "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par",
        "J0613-0200_NANOGrav_dfg+12.tim",
        dict(ephem="DE405", planets=False, include_bipm=False),
        3e-8,
        "",
    ),
    (
        "J1853+1303_NANOGrav_11yv0.gls.par",
        "J1853+1303_NANOGrav_11yv0.tim",
        dict(ephem="DE421", planets=False),
        4e-8,  # PINT's tolerance: ELL1 is carried to higher order than tempo2's
        "",
    ),
    (
        "J1744-1134.basic.par",
        "J1744-1134.Rcvr1_2.GASP.8y.x.tim",
        dict(planets=False, include_bipm=False),
        3e-8,
        "",
    ),
    (
        "B1855+09_NANOGrav_9yv1.gls.par",
        "B1855+09_NANOGrav_9yv1.tim",
        dict(ephem="DE421", planets=False),
        1e-6,
        # Looser on purpose. PINT reproduces this golden only to 7.565e-07 --
        # the *identical* figure JaxPINT gets -- so the residual disagreement is
        # between the committed file and both implementations, not between them.
        # The golden's generation settings could not be recovered; PINT has no
        # test asserting against it, which is why the drift went unnoticed.
        # test_jaxpint_matches_pint_where_golden_is_stale pins the real claim.
        "golden stale; PINT shows the same 7.565e-07 offset",
    ),
]

# Excluded from *this* set, with reasons, so the gap is visible not implied:
#   J0023+0923_NANOGrav_11yv0.gls.par -- PINT's golden has 8165 rows while the
#     .tim parses to 8161 TOAs; the 4-row difference is unexplained, so any
#     comparison would align mismatched arrays. The pulsar IS covered, via the
#     generated golden below (whose nobs=8161 matches) -- and it is the one that
#     exposed the missing FB2+ support.
#   testtimes.par -- testtimes.tim reads fine (Princeton), but the .par
#     references components not present in the ParameterVector, so no model can
#     be built. See _NO_MODEL_BUILD.


_PINT_GOLDENS = pathlib.Path(__file__).resolve().parent / "data" / "pint_goldens"


def _golden(par: str, suffix: str = ".tempo2_test") -> np.ndarray:
    """First column of an upstream golden (``residuals``); files vary in width.
    """
    g = np.genfromtxt(_PINT_GOLDENS / (par + suffix), skip_header=1)
    return g[:, 0] if g.ndim > 1 else g


def _jaxpint_residuals(par: str, tim: str, **kwargs) -> np.ndarray:
    from jaxpint.fitters import compute_time_residuals
    from jaxpint.native import get_model_and_toas
    from jaxpint.par import get_model

    par_path = _example(par)
    model, _noise, toa_data = get_model_and_toas(par_path, _example(tim), **kwargs)
    params = get_model(par_path).params
    return np.asarray(compute_time_residuals(model, toa_data, params), dtype=float)


def _demean(x: np.ndarray) -> np.ndarray:
    return x - x.mean()


@pytest.mark.slow
@pytest.mark.parametrize(
    "par,tim,kwargs,tol,note",
    _TEMPO2_CASES,
    ids=[c[0].split(".par")[0] for c in _TEMPO2_CASES],
)
def test_residuals_match_tempo2(par, tim, kwargs, tol, note):
    """JaxPINT time residuals vs tempo2's, on the same par/tim."""
    gold = _golden(par)
    resids = _jaxpint_residuals(par, tim, **kwargs)
    assert len(resids) == len(gold), (len(resids), len(gold))

    diff = np.max(np.abs(_demean(resids) - _demean(gold)))
    assert diff < tol, f"max|diff| = {diff:.3e} s exceeds {tol:.1e} s. {note}"


@pytest.mark.slow
def test_jaxpint_matches_pint_where_golden_is_stale():
    """B1855 9yv1's golden disagrees with *both* implementations equally.

    Guards the claim behind that case's loosened tolerance: JaxPINT and PINT
    agree with each other far more tightly than either agrees with the file,
    which localizes the discrepancy to the golden rather than to JaxPINT.
    """
    pytest.importorskip("pint")
    import pint.models as pm
    import pint.toa as pt
    from pint.residuals import Residuals

    par, tim = "B1855+09_NANOGrav_9yv1.gls.par", "B1855+09_NANOGrav_9yv1.tim"
    model = pm.get_model(_example(par))
    toas = pt.get_TOAs(_example(tim), ephem="DE421", planets=False)
    pint_r = Residuals(toas, model, use_weighted_mean=False).time_resids.to_value("s")

    jax_r = _jaxpint_residuals(par, tim, ephem="DE421", planets=False)
    gold = _golden(par)

    vs_pint = np.max(np.abs(_demean(jax_r) - _demean(pint_r)))
    pint_vs_gold = np.max(np.abs(_demean(pint_r) - _demean(gold)))

    assert vs_pint < 1e-8, f"JaxPINT vs PINT: {vs_pint:.3e} s"
    # The golden is the outlier: PINT is ~2 orders further from it than from us.
    assert pint_vs_gold > 10 * vs_pint, (pint_vs_gold, vs_pint)


# ---------------------------------------------------------------------------
# Generated tempo2 goldens (tests/data/tempo2_goldens/)
#
# PINT ships goldens for 6 usable pairs; these cover 15, generated once with
# tools/gen_tempo2_goldens.py against tempo2 2021.07.1. tempo2 is frozen, so a
# committed answer is as good as a live one and needs no tempo2 in CI.
#
# Unlike PINT's, every file records its provenance (tempo2 + libstempo version,
# a SHA-256 over $TEMPO2/clock, the par/tim, nobs). 
# Ordering and mean-subtraction conventions are described in the module
# docstring; both apply here unchanged.
# ---------------------------------------------------------------------------

_GOLDEN_DIR = pathlib.Path(__file__).resolve().parent / "data" / "tempo2_goldens"

# Measured 2026-07-21; tolerance is ~3x the observed max|diff| so the test
# catches regressions without being brittle.
_GENERATED_TOL = {
    "B1855+09_NANOGrav_12yv3.wb.gls.par": 6e-7,
    "B1855+09_NANOGrav_9yv1.gls.par": 8e-8,
    "B1855+09_NANOGrav_dfg+12_DMX.par": 6e-8,
    "B1953+29_NANOGrav_dfg+12_TAI_FB90.par": 3e-8,
    "J0613-0200_NANOGrav_9yv1.gls.par": 3e-6,
    "J0613-0200_NANOGrav_dfg+12_TAI_FB90.par": 2e-8,
    "J1614-2230_NANOGrav_12yv3.wb.gls.par": 4e-7,
    "J1643-1224_NANOGrav_9yv1.gls.par": 3e-8,
    "J1713+0747_NANOGrav_11yv0_short.gls.ICRS.par": 2e-7,
    "J1713+0747_small.gls.par": 5e-9,
    "J1853+1303_NANOGrav_11yv0.gls.par": 5e-8,
    "J1909-3744.NB.par": 1e-7,
    "ecorr_fit_test.par": 7e-6,
    # Princeton-format .tim, readable since the Princeton parser landed. This
    # is PINT's flagship tutorial dataset, previously unreadable by JaxPINT.
    "NGC6440E.par": 5e-7,  # measured 1.43e-07
    "J0023+0923_NANOGrav_11yv0.gls.par": 5e-8,
    #   NGC6440E_PHASETEST -- excluded from this generic test and asserted
    #     precisely in test_phasetest_disagreement_is_exactly_the_phase_command.
}


# Handled by its own test; see below.
_PHASETEST = "NGC6440E_PHASETEST.par"

# Goldens whose .tim parses fine but whose .par cannot yet build a model, so
# there is nothing to compare. Excluded explicitly rather than by a silent
# "skip if no tolerance entry", which would also hide a genuinely forgotten one.
#
#   piecewise / slug -- IFUNC/piecewise-spindown pars raise in MJD handling
#   testtimes        -- references components not present in the ParameterVector
#
# The Princeton *reader* is covered regardless: tests/test_native_tim.py checks
# all four against PINT's read_toa_file. These are model-build gaps, not parser
# gaps, and the goldens are kept so they become live the moment those land.
_NO_MODEL_BUILD = {"piecewise.par", "slug.par", "testtimes.par"}

# 0437 (Parkes format) -- excluded from the residual comparison because the
# *reference* is unusable, not because JaxPINT disagrees with it. The reader is
# fully covered: test_parsed_toas_match_tempo2 and
# test_parsed_mjds_match_the_file_text below both include this pair, and
# tests/test_native_tim.py exercises the column parser directly.
#
# The 5.83e-03 s element-wise difference is fully accounted for:
#
#   * P0 = 5.757452 ms for J0437-4715, and the disagreement is 5.83e-03 s.
#     Binning the difference by nearest whole cycle gives {-1: 584, 0: 3999,
#     +1: 580} -- 22.5% of TOAs differ by exactly one pulse period.
#   * What remains after removing whole cycles has rms 1.647e-03 s. A uniform
#     distribution over one period predicts P0/sqrt(12) = 1.662e-03 s, a 1%
#     match, and the residual difference correlates with orbital phase at 0.011
#     and with epoch at 0.002. The two phase sets are statistically independent,
#     not close-but-offset.
#   * tempo2 cannot phase-connect this pair either. Its own residuals have rms
#     1.55e-03 s = 0.269 P0 and span 5.74e-03 s = 0.997 P0 -- a full period --
#     against a median TOA error of 200 ns and a par declaring TRES 0.87 (us).
#
# So the par does not fit its .tim in either implementation, cycle assignment
# near +-P0/2 is arbitrary, and the element-wise difference saturates at one
# period. Comparing residuals here tests nothing. (The earlier reading that the
# distributions "match to 4.39e-04 s" was a red herring: that is what two
# near-uniform distributions do.)
#
# Why the par does not fit is a separate question from Parkes support and is not
# chased here. It is a 1994-96 TEMPO-era file: PBDOT 7.45 wants the TEMPO1 1e-12
# scaling, and CLK UTC(NIST) is unimplemented (JaxPINT warns and falls back to
# TT(BIPM2023); tempo2 warns and uses TT(UTC(NIST))).
_UNDIAGNOSED = {"0437.par"}


def _generated_goldens():
    if not _GOLDEN_DIR.is_dir():
        return []
    return [
        g
        for g in sorted(_GOLDEN_DIR.glob("*.tempo2_golden"))
        if not g.name.startswith(_PHASETEST)
        and not any(g.name.startswith(n) for n in _NO_MODEL_BUILD | _UNDIAGNOSED)
    ]


@pytest.mark.slow
@pytest.mark.parametrize(
    "golden", _generated_goldens(), ids=lambda p: p.name.split(".par")[0]
)
def test_residuals_match_generated_tempo2(golden):
    """JaxPINT vs freshly generated tempo2 residuals, 15 par/tim pairs."""
    meta = {}
    for line in golden.read_text().splitlines():
        if not line.startswith("#"):
            break
        key, _, val = line[1:].partition(":")
        meta[key.strip()] = val.strip()

    data = np.genfromtxt(golden, comments="#")
    gold = data[:, 1]
    par, tim = meta["par"], meta["tim"]

    resids = _jaxpint_residuals(par, tim, planets=False)
    assert len(resids) == len(gold) == int(meta["nobs"]), (
        len(resids),
        len(gold),
        meta["nobs"],
    )

    tol = _GENERATED_TOL[par]
    diff = np.max(np.abs(_demean(resids) - _demean(gold)))
    assert diff < tol, (
        f"{par}: max|diff| = {diff:.3e} s exceeds {tol:.1e} s "
        f"(tempo2 {meta['tempo2']}, clock {meta['clock'][:23]})"
    )


def test_generated_goldens_are_well_formed():
    """Every golden carries provenance and the row count it claims.

    Guards the generator: an earlier version let tempo2's stdout warnings land
    in the residual columns, which numpy only caught as a parse error.
    """
    goldens = _generated_goldens()
    assert goldens, "no generated goldens found"
    for g in goldens:
        text = g.read_text()
        for field in ("tempo2:", "libstempo:", "clock: sha256:", "nobs:", "par:"):
            assert field in text, f"{g.name} missing provenance field {field!r}"
        rows = [ln for ln in text.splitlines() if not ln.startswith("#") and ln.strip()]
        nobs = int([ln for ln in text.splitlines() if ln.startswith("# nobs")][0].split()[-1])
        assert len(rows) == nobs, f"{g.name}: {len(rows)} rows vs nobs={nobs}"
        assert all(len(r.split()) == 2 for r in rows), f"{g.name}: malformed row"


@pytest.mark.slow
def test_phasetest_disagreement_is_exactly_the_phase_command():
    """NGC6440E_PHASETEST: JaxPINT and tempo2 differ *only* by the PHASE command.

    This file looks like the worst disagreement in the suite (4.9e-03 s, five
    orders above every other pair). It is not a JaxPINT error -- it is tempo2
    declining to apply the ``PHASE`` command, and the fixture exists to exercise
    exactly that.
    """
    par, tim = "NGC6440E_PHASETEST.par", "NGC6440E_PHASETEST.tim"
    golden = _GOLDEN_DIR / f"{par}.tempo2_golden"
    if not golden.exists():
        pytest.skip("generated goldens not present")

    F0 = 61.485476554372754669  # from the par
    gold = np.genfromtxt(golden, comments="#")[:, 1]
    resids = _jaxpint_residuals(par, tim, planets=False)

    turns = (_demean(resids) - _demean(gold)) * F0
    expected = np.array([0.0, 0.2, -0.3])  # cumulative PHASE, padd already netted
    deviation = np.min(np.abs(turns[:, None] - expected[None, :]), axis=1)

    # Every TOA must sit on one of the three known offsets...
    assert deviation.max() < 1e-5, (
        f"difference not explained by PHASE: worst deviation "
        f"{deviation.max():.3e} turns ({deviation.max() / F0:.3e} s)"
    )
    # ...and the leftover scatter must be ordinary cross-implementation noise,
    # i.e. comparable to the other files rather than anything structural.
    assert deviation.max() / F0 < 3e-7, f"{deviation.max() / F0:.3e} s residual scatter"

    # And the offsets must actually be the ones the .tim asks for.
    seen = {round(float(t), 1) for t in np.round(turns, 1)}
    assert seen == {0.0, 0.2, -0.3}, seen


# ---------------------------------------------------------------------------
# TEMPO (tempo1) goldens -- the only non-tempo2 reference available
#
# Everything above compares against tempo2 twice over (PINT's committed files
# and our libstempo regeneration). These `.tempo_test` files come from TEMPO,
# a separate codebase, so they are the one genuinely independent check in the
# suite -- worth more per case than another tempo2 comparison.
#
# Format matches `.tempo2_test`: "# residuals BinaryDelay", column 0 is what
# we want.
# ---------------------------------------------------------------------------

# (par, tim, tolerance in seconds)
_TEMPO_CASES = [
    ("B1855+09_NANOGrav_dfg+12_DMX.par", "B1855+09_NANOGrav_dfg+12.tim", 5e-8),
    ("B1855+09_NANOGrav_dfg+12_modified_DD.par", "B1855+09_NANOGrav_dfg+12.tim", 4e-8),
    ("J1713+0747_NANOGrav_11yv0.gls.par", "J1713+0747_NANOGrav_11yv0_short.tim", 3e-7),
    # Simulated data with FD parameters; agrees at 3.9e-06, looser than the
    # real-data cases but stable. Not diagnosed further.
    ("test_FD.par", "test_FD.simulate.pint_corrected", 1e-5),
]

# Excluded from the TEMPO set, with reasons:
#   B1855+09_NANOGrav_dfg+12_TAI.par  -- golden has 703 rows, the .tim parses to
#     702. Off-by-one of unknown origin; comparing would misalign the arrays.
#   J1744-1134.t1.par                 -- same, 1463 vs 1462.
#   B1855+09_NANOGrav_dfg+12_modified.par -- 2.8e-04 against its golden, four
#     orders worse than the _DD and _DMX siblings on the same .tim. Diagnosed:
#     the golden is the outlier, not JaxPINT. PINT is 2.827e-04 from the same
#     file -- essentially the same distance -- so both modern implementations
#     disagree with it equally, the way they do with the B1855 9yv1 tempo2
#     golden above.
#
#     Excluded because the reference is wrong for our purposes, not because
#     JaxPINT is. Re-including it would need a golden generated with tempo2
#     (tools/gen_tempo2_goldens.py) rather than TEMPO.
#   B1855+09_NANOGrav_9yv1_whitened   -- whitened GLS residuals (print_resid -mrW),
#     which needs JaxPINT to apply the same GLS whitening before comparing.
#   J1614-2230_NANOGrav_12yv3.wb      -- wideband; pairs with the .wb.gls.par but
#     was not validated in this pass.


@pytest.mark.slow
@pytest.mark.parametrize(
    "par,tim,tol", _TEMPO_CASES, ids=[c[0].split(".par")[0] for c in _TEMPO_CASES]
)
def test_residuals_match_tempo(par, tim, tol):
    """JaxPINT time residuals vs TEMPO's, on the same par/tim."""
    gold = _golden(par, suffix=".tempo_test")
    resids = _jaxpint_residuals(par, tim, planets=False)
    assert len(resids) == len(gold), (len(resids), len(gold))

    diff = np.max(np.abs(_demean(resids) - _demean(gold)))
    assert diff < tol, f"{par}: max|diff| = {diff:.3e} s exceeds {tol:.1e} s"


# ----------------------------------------------------------------------------
# Parse-level cross-check: JaxPINT's .tim reader vs tempo2's
# ----------------------------------------------------------------------------
#
# Everything above compares *residuals*, which run the whole stack -- reader,
# clock chain, ephemeris, timing model. That makes a disagreement hard to
# attribute, and it needs a par file that phase-connects its .tim. The checks
# below stop right after reading: for each TOA, did the two readers pull the
# same numbers out of the same columns?
#
# This is what makes 0437 testable. Its residuals are excluded (see
# _UNDIAGNOSED) because tempo2 cannot phase-connect that pair either -- but the
# Parkes column parser, the thing we actually added, is fully checked here.
# Same for piecewise/slug/testtimes, which are in _NO_MODEL_BUILD: no model, but
# their readers still get exercised.

_PARSE_GOLDENS = sorted(_GOLDEN_DIR.glob("*.parse_golden"))

# Tolerance on the recombined MJD, in seconds. Set by *tempo2*, not by us:
# tempo2 rounds the arrival time through a double on the way in, so on the
# NANOGrav files (21-significant-digit MJD strings) its stored value departs
# from the file text by up to 8.4e-07 s. JaxPINT's int/frac split reproduces the
# same text to 4.8e-12 s -- the float64 limit on a fractional day, and 175,000x
# tighter. test_parsed_mjds_match_the_file_text asserts that directly, without
# tempo2 in the loop; this looser number only bounds the disagreement between
# the two readers.
_PARSE_MJD_TOL = 2e-6

# .tim directives, as opposed to TOA lines. Only used by the file-text test
# below to find the TOA lines; the reader has its own, authoritative list.
_TIM_COMMANDS = frozenset(
    {
        "FORMAT", "MODE", "C", "#", "JUMP", "TIME", "PHASE", "SKIP", "NOSKIP",
        "INFO", "EFAC", "EQUAD", "EMAX", "EMIN", "FMAX", "FMIN", "TRACK",
        "SIGMA", "END", "INCLUDE",
    }
)


def _read_parse_golden(path):
    """(meta, mjd_int, mjd_frac, freq_mhz, err_us) from a .parse_golden."""
    meta, rows = {}, []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            key, _, val = line[1:].partition(":")
            meta[key.strip()] = val.strip()
        elif line.strip():
            rows.append(line.split())
    # mjd_frac is read as longdouble: it was written from tempo2's long double
    # with 18 decimals, and np.genfromtxt would demote it to float64 here.
    return (
        meta,
        np.array([int(r[0]) for r in rows], dtype=np.int64),
        np.array([np.longdouble(r[1]) for r in rows]),
        np.array([float(r[2]) for r in rows]),
        np.array([float(r[3]) for r in rows]),
    )


@pytest.mark.parametrize(
    "golden", _PARSE_GOLDENS, ids=lambda p: p.name.split(".par")[0]
)
def test_parsed_toas_match_tempo2(golden):
    """JaxPINT's .tim reader vs tempo2's, field by field, before any physics."""
    from jaxpint.tim import read_tim

    meta, g_int, g_frac, g_freq, g_err = _read_parse_golden(golden)
    parsed = read_tim(_example(meta["tim"])).toas

    assert len(parsed) == len(g_int) == int(meta["nobs"]), (
        f"{meta['tim']}: JaxPINT read {len(parsed)} TOAs, "
        f"tempo2 read {len(g_int)} (nobs={meta['nobs']})"
    )

    mjd_int = np.array([t.mjd_int for t in parsed], dtype=np.int64)
    mjd_frac = np.array([t.mjd_frac for t in parsed])
    diff = np.abs((mjd_int - g_int) + (mjd_frac - g_frac)) * 86400.0
    assert diff.max() < _PARSE_MJD_TOL, (
        f"{meta['tim']}: max|MJD diff| = {diff.max():.3e} s "
        f"exceeds {_PARSE_MJD_TOL:.1e} s (tempo2 {meta['tempo2']})"
    )

    # Frequency and error are exact -- both readers take these straight off the
    # line with no arithmetic. The one mapping is PINT's convention that a
    # declared frequency of 0 means *infinite* frequency (a barycentric or
    # frequency-independent TOA); tempo2 keeps the literal 0.
    freq = np.array([t.freq_mhz for t in parsed])
    infinite = np.isinf(freq)
    assert np.all(g_freq[infinite] == 0.0), (
        f"{meta['tim']}: JaxPINT read infinite frequency where tempo2 did not"
    )
    assert np.array_equal(freq[~infinite], g_freq[~infinite]), (
        f"{meta['tim']}: frequency mismatch"
    )

    err_us = np.array([t.error_s for t in parsed]) * 1e6
    assert np.allclose(err_us, g_err, rtol=0, atol=1e-9), (
        f"{meta['tim']}: max|error diff| = {np.abs(err_us - g_err).max():.3e} us"
    )


def test_parse_goldens_cover_every_input_pair():
    """Every pair with a residual golden also has a parse golden.

    The generator writes both from the same PAIRS list, so a gap here means a
    tempo2 run failed silently and left the residual check as the only cover.
    """
    residual = {p.name.split(".par.")[0] for p in _GOLDEN_DIR.glob("*.tempo2_golden")}
    parsed = {p.name.split(".par.")[0] for p in _PARSE_GOLDENS}
    assert residual, "no residual goldens found"
    assert not residual - parsed, f"missing parse goldens: {sorted(residual - parsed)}"


@pytest.mark.parametrize(
    "golden", _PARSE_GOLDENS, ids=lambda p: p.name.split(".par")[0]
)
def test_parsed_mjds_match_the_file_text(golden):
    """The reader reproduces the MJD digits in the file, to float64 on the frac.

    Stronger than the tempo2 comparison above and independent of it: the .tim
    text is the ground truth, so this holds tempo2's rounding against us rather
    than adopting it as the tolerance. Guards the int/frac split -- recombining
    to a single float64 would lose ~1 us at MJD 55000, which is the whole reason
    the split exists.
    """
    from decimal import Decimal, getcontext

    from jaxpint.tim import read_tim

    getcontext().prec = 50
    meta, _g_int, _g_frac, _g_freq, _g_err = _read_parse_golden(golden)
    path = pathlib.Path(_example(meta["tim"]))
    parsed = read_tim(str(path)).toas

    # Pull the MJD token straight out of each line, in the file's own dialect.
    #
    # The dialect is decided once per *file*, not per line. Deciding per line
    # silently corrupts this test: a Tempo2 free-form line can happen to carry a
    # "." at column 41, which is the Parkes decimal-point marker, and the Parkes
    # slice then reads digits out of the middle of the TOA's name field.
    # Command lines are dropped first -- 0437.tim carries nine bare "JUMP"s, and
    # letting those into the dialect vote flips a Parkes file to free-form.
    raw = [ln for ln in path.read_text().splitlines() if ln.strip()]
    lines = [ln for ln in raw if ln.split()[0].upper() not in _TIM_COMMANDS]

    # A declared FORMAT 1 is authoritative and outranks the column heuristic --
    # the same rule the reader itself follows. Without it this test reproduces
    # PINT's dispatcher bug: every line of ecorr_fit_test.tim happens to carry a
    # "." at column 41, which is the Parkes decimal-point marker, so a pure
    # column vote reads a free-form Tempo2 file as fixed-column Parkes.
    declares_tempo2 = any(ln.split()[:2] == ["FORMAT", "1"] for ln in raw)
    parkes = (
        not declares_tempo2
        and bool(lines)
        and all(len(ln) >= 62 and ln[41] == "." for ln in lines)
    )

    tokens = []
    for line in lines:
        if parkes:
            tokens.append(line[34:41] + "." + line[42:55])
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        tokens.append(fields[2])  # Tempo2 free-form: name freq MJD err site
    if len(tokens) != len(parsed):
        pytest.skip(f"{meta['tim']}: dialect not handled by this test's tokenizer")

    worst = max(
        abs(float(Decimal(tok) - (Decimal(int(t.mjd_int)) + Decimal(float(t.mjd_frac)))))
        for t, tok in zip(parsed, tokens)
    )
    assert worst * 86400.0 < 1e-10, (
        f"{meta['tim']}: reader departs from the file text by {worst * 86400:.3e} s"
    )
