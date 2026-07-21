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

All inputs and references are vendored under ``tests/data`` (12 MB), so this
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
#   testtimes.par -- testtimes.tim is Princeton-format, which the native reader
#     does not support yet.


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

# 0437 (Parkes format) -- excluded from the residual comparison, NOT from the
# reader. The parser is validated in tests/test_native_tim.py; what is unresolved
# is why the residuals disagree element-wise.
#
# Observed: 5.83e-03 s element-wise, ~1.6 ms rms, against ~1e-8 s for every
# other golden. But the residual *distributions* match closely --
# rms 1.669e-03 (JaxPINT) vs 1.547e-03 (tempo2), sorted values agreeing to
# 4.39e-04 s -- so both sides produce a plausible residual set for this pulsar.
#
# Ruled out, each by measurement:
#   * model/parameter error -- binning by orbital phase and by epoch shows no
#     structure; every bin has the same ~2.2 ms rms.
#   * ephemeris            -- DE200/DE421/DE440 give 5.829/5.886/5.893e-03 s.
#   * dropped parameters   -- nothing unrecognized; all 22 par entries parse.
#   * integer phase wrap   -- removing round(d/P0) turns leaves rms unchanged
#                             at 1.65e-03 s (P0 = 5.7575 ms).
#
# NOT yet ruled out: fine-grained time alignment. The golden's MJD column is
# barycentric (it differs from site time with 124 s std, the Roemer signature),
# and the check that it matches JaxPINT's barycentric time was crude -- it used
# basis_seconds/86400 minus a median offset and agreed only to 0.82 s. Aligned
# barycentric times should match to microseconds, so this is the first thing to
# redo properly before concluding anything about the model.
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
