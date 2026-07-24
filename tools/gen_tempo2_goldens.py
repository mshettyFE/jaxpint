#!/usr/bin/env python
"""Dev tool (NOT shipped): generate tempo2 reference data via libstempo.

Two goldens are written per par/tim pair, testing different layers:

``<par>.tempo2_golden``
    Residuals -- the end-to-end check. Exercises the reader, clock corrections,
    ephemeris, and timing model together. Requires a par file that actually
    phase-connects its .tim, which is not free (see ``_PARSE_WORKER``).

``<par>.parse_golden``
    Site arrival times, frequencies and errors straight out of tempo2's parser,
    before any physics. Tests only whether the two readers agree on what the
    file says. Works on pairs the residual check cannot use, and localizes a
    failure to the reader instead of leaving it anywhere in the stack.

Why generate-once rather than compare-live:

tempo2 is effectively frozen (2021.07.1 is the current release), so for a given
par/tim its output is fixed. A live comparison would recompute the same number
every run while requiring a working tempo2 on every machine that runs tests.
Committing the answer costs nothing in fidelity and removes the dependency from
CI. What live *would* buy -- protection against a stale reference -- is bought
instead by the provenance header below.

PINT's own ``.tempo2_test`` goldens record bare numbers with no header. When one
of them (B1855+09 9yr) turned out to disagree with both PINT and JaxPINT at
7.6e-07 s, there was no way to tell whether the file or the code was wrong. Every
file written here records what produced it:

* tempo2 and libstempo versions,
* a SHA-256 over the whole ``$TEMPO2/clock`` directory (61 files) -- the clock
  data is what actually drifts, and a version string does not pin it,
* the ephemeris, the par/tim, and the observation count.

Usage::

    python tools/gen_tempo2_goldens.py                 # write missing goldens
    python tools/gen_tempo2_goldens.py --force         # rewrite all
    python tools/gen_tempo2_goldens.py --list          # show the work list

Needs a working tempo2: ``$TEMPO2`` set and ``$LD_LIBRARY_PATH`` covering
libtempo2/libsofa. Each pulsar is run in a **subprocess** -- libstempo inherits
tempo2's habit of segfaulting on input it dislikes, which would otherwise take
the whole run down.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
# Inputs come from the vendored copy only -- no PINT source checkout, matching
# the test suite. An earlier version fell back to a sibling ../PINT checkout,
# which reintroduced exactly the dependency vendoring removed, and did so only
# to serve two PAIRS entries that had never produced a golden. Both are gone.
_VENDORED = _REPO / "tests" / "data" / "pint_inputs"


def _input(name: str) -> Path:
    return _VENDORED / name


_OUT_DIR = _REPO / "tests" / "data" / "tempo2_goldens"

# Pairs that load in BOTH JaxPINT and libstempo (surveyed 2026-07-21).
# Pairs failing either side are excluded rather than silently skipped; see
# --list output and the module docstring of tests/test_cross_implementation.py.
# Two entries were removed after never once producing a golden -- each run
# printed an `exit 1` that a maintainer learns to scroll past, which is where a
# real failure would hide:
#
#   B1953+29_NANOGrav_dfg+12.par  -- the file does not exist, anywhere. The
#     entry was simply wrong; the real one is the _TAI_FB90 variant, below.
#   B1937+21.basic.par + CHIME tim -- loads with **zero** TOAs (tempo2 matches
#     none of them), so `residuals()` raises. Not a pairing typo; the two files
#     genuinely do not go together.
PAIRS: list[tuple[str, str]] = [
    ("B1855+09_NANOGrav_12yv3.wb.gls.par", "B1855+09_NANOGrav_12yv3.wb.tim"),
    ("B1855+09_NANOGrav_9yv1.gls.par", "B1855+09_NANOGrav_9yv1.tim"),
    ("B1855+09_NANOGrav_dfg+12_DMX.par", "B1855+09_NANOGrav_dfg+12.tim"),
    ("B1953+29_NANOGrav_dfg+12_TAI_FB90.par", "B1953+29_NANOGrav_dfg+12.tim"),
    ("J0023+0923_NANOGrav_11yv0.gls.par", "J0023+0923_NANOGrav_11yv0.tim"),
    ("J0613-0200_NANOGrav_9yv1.gls.par", "J0613-0200_NANOGrav_9yv1.tim"),
    ("J0613-0200_NANOGrav_dfg+12_TAI_FB90.par", "J0613-0200_NANOGrav_dfg+12.tim"),
    ("J1614-2230_NANOGrav_12yv3.wb.gls.par", "J1614-2230_NANOGrav_12yv3.wb.tim"),
    ("J1643-1224_NANOGrav_9yv1.gls.par", "J1643-1224_NANOGrav_9yv1.tim"),
    (
        "J1713+0747_NANOGrav_11yv0_short.gls.ICRS.par",
        "J1713+0747_NANOGrav_11yv0_short.tim",
    ),
    ("J1713+0747_small.gls.par", "J1713+0747_small.tim"),
    ("J1853+1303_NANOGrav_11yv0.gls.par", "J1853+1303_NANOGrav_11yv0.tim"),
    ("J1909-3744.NB.par", "J1909-3744.NB.tim"),
    ("NGC6440E_PHASETEST.par", "NGC6440E_PHASETEST.tim"),
    ("ecorr_fit_test.par", "ecorr_fit_test.tim"),
    # --- Princeton (fixed-column) .tim files -------------------------------
    # tempo2 reads the Princeton format natively, so it can supply independent
    # references for a reader JaxPINT does not have yet. Generating these FIRST
    # means the parser lands with a cross-check instead of being validated only
    # against PINT -- which is how the FB2+ gap survived.
    ("NGC6440E.par", "NGC6440E.tim"),
    ("piecewise.par", "piecewise.tim"),
    ("slug.par", "slug.tim"),
    ("testtimes.par", "testtimes.tim"),
    # --- Parkes (fixed-column) --------------------------------------------
    # TEMPO's own 0437 test data, the only genuine 80-column Parkes file found
    # in ~4,400 surveyed .tim files. Vendored from FileRepo/examples.
    ("0437.par", "0437.tim"),
]

# Runs libstempo out-of-process, writing "MJD residual" per TOA to argv[3].
#
# The data goes to a *file*, not stdout: tempo2 prints warnings to stdout as
# well as stderr ("Unknown parameter in par file: DMX", "Please place MODE flags
# ..."), and an earlier version of this script had those lines land in the
# middle of the residual columns. Writing to a separate fd keeps tempo2's
# chatter and our data from ever sharing a stream.
_WORKER = r"""
import sys, warnings
warnings.simplefilter("ignore")
import libstempo
psr = libstempo.tempopulsar(parfile=sys.argv[1], timfile=sys.argv[2], maxobs=60000)
r, t = psr.residuals(), psr.toas()
with open(sys.argv[3], "w") as fh:
    fh.write("%d\n" % psr.nobs)
    for mjd, res in zip(t, r):
        fh.write("%.15f %.17e\n" % (mjd, res))
"""

# Runs libstempo out-of-process, writing what tempo2's *parser* produced.
#
# This is deliberately upstream of all physics. ``psr.stoas`` is ``obsn[].sat``,
# the site arrival time exactly as tempo2 read it from the .tim file -- no clock
# correction, no barycentring, no delay model. Comparing it to JaxPINT's parsed
# MJD tests one thing only: did the two readers pull the same numbers out of the
# same columns?
#
# Why that is worth a separate golden: residual comparisons need a par file that
# phase-connects. 0437 does not -- tempo2's own residuals span 0.997 of a pulse
# period (rms 1.55e-03 s vs the 0.87 us the par claims in TRES), so cycle
# assignment near +-P/2 is arbitrary and the element-wise difference saturates at
# one period. That made the only Parkes-format pair we have unusable as a
# residual check. It is perfectly usable as a *parser* check, because a parser
# does not care whether the ephemeris fits.
#
# ``stoas`` is a long double. Writing it as a single "%.15f" would round to
# float64 and throw away ~4e-07 s -- above the microsecond precision this project
# holds itself to elsewhere. So it is split into integer day + fractional day,
# the same int/frac representation JaxPINT uses internally; that round-trips
# exactly. ``format_float_positional`` is used because Python's ``%`` operator
# demotes a long double to float on the way in.
_PARSE_WORKER = r"""
import sys, warnings
warnings.simplefilter("ignore")
import libstempo, numpy
psr = libstempo.tempopulsar(parfile=sys.argv[1], timfile=sys.argv[2], maxobs=60000)
sat = numpy.asarray(psr.stoas)                       # long double, as parsed
day = numpy.floor(sat).astype(numpy.int64)
frac = sat - day
freq, err = numpy.asarray(psr.freqs), numpy.asarray(psr.toaerrs)
with open(sys.argv[3], "w") as fh:
    fh.write("%d\n" % psr.nobs)
    for d, f, nu, e in zip(day, frac, freq, err):
        fh.write("%d %s %.9f %.9f\n" % (
            d, numpy.format_float_positional(f, precision=18, unique=False), nu, e))
"""


def _clock_fingerprint() -> str:
    """SHA-256 over $TEMPO2/clock -- the data a version string fails to pin."""
    t2 = os.environ.get("TEMPO2")
    if not t2:
        return "unknown (TEMPO2 unset)"
    files = sorted(Path(t2, "clock").glob("*.clk"))
    h = hashlib.sha256()
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return f"sha256:{h.hexdigest()} ({len(files)} files)"


def _versions() -> dict[str, str]:
    try:
        t2v = subprocess.run(
            ["tempo2", "-v"], capture_output=True, text=True, timeout=60
        ).stdout.strip()
    except Exception:
        t2v = "unknown"
    try:
        # Deliberately not installable in CI (needs tempo2 built; see the
        # `reference` extra in pyproject.toml) -- the except arm handles
        # absence at runtime, so only the static checker needs quieting.
        import libstempo  # pyright: ignore[reportMissingImports]

        ltv = getattr(libstempo, "__version__", "unknown")
    except Exception:
        ltv = "unavailable"
    return {"tempo2": t2v, "libstempo": ltv}


def _generate(
    par: str,
    tim: str,
    *,
    source: str = _WORKER,
    ncols: int = 2,
    title: str = "tempo2 reference residuals",
    columns: str = "MJD residual_seconds",
) -> str | None:
    """Run libstempo on one pair; return the golden file text, or None on failure.

    ``source``/``ncols``/``title``/``columns`` select which worker to run and how
    to validate and label its output -- residuals (``_WORKER``, 2 columns) or
    parsed TOA fields (``_PARSE_WORKER``, 4 columns).
    """
    worker = _OUT_DIR / "_worker.py"
    payload = _OUT_DIR / "_payload.txt"
    worker.write_text(source)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                str(worker),
                str(_input(par)),
                str(_input(tim)),
                str(payload),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        raw = payload.read_text() if payload.exists() else ""
    except subprocess.TimeoutExpired:
        print(f"  {par}: TIMEOUT", file=sys.stderr)
        return None
    finally:
        worker.unlink(missing_ok=True)
        payload.unlink(missing_ok=True)

    if proc.returncode != 0 or not raw.strip():
        why = "SEGFAULT" if proc.returncode < 0 else f"exit {proc.returncode}"
        print(f"  {par}: {why}", file=sys.stderr)
        return None

    lines = raw.strip().splitlines()
    nobs, rows = int(lines[0]), lines[1:]

    # Self-check: every row must be exactly "MJD residual", and there must be
    # nobs of them. Without this, stray tempo2 output silently corrupts the
    # golden -- which is exactly what happened before the payload-file change.
    if len(rows) != nobs:
        print(f"  {par}: got {len(rows)} rows, expected nobs={nobs}", file=sys.stderr)
        return None
    bad = [r for r in rows if len(r.split()) != ncols]
    if bad:
        print(
            f"  {par}: {len(bad)} malformed rows, e.g. {bad[0][:60]!r}", file=sys.stderr
        )
        return None
    meta = {
        **_versions(),
        "clock": _clock_fingerprint(),
        "par": par,
        "tim": tim,
        "nobs": nobs,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "tools/gen_tempo2_goldens.py",
    }
    header = "\n".join(f"# {k}: {v}" for k, v in meta.items())
    return (
        f"# {title}, generated -- DO NOT EDIT.\n"
        f"{header}\n"
        f"# Columns: {columns}\n" + "\n".join(rows) + "\n"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rewrite existing goldens")
    ap.add_argument("--list", action="store_true", help="print the work list and exit")
    args = ap.parse_args(argv)

    if args.list:
        for par, tim in PAIRS:
            res = (_OUT_DIR / f"{par}.tempo2_golden").exists()
            prs = (_OUT_DIR / f"{par}.parse_golden").exists()
            flags = f"{'r' if res else '-'}{'p' if prs else '-'}"
            print(f"  [{flags}] {par}  +  {tim}")
        print("\n  r = residual golden, p = parse golden")
        return 0

    if not os.environ.get("TEMPO2"):
        print("TEMPO2 is unset; tempo2 runtime data is required.", file=sys.stderr)
        return 2

    # Two goldens per pair, from two independent tempo2 runs:
    #   .tempo2_golden -- residuals, the end-to-end check (needs a par that fits)
    #   .parse_golden  -- parsed TOA fields, the reader check (does not)
    # (suffix, worker source, column count, title, column legend)
    kinds: tuple[tuple[str, str, int, str, str], ...] = (
        (
            "tempo2_golden",
            _WORKER,
            2,
            "tempo2 reference residuals",
            "MJD residual_seconds",
        ),
        (
            "parse_golden",
            _PARSE_WORKER,
            4,
            "tempo2 parsed TOA fields (site arrival times)",
            "mjd_int mjd_frac freq_mhz err_us",
        ),
    )

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = skipped = failed = 0
    for par, tim in PAIRS:
        for suffix, source, ncols, title, columns in kinds:
            out = _OUT_DIR / f"{par}.{suffix}"
            if out.exists() and not args.force:
                skipped += 1
                continue
            text = _generate(
                par,
                tim,
                source=source,
                ncols=ncols,
                title=title,
                columns=columns,
            )
            if text is None:
                failed += 1
                continue
            out.write_text(text)
            written += 1
            print(f"  wrote {out.name}")

    print(f"\nwritten={written} skipped={skipped} failed={failed}")
    (_OUT_DIR / "PROVENANCE.json").write_text(
        json.dumps({**_versions(), "clock": _clock_fingerprint()}, indent=2) + "\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
