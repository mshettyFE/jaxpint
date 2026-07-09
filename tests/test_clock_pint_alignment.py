"""Drift guard: keep JaxPINT's pinned clock snapshot and PINT's clock unified.

JaxPINT pins the IPTA ``pulsar-clock-corrections`` repo at ``SEED_CLOCK_REF``;
PINT instead resolves clock files from the repo's moving ``main`` HEAD
(``pint.observatory.global_clock_corrections.global_clock_correction_url_base``).
Left alone the two drift as upstream commits land, breaking JaxPINT-vs-PINT
parity with real (but spurious) numerical differences.

:func:`jaxpint.clock.config.set_pint_clock_override` unifies them by pointing
``$PINT_CLOCK_OVERRIDE`` at JaxPINT's snapshot, so PINT reads each clock file
from JaxPINT's pinned bytes (the highest-priority entry in PINT's resolution
order).  These tests assert the unification is

1. **complete** -- every clock file PINT needs is present in the snapshot, so
   PINT never falls through to its own ``main`` download for it, and
2. **effective** -- with the override set, PINT actually resolves each file from
   JaxPINT's snapshot rather than the global repo.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytest.importorskip("pint")
import pint.observatory as pobs

from jaxpint.clock import clock_dir, ensure_fresh
from jaxpint.clock.config import set_pint_clock_override

# The clock files the PINT parity / bridge tests actually exercise. Each must
# live in JaxPINT's pinned snapshot; a missing one sends PINT to its own `main`
# download for that name, reopening drift. The format governs the reader PINT
# uses (tempo2 for `.clk`, tempo for `.dat`).
_REQUIRED = [
    ("tai2tt_bipm2023.clk", "tempo2"),  # PINT default BIPM (pint.toa.bipm_default)
    ("tai2tt_bipm2019.clk", "tempo2"),
    ("gps2utc.clk", "tempo2"),
    ("time_gbt.dat", "tempo"),
]


@pytest.fixture(scope="module")
def snapshot() -> Path:
    """The populated JaxPINT clock snapshot dir, or skip if unavailable."""
    try:
        ensure_fresh()
    except Exception as exc:  # offline with a cold cache
        pytest.skip(f"clock snapshot unavailable (offline?): {exc}")
    dest = clock_dir()
    if not (dest / "SNAPSHOT.json").exists() and not any(dest.glob("*.clk")):
        pytest.skip("clock snapshot not downloaded")
    return dest


def test_snapshot_covers_pint_required_clock_files(snapshot: Path):
    """Every clock file PINT needs is present in JaxPINT's pinned snapshot.

    This is the contract that keeps the override *complete*: a missing file would
    send PINT to its own ``main`` download for that name, reopening drift.
    """
    missing = [name for name, _ in _REQUIRED if not (snapshot / name).exists()]
    assert not missing, (
        f"clock file(s) {missing} required by PINT are absent from JaxPINT's "
        f"snapshot at {snapshot}; PINT would fetch them from its `main` HEAD and "
        f"drift from the pinned bytes. Bump SEED_CLOCK_REF or extend the snapshot."
    )


def test_pint_resolves_clock_files_from_jaxpint_snapshot(snapshot: Path):
    """With the override set, PINT resolves each file from JaxPINT's snapshot.

    Proves the override is *effective*: ``find_clock_file`` returns the file from
    JaxPINT's snapshot dir, not PINT's global (``main``) copy.
    """
    override = set_pint_clock_override()
    assert Path(os.environ["PINT_CLOCK_OVERRIDE"]) == override == snapshot

    for name, fmt in _REQUIRED:
        try:
            cf = pobs.find_clock_file(name, format=fmt)
        except Exception as exc:  # PINT builds its global Index (needs the manifest)
            pytest.skip(f"PINT clock index unavailable (offline?): {exc}")
        resolved = Path(cf.filename)
        assert resolved == snapshot / name, (
            f"PINT resolved {name} from {resolved}, not JaxPINT's snapshot "
            f"({snapshot / name}); PINT_CLOCK_OVERRIDE is not taking effect, so "
            f"the two clock sources can drift."
        )
