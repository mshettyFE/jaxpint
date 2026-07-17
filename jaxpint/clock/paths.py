"""Locate clock data and keep it fresh (lightweight auto-update).

Resolution: ``$JAXPINT_CLOCK_DIR`` → the packaged ``jaxpint/data/clock`` dir
(writable from a checkout). The committed *schema* (``clock_metadata.json``) is
always read from the package; the *bulk* (``index.txt``, ``*.clk``/``*.dat``,
``SNAPSHOT.json``) is downloaded into the resolved dir.

:func:`ensure_fresh` is the auto-update core — called lazily on first clock
access, never at import. Network failures are non-fatal (warn + use cache).

When the TTL has expired but the IPTA repo is unreachable, :func:`check_staleness`
emits a :class:`StaleClockWarning` if the cached snapshot is too old to trust --
the offline fallback for the normal auto-refresh.
"""

from __future__ import annotations

import datetime
import functools
import json
import urllib.error
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Optional

from . import config, fetch

_ensured = False


@functools.cache
def _bundled_clock_dir() -> Path:
    # The packaged data dir is fixed for the life of the process; resolve the
    # importlib.resources path once (clock_dir() hits this on every call).
    return Path(str(files("jaxpint.data").joinpath("clock")))


def clock_dir() -> Path:
    """The active clock directory: ``$JAXPINT_CLOCK_DIR`` or the packaged dir."""
    override = config.get("JAXPINT_CLOCK_DIR")
    if override:
        return Path(override).expanduser()
    return _bundled_clock_dir()


@functools.cache
def read_metadata() -> dict:
    """The committed read-schema (always from the package, not the cache).

    Parsed once per process and cached: the committed JSON never changes at
    runtime.  The returned dict is shared -- callers must treat it as read-only.
    """
    path = _bundled_clock_dir() / "clock_metadata.json"
    return json.loads(path.read_text())


def snapshot_info() -> Optional[dict]:
    """Parsed ``SNAPSHOT.json`` from the active dir, or ``None`` if absent."""
    return fetch.read_snapshot(clock_dir())


def _to_date(value) -> datetime.date:
    if value is None:
        return datetime.date.today()
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    return value


def _age_days(checked: Optional[str], today=None) -> float:
    if not checked:
        return float("inf")
    return (_to_date(today) - datetime.date.fromisoformat(checked)).days


HARD_STALE_DAYS = 180


class StaleClockWarning(UserWarning):
    """Emitted when clock data is old and could not be refreshed."""


def check_staleness(
    snapshot: Optional[dict],
    *,
    today=None,
    hard_days: int = HARD_STALE_DAYS,
) -> Optional[int]:
    """Warn (``StaleClockWarning``) if the snapshot is older than ``hard_days``.

    The offline fallback for the TTL auto-update: when the cache cannot be
    refreshed, warn if its data is too old to trust.  Age is measured from the
    snapshot's ``commit_date`` (the upstream IPTA commit date -- the true age of
    the clock data, not ``downloaded_date``/``checked_date``).  Distinct from
    :class:`~jaxpint.clock.clockfile.ClockCorrectionOutOfRange`, which warns
    about a TOA past a clock file's data *range* rather than the snapshot's
    *age*.  Returns the age in days when a warning fired, else ``None``.
    """
    commit_date = (snapshot or {}).get("commit_date")
    if not commit_date:
        return None
    age = (_to_date(today) - datetime.date.fromisoformat(commit_date)).days
    if age > hard_days:
        warnings.warn(
            f"clock snapshot {(snapshot or {}).get('ref')!r} is {age} days old "
            f"and could not be refreshed (offline); corrections may be out of "
            f"date. Reconnect, or pin a known commit via $JAXPINT_CLOCK_REF.",
            StaleClockWarning,
            stacklevel=2,
        )
        return age
    return None


def _download(ref: str, dest: Path, *, commit_date=None) -> dict:
    try:
        return fetch.download_snapshot(ref, dest, commit_date=commit_date)
    except PermissionError as exc:
        raise PermissionError(
            f"clock cache directory {dest} is not writable; set "
            f"$JAXPINT_CLOCK_DIR to a writable path (e.g. on a read-only install)."
        ) from exc


def _touch_checked(dest: Path, snapshot: dict, today=None) -> None:
    snapshot = dict(snapshot)
    snapshot["checked_date"] = _to_date(today).isoformat()
    fetch.write_snapshot(dest, snapshot)


def _do_ensure(today=None) -> None:
    dest = clock_dir()

    pin = config.get("JAXPINT_CLOCK_REF")
    if pin:
        snap = snapshot_info()
        if snap is None or snap.get("ref") != pin:
            _download(pin, dest)
        return  # pinned == frozen; never auto-update

    snap = snapshot_info()
    if snap is None:  # cold cache → fetch latest (needs network once)
        sha, date = fetch.resolve_latest_sha()
        _download(sha, dest, commit_date=date)
        return

    ttl = config.get("JAXPINT_CLOCK_TTL_DAYS")
    if _age_days(snap.get("checked_date"), today) <= ttl:
        return  # within cadence

    try:
        sha, date = fetch.resolve_latest_sha()
    except (urllib.error.URLError, OSError):
        check_staleness(snap, today=today)  # offline: keep cache, maybe warn
        return
    if sha == snap.get("ref"):
        _touch_checked(dest, snap, today)
    else:
        _download(sha, dest, commit_date=date)


def ensure_fresh(*, today=None, force: bool = False) -> None:
    """Lazily auto-update the cache (once per process unless ``force``)."""
    global _ensured
    if _ensured and not force:
        return
    _do_ensure(today=today)
    _ensured = True


def update_clocks(ref: str = "latest") -> dict:
    """Force a fresh clock-snapshot download (the rare manual override).

    Auto-update via :func:`ensure_fresh` is the default; this always downloads.

    Parameters
    ----------
    ref:
        ``"latest"`` (the IPTA ``main`` HEAD) or an explicit commit SHA.

    Returns
    -------
    dict
        ``{ref_old, ref_new, added, removed}``.
    """
    global _ensured
    dest = clock_dir()
    if ref == "latest":
        sha, date = fetch.resolve_latest_sha()
    else:
        sha, date = ref, None
    diff = _download(sha, dest, commit_date=date)
    _ensured = True
    return diff


def clock_file_path(name: str) -> Path:
    """Path to clock file ``name`` in the active dir (auto-updates first)."""
    ensure_fresh()
    name = Path(name).name
    path = clock_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"clock file {name!r} not found in {clock_dir()}; the snapshot may "
            f"be incomplete — try jaxpint.clock.update_clocks()."
        )
    return path
