"""Locate clock data and keep it fresh (lightweight auto-update).

Resolution: ``$JAXPINT_CLOCK_DIR`` → the packaged ``jaxpint/data/clock`` dir
(writable from a checkout). The committed *schema* (``clock_metadata.json``) is
always read from the package; the *bulk* (``index.txt``, ``*.clk``/``*.dat``,
``SNAPSHOT.json``) is downloaded into the resolved dir.

:func:`ensure_fresh` is the auto-update core — called lazily on first clock
access, never at import. Network failures are non-fatal (warn + use cache).
"""

from __future__ import annotations

import datetime
import json
import urllib.error
from importlib.resources import files
from pathlib import Path
from typing import Optional

from . import config, fetch
from .staleness import check_staleness

_ensured = False


def _bundled_clock_dir() -> Path:
    return Path(str(files("jaxpint.data").joinpath("clock")))


def clock_dir() -> Path:
    """The active clock directory: ``$JAXPINT_CLOCK_DIR`` or the packaged dir."""
    override = config.get("JAXPINT_CLOCK_DIR")
    if override:
        return Path(override).expanduser()
    return _bundled_clock_dir()


def read_metadata() -> dict:
    """The committed read-schema (always from the package, not the cache)."""
    path = _bundled_clock_dir() / "clock_metadata.json"
    return json.loads(path.read_text())


def snapshot_info() -> Optional[dict]:
    """Parsed ``SNAPSHOT.json`` from the active dir, or ``None`` if absent."""
    return fetch.read_snapshot(clock_dir())


def read_index() -> list[fetch.IndexEntry]:
    """Parsed ``index.txt`` from the active dir (empty if not downloaded yet)."""
    path = clock_dir() / "index.txt"
    if not path.exists():
        return []
    return fetch.parse_index(path.read_text())


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


def clock_file_path(name: str) -> Path:
    """Path to clock file ``name`` in the active dir (auto-updates first)."""
    ensure_fresh()
    path = clock_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"clock file {name!r} not found in {clock_dir()}; the snapshot may "
            f"be incomplete — try jaxpint.clock.update_clocks()."
        )
    return path
