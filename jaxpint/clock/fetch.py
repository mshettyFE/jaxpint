"""Raw-HTTPS fetch + manifest parsing for the IPTA clock snapshot (PINT-free).

All network access funnels through the single seam :func:`_http_get`, which is
also the one place tests monkeypatch.  Uses only stdlib ``urllib`` (+ ``certifi``
for the SSL context *when importable*, which fixes the macOS framework-Python
cert gotcha).  File writes are binary + atomic (``os.replace``), correct on both
POSIX and Windows.
"""

from __future__ import annotations

import datetime
import json
import os
import ssl
import urllib.request
from pathlib import Path, PurePosixPath
from typing import NamedTuple, Optional

# The IPTA repository location lives in _pinned.py (single source of truth);
# these are local aliases so the rest of this module reads naturally.
from ._pinned import IPTA_API_COMMIT as API_COMMIT
from ._pinned import IPTA_RAW_BASE as RAW_BASE


class IndexEntry(NamedTuple):
    """One ``index.txt`` row (matches PINT's ``IndexEntry`` semantics)."""

    file: str                       # repo-relative path, e.g. tempo/clock/time_gbt.dat
    update_interval_days: float     # float; "inf" -> math.inf
    invalid_if_older_than: Optional[str]  # ISO date, or None for "---"
    extra: str

def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi  # noqa: PLC0415

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover - falls back to system store
        return ssl.create_default_context()

def _http_get(url: str, *, accept: Optional[str] = None, timeout: float = 30.0) -> bytes:
    """The single network seam: GET ``url`` and return raw bytes.

    Every fetch in this package goes through here; tests monkeypatch it.
    """
    headers = {"User-Agent": "jaxpint-clock"}
    if accept is not None:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as resp:
        return resp.read()


def resolve_latest_sha() -> tuple[str, str]:
    """Return ``(sha, iso_date)`` of the IPTA repo's current ``main`` HEAD."""
    data = json.loads(_http_get(API_COMMIT, accept="application/vnd.github+json"))
    return data["sha"], data["commit"]["committer"]["date"][:10]

def parse_index(text: str) -> list[IndexEntry]:
    """Parse ``index.txt`` text into :class:`IndexEntry` rows (``---`` -> None)."""
    out: list[IndexEntry] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=3)
        if len(parts) < 3:
            continue
        out.append(
            IndexEntry(
                file=parts[0],
                update_interval_days=float(parts[1]),
                invalid_if_older_than=None if parts[2] == "---" else parts[2],
                extra=parts[3] if len(parts) > 3 else "",
            )
        )
    return out


def _atomic_write(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically and in binary (no CRLF mangling)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def _today() -> str:
    return datetime.date.today().isoformat()


# --- SNAPSHOT.json: this module owns its name, format, and read/write --------

SNAPSHOT_NAME = "SNAPSHOT.json"


def read_snapshot(dest) -> Optional[dict]:
    """Parse ``SNAPSHOT.json`` in ``dest``; ``None`` if absent or unreadable."""
    path = Path(dest) / SNAPSHOT_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):  # pragma: no cover
        return None


def write_snapshot(dest, snapshot: dict) -> None:
    """Atomically write ``snapshot`` as ``SNAPSHOT.json`` in ``dest``."""
    _atomic_write(
        Path(dest) / SNAPSHOT_NAME, (json.dumps(snapshot, indent=2) + "\n").encode()
    )


def download_snapshot(
    ref: str,
    dest,
    *,
    commit_date: Optional[str] = None,
    index_text: Optional[str] = None,
) -> dict:
    """Download the full clock snapshot at ``ref`` into ``dest``.

    Fetches ``index.txt`` (unless supplied), then every file it lists, writing
    each flat by basename, plus a ``SNAPSHOT.json`` provenance stamp.  Returns a
    diff summary (``added``/``removed`` basenames, old/new ref) vs the prior
    snapshot in ``dest``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    prior = read_snapshot(dest)
    old_names = set()
    old_index = dest / "index.txt"
    if old_index.exists():
        old_names = {PurePosixPath(e.file).name for e in parse_index(old_index.read_text())}

    base = f"{RAW_BASE}/{ref}"
    if index_text is None:
        index_text = _http_get(f"{base}/index.txt").decode()
    entries = parse_index(index_text)

    _atomic_write(dest / "index.txt", index_text.encode())
    new_names = set()
    for e in entries:
        name = PurePosixPath(e.file).name
        new_names.add(name)
        _atomic_write(dest / name, _http_get(f"{base}/{e.file}"))

    today = _today()
    write_snapshot(
        dest,
        {
            "ref": ref,
            "commit_date": commit_date,
            "downloaded_date": today,
            "checked_date": today,
            "source_url": RAW_BASE,
        },
    )

    return {
        "ref_old": (prior or {}).get("ref"),
        "ref_new": ref,
        "added": sorted(new_names - old_names),
        "removed": sorted(old_names - new_names),
    }
