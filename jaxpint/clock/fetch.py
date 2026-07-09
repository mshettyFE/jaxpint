"""Raw-HTTPS fetch + manifest parsing for the IPTA clock snapshot (PINT-free).

All network access funnels through the single seam :func:`_http_get`, which is
also the one place tests monkeypatch.  Uses stdlib ``urllib`` over an SSL context
backed by the ``certifi`` CA bundle (a declared dependency), which fixes the
macOS framework-Python cert gotcha.  File writes are binary + atomic
(``os.replace``), correct on both POSIX and Windows.
"""

from __future__ import annotations

import certifi
import datetime
import json
import os
import ssl
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Callable, NamedTuple, Optional

from ._pinned import IPTA_API_COMMIT as API_COMMIT
from ._pinned import IPTA_RAW_BASE as RAW_BASE


class IndexEntry(NamedTuple):
    """One ``index.txt`` row (matches PINT's ``IndexEntry`` semantics)."""

    file: str  # repo-relative path, e.g. tempo/clock/time_gbt.dat
    update_interval_days: float
    invalid_if_older_than: Optional[str]  # ISO date, or None for "---"
    extra: str


# Transient statuses worth retrying: GitHub's rate-limit (429) and the usual
# gateway/server hiccups.  A cold cache under ``pytest -n auto`` has every xdist
# worker cold-download the snapshot at once, which readily trips 429.
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _retry_delay(exc: urllib.error.HTTPError, attempt: int) -> float:
    """Seconds to wait before the next retry.

    Honors a numeric ``Retry-After`` header when present; otherwise exponential
    backoff (1, 2, 4, ... seconds).
    """
    header = exc.headers.get("Retry-After") if exc.headers else None
    if header:
        try:
            return max(0.0, float(header))
        except ValueError:
            pass  # HTTP-date form -> fall back to backoff
    return float(2**attempt)


def _http_get(
    url: str,
    *,
    accept: Optional[str] = None,
    timeout: float = 30.0,
    max_attempts: int = 4,
    _sleep: Callable[[float], None] = time.sleep,
) -> bytes:
    """The single network seam: GET ``url`` and return raw bytes.

    Every fetch in this package goes through here; tests monkeypatch it.
    Transient rate-limit / server errors (HTTP 429, 500, 502, 503, 504) are
    retried with backoff (honoring ``Retry-After``); any other error propagates
    immediately, so a genuinely offline caller still fails fast.
    """
    headers = {"User-Agent": "jaxpint-clock"}
    if accept is not None:
        headers["Accept"] = accept
    context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(max_attempts):
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code not in _RETRYABLE_STATUS or attempt == max_attempts - 1:
                raise
            _sleep(_retry_delay(exc, attempt))
    raise AssertionError("unreachable")  # pragma: no cover


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
    """Write ``data`` to ``path`` atomically and in binary (no CRLF mangling).

    Uses a uniquely-named temp file in the same directory, so concurrent writers
    (e.g. parallel ``pytest-xdist`` workers cold-downloading the same snapshot
    into the shared package dir) never race on a shared ``.tmp`` name.  The final
    ``os.replace`` is atomic and last-writer-wins on identical content.
    """
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
        old_names = {
            PurePosixPath(e.file).name for e in parse_index(old_index.read_text())
        }

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

    today = datetime.date.today().isoformat()

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
