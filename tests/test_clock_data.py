"""Tests for the clock data layer (:mod:`jaxpint.clock`).

All network is mocked through the single seam ``fetch._http_get`` (+ a
monkeypatched ``resolve_latest_sha``); no test touches the network. A fake
in-memory IPTA "repo" serves an ``index.txt`` + two files.
"""

from __future__ import annotations

import json
import math
import urllib.error
from pathlib import Path

import pytest

from jaxpint import clock
from jaxpint.clock import config, fetch, paths, staleness, update

REF = "deadbeef" * 5  # a fake 40-char-ish sha
NEWREF = "feedface" * 5


def _fake_repo(ref):
    base = f"{fetch.RAW_BASE}/{ref}"
    index = (
        "# File  Update  Invalid\n"
        "T2runtime/clock/gps2utc.clk        1.0   ---\n"
        "tempo/clock/time_gbt.dat           7.0   2022-05-20\n"
    )
    return {
        f"{base}/index.txt": index.encode(),
        f"{base}/T2runtime/clock/gps2utc.clk": b"# UTC(GPS) UTC\n50000 0.0\n",
        f"{base}/tempo/clock/time_gbt.dat": b"   MJD\n 50000 0.0 0.0 1\n",
    }

@pytest.fixture
def fresh_state():
    """Reset the lazy-ensure guard before/after each test."""
    paths._ensured = False
    yield
    paths._ensured = False


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("JAXPINT_CLOCK_DIR", str(tmp_path))
    return tmp_path


def _serve(monkeypatch, mapping):
    def fake_get(url, **kw):
        if url in mapping:
            return mapping[url]
        raise urllib.error.URLError(f"unmapped {url}")
    monkeypatch.setattr(fetch, "_http_get", fake_get)


# --------------------------------------------------------------------------- config


def test_config_defaults_and_override(monkeypatch):
    monkeypatch.delenv("JAXPINT_CLOCK_TTL_DAYS", raising=False)
    assert config.get("JAXPINT_CLOCK_TTL_DAYS") == 7.0
    assert config.get("JAXPINT_CLOCK_DIR") is None
    monkeypatch.setenv("JAXPINT_CLOCK_TTL_DAYS", "0")
    assert config.get("JAXPINT_CLOCK_TTL_DAYS") == 0.0


def test_config_invalid_value(monkeypatch):
    monkeypatch.setenv("JAXPINT_CLOCK_TTL_DAYS", "soon")
    with pytest.raises(ValueError, match="JAXPINT_CLOCK_TTL_DAYS"):
        config.get("JAXPINT_CLOCK_TTL_DAYS")


def test_config_describe_lists_all():
    text = config.describe()
    for name in config.OPTIONS:
        assert name in text


def test_no_stray_env_reads():
    """Only config.py may read the environment in the clock package."""
    pkg = Path(clock.__file__).parent
    for py in pkg.glob("*.py"):
        if py.name == "config.py":
            continue
        src = py.read_text()
        assert "os.environ" not in src and "os.getenv" not in src, py.name


# --------------------------------------------------------------------------- schema


def test_shipped_metadata():
    md = clock.read_metadata()
    assert {"files", "observatories", "default_bipm"} <= set(md)
    assert "gps2utc.clk" in md["files"]
    assert md["default_bipm"].upper().startswith("BIPM")
    # every observatory references files that have a schema entry or a known ext
    exts = set(md["format_by_extension"])
    for name, info in md["observatories"].items():
        for fn in info["clock_file"]:
            assert fn in md["files"] or Path(fn).suffix in exts, (name, fn)


# --------------------------------------------------------------------------- index parse


def test_parse_index_semantics():
    text = (
        "# header\n"
        "tempo/clock/time_ao.dat     inf   2022-05-20\n"
        "T2runtime/clock/gps2utc.clk 1.0   ---\n"
        "\n"
    )
    rows = fetch.parse_index(text)
    assert len(rows) == 2
    assert rows[0].file == "tempo/clock/time_ao.dat"
    assert math.isinf(rows[0].update_interval_days)
    assert rows[0].invalid_if_older_than == "2022-05-20"
    assert rows[1].invalid_if_older_than is None  # "---" -> None


# --------------------------------------------------------------------------- download


def test_download_snapshot(cache, monkeypatch):
    _serve(monkeypatch, _fake_repo(REF))
    diff = fetch.download_snapshot(REF, cache, commit_date="2026-05-01")
    assert (cache / "gps2utc.clk").exists()
    assert (cache / "time_gbt.dat").exists()
    snap = json.loads((cache / "SNAPSHOT.json").read_text())
    assert snap["ref"] == REF and snap["commit_date"] == "2026-05-01"
    assert sorted(diff["added"]) == ["gps2utc.clk", "time_gbt.dat"]
    # atomic writes leave no temp files
    assert not list(cache.glob("*.tmp"))


# --------------------------------------------------------------------------- ensure_fresh


def test_cold_cache_downloads_latest(cache, monkeypatch, fresh_state):
    _serve(monkeypatch, _fake_repo(REF))
    monkeypatch.setattr(fetch, "resolve_latest_sha", lambda: (REF, "2026-05-01"))
    paths.ensure_fresh(force=True)
    assert clock.snapshot_info()["ref"] == REF


def test_within_ttl_no_check(cache, monkeypatch, fresh_state):
    (cache / "SNAPSHOT.json").write_text(json.dumps(
        {"ref": REF, "commit_date": "2026-05-01", "checked_date": "2026-05-20"}))
    def boom():
        raise AssertionError("should not check within TTL")
    monkeypatch.setattr(fetch, "resolve_latest_sha", boom)
    paths.ensure_fresh(force=True, today="2026-05-22")  # 2 days < 7


def test_ttl_elapsed_head_unchanged_touches(cache, monkeypatch, fresh_state):
    (cache / "SNAPSHOT.json").write_text(json.dumps(
        {"ref": REF, "commit_date": "2026-05-01", "checked_date": "2026-05-01"}))
    monkeypatch.setattr(fetch, "resolve_latest_sha", lambda: (REF, "2026-05-01"))
    # _http_get must NOT be called (no file re-download)
    monkeypatch.setattr(fetch, "_http_get",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no dl")))
    paths.ensure_fresh(force=True, today="2026-06-01")
    assert clock.snapshot_info()["checked_date"] == "2026-06-01"


def test_ttl_elapsed_head_newer_redownloads(cache, monkeypatch, fresh_state):
    (cache / "SNAPSHOT.json").write_text(json.dumps(
        {"ref": REF, "commit_date": "2026-05-01", "checked_date": "2026-05-01"}))
    _serve(monkeypatch, _fake_repo(NEWREF))
    monkeypatch.setattr(fetch, "resolve_latest_sha", lambda: (NEWREF, "2026-06-01"))
    paths.ensure_fresh(force=True, today="2026-06-01")
    assert clock.snapshot_info()["ref"] == NEWREF


def test_pinned_ref_freezes(cache, monkeypatch, fresh_state):
    monkeypatch.setenv("JAXPINT_CLOCK_REF", REF)
    _serve(monkeypatch, _fake_repo(REF))
    monkeypatch.setattr(fetch, "resolve_latest_sha",
                        lambda: (_ for _ in ()).throw(AssertionError("no API on pin")))
    paths.ensure_fresh(force=True)
    assert clock.snapshot_info()["ref"] == REF


def test_offline_stale_warns(cache, monkeypatch, fresh_state):
    (cache / "SNAPSHOT.json").write_text(json.dumps(
        {"ref": REF, "commit_date": "2025-01-01", "checked_date": "2025-01-01"}))
    monkeypatch.setattr(fetch, "resolve_latest_sha",
                        lambda: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.warns(staleness.StaleClockWarning):
        paths.ensure_fresh(force=True, today="2026-05-30")


def test_offline_cold_errors(cache, monkeypatch, fresh_state):
    monkeypatch.setattr(fetch, "resolve_latest_sha",
                        lambda: (_ for _ in ()).throw(urllib.error.URLError("offline")))
    with pytest.raises(urllib.error.URLError):
        paths.ensure_fresh(force=True)


def test_update_clocks_forces_refresh(cache, monkeypatch, fresh_state):
    _serve(monkeypatch, _fake_repo(NEWREF))
    diff = update.update_clocks(ref=NEWREF)
    assert diff["ref_new"] == NEWREF
    assert clock.snapshot_info()["ref"] == NEWREF


def test_clock_file_path_missing_raises(cache, monkeypatch, fresh_state):
    _serve(monkeypatch, _fake_repo(REF))
    monkeypatch.setattr(fetch, "resolve_latest_sha", lambda: (REF, "2026-05-01"))
    assert paths.clock_file_path("gps2utc.clk").exists()
    with pytest.raises(FileNotFoundError):
        paths.clock_file_path("does_not_exist.clk")


# --------------------------------------------------------------------------- no-bloat


def test_no_bulk_clock_files_committed():
    """The repo must never *commit* .clk/.dat (they are download-only).

    Checks git's tracked set, not the working tree: the bulk snapshot is
    downloaded into ``jaxpint/data/clock/`` at runtime (and is gitignored), so
    those files legitimately exist on disk for any developer who has run the
    chain — what must stay empty is the *committed* set.
    """
    import subprocess

    repo = Path(clock.__file__).resolve().parents[2]
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "jaxpint/*.clk", "jaxpint/*.dat",
             "jaxpint/**/*.clk", "jaxpint/**/*.dat"],
            cwd=repo, capture_output=True, text=True, check=True,
        ).stdout.split()
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert not tracked, f"bulk clock files are committed (should be gitignored): {tracked}"
