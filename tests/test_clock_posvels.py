"""Regression tests for the cert-resilient ephemeris loader in clock/posvels.py.

The bug being guarded against: ``compute_posvels`` used to call
``astropy.coordinates.solar_system_ephemeris.set("DE440")`` directly, which
hits astropy's default HTTPS URL. In an HPC container without a working CA
bundle, TLS verification fails and the entire pipeline stops at TOA load.

The fix replaces that with a fallback chain whose final step feeds
``astropy.utils.data.download_file`` a ``sources=`` list with FTP first (no
TLS), matching the trick PINT uses in
``pint.solar_system_ephemerides.load_kernel``.

These tests pin the parts of that fallback that, if regressed, would
re-introduce the broken-SSL failure. They run fully offline.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jaxpint.clock.posvels import (
    _EPHEMERIS_MIRRORS,
    _LOADED_EPHEMS,
    _ensure_ephemeris,
)


@pytest.fixture(autouse=True)
def _clear_module_cache():
    """Reset the process-local cache so tests don't see each other."""
    _LOADED_EPHEMS.clear()
    yield
    _LOADED_EPHEMS.clear()


def test_first_mirror_is_ftp():
    """The SSL-bypass works because the first mirror uses FTP (no TLS).

    If someone reorders this list and puts HTTPS first, the fallback regresses
    to the original failure mode in CA-less containers.
    """
    assert _EPHEMERIS_MIRRORS[0].startswith("ftp://"), (
        "First mirror must be FTP to dodge TLS in CA-less containers; "
        f"got {_EPHEMERIS_MIRRORS[0]!r}"
    )


def test_env_var_path_short_circuits_network(tmp_path, monkeypatch):
    """``JAXPINT_EPHEM_PATH`` must let users pre-stage the kernel and bypass
    every network call — the airtight option on locked-down nodes."""
    bsp = tmp_path / "de440.bsp"
    bsp.touch()
    monkeypatch.setenv("JAXPINT_EPHEM_PATH", str(bsp))
    sse = MagicMock()
    dl = MagicMock()
    monkeypatch.setattr("jaxpint.clock.posvels.solar_system_ephemeris", sse)
    monkeypatch.setattr("jaxpint.clock.posvels.download_file", dl)

    assert _ensure_ephemeris("DE440") == str(bsp)
    sse.set.assert_not_called()
    dl.assert_not_called()


def test_env_var_directory_resolves_to_named_bsp(tmp_path, monkeypatch):
    """If ``JAXPINT_EPHEM_PATH`` is a directory, the loader picks up
    ``<dir>/<ephem>.bsp`` inside it (matches PINT's _load_kernel_local)."""
    (tmp_path / "de440.bsp").touch()
    monkeypatch.setenv("JAXPINT_EPHEM_PATH", str(tmp_path))
    monkeypatch.setattr("jaxpint.clock.posvels.solar_system_ephemeris", MagicMock())
    monkeypatch.setattr("jaxpint.clock.posvels.download_file", MagicMock())

    resolved = _ensure_ephemeris("DE440")
    assert resolved.endswith("/de440.bsp")


def test_falls_back_to_download_file_with_ftp_first(monkeypatch):
    """When astropy's name-based resolver raises (the original SSL bug), the
    loader must fall through to ``download_file(sources=[ftp, https, ...])``.

    This is the load-bearing test: it pins the exact mechanism that fixed the
    cluster failure. Removing the ``sources=`` argument or reordering the list
    to put HTTPS first re-breaks containers without CA bundles.
    """
    bad_sse = MagicMock()
    bad_sse.set.side_effect = OSError(
        "SSL: CERTIFICATE_VERIFY_FAILED (simulated)"
    )
    monkeypatch.setattr("jaxpint.clock.posvels.solar_system_ephemeris", bad_sse)
    dl = MagicMock(return_value="/fake/cache/de440.bsp")
    monkeypatch.setattr("jaxpint.clock.posvels.download_file", dl)

    assert _ensure_ephemeris("DE440") == "/fake/cache/de440.bsp"

    dl.assert_called_once()
    sources = dl.call_args.kwargs.get("sources")
    assert sources is not None, (
        "must pass sources= so astropy walks the mirror list"
    )
    assert sources[0].startswith("ftp://"), (
        f"first source must be FTP — that IS the workaround; got {sources[0]!r}"
    )
    assert sources[0].endswith("de440.bsp"), (
        f"first source must end with the requested .bsp; got {sources[0]!r}"
    )


def test_cache_skips_repeat_resolution(monkeypatch):
    """Second call for the same ephem should be a no-op (no astropy, no
    download). In-process caching matters because compute_posvels can be
    called once per pulsar in a hot loop.
    """
    sse = MagicMock()
    monkeypatch.setattr("jaxpint.clock.posvels.solar_system_ephemeris", sse)
    monkeypatch.setattr("jaxpint.clock.posvels.download_file", MagicMock())

    _ensure_ephemeris("DE440")
    sse.set.reset_mock()
    _ensure_ephemeris("DE440")
    sse.set.assert_not_called()
