"""PINT-free unit tests for observatory resolution (jaxpint.clock.observatory)."""

from __future__ import annotations

import pytest

from jaxpint.clock.observatory import UnknownObservatory, resolve_observatory


def test_resolve_by_name():
    cfg = resolve_observatory("gbt")
    assert cfg.canonical == "gbt"
    assert cfg.clock_files == ("time_gbt.dat",)
    assert cfg.apply_gps2utc and cfg.timescale == "utc"


def test_resolve_itoa_code_casefold():
    for tok in ("ao", "AO", "Ao"):
        cfg = resolve_observatory(tok)
        assert cfg.canonical == "arecibo"
        assert cfg.clock_files == ("time_ao.dat",)


def test_resolve_tempo_code():
    assert resolve_observatory("1").canonical == "gbt"
    assert resolve_observatory("3").canonical == "arecibo"


def test_resolve_chime_empty_clockfile():
    cfg = resolve_observatory("chime")
    assert cfg.canonical == "chime"
    assert cfg.clock_files == ()
    assert cfg.apply_gps2utc is True
    assert cfg.timescale == "utc"


def test_resolve_barycenter():
    cfg = resolve_observatory("@")
    assert cfg.canonical == "barycenter"
    assert cfg.clock_files == ()
    assert cfg.apply_gps2utc is False
    assert cfg.timescale == "tdb"


def test_unknown_token():
    with pytest.raises(UnknownObservatory):
        resolve_observatory("definitely_not_an_observatory")
    with pytest.raises(UnknownObservatory):
        resolve_observatory("")
