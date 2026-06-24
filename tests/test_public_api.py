"""Guard the top-level public API declaration (`jaxpint.__all__`).

`__all__` is hand-maintained alongside the re-export imports in
``jaxpint/__init__.py``, so it can drift: a name gets listed twice, or added to
``__all__`` without a matching import (or vice versa). These tests catch both.
"""

import importlib

import pytest


def test_all_has_no_duplicates():
    jaxpint = importlib.import_module("jaxpint")
    dupes = sorted({n for n in jaxpint.__all__ if jaxpint.__all__.count(n) > 1})
    assert not dupes, f"duplicate entries in jaxpint.__all__: {dupes}"


def test_all_names_resolve():
    # Five PINT-backed names resolve lazily via the module __getattr__, which
    # imports jaxpint.bridge on demand -- so the resolvability check needs PINT.
    pytest.importorskip("pint")
    jaxpint = importlib.import_module("jaxpint")
    missing = [name for name in jaxpint.__all__ if not hasattr(jaxpint, name)]
    assert not missing, f"names in jaxpint.__all__ that do not resolve: {missing}"
