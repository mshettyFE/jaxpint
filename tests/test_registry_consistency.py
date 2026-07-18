"""Integrity checks for the consolidated component registry.

``jaxpint.par.registry_table`` is now the single source of truth for the parser
tables, execution order, and PINT-name map.  Once those structures *derive* from
the registry, comparing them back to the registry is tautological, so this file
keeps only checks with an independent oracle:

- structural invariants (full enum coverage, no duplicates, classes resolve);
- the PINT-name map cross-checked against PINT's own component registry.

(The original migration-time tests that pinned the registry to the hand-written
tables did their job in git history; they are intentionally not kept here, since
post-migration they would only compare the registry to itself.)
"""

from __future__ import annotations

import pytest

from jaxpint.par import registry_table as R
from jaxpint.par._component_registry import registered
from jaxpint.par.registry import Component


def test_registry_covers_enum():
    # ``_registry()`` validates on first use (coverage + EXECUTION_ORDER sanity)
    # and returns a dict keyed by Component, so uniqueness is structural.
    assert set(R._registry()) == set(Component), "registry must cover the Component enum"


def test_binary_components_match():
    """The ``is_binary`` flags pick out exactly the binary components."""
    assert R.binary_components() == {
        Component.BINARY,
        Component.BINARY_BT_PIECEWISE,
    }


def test_registered_classes_resolve():
    """Every self-registered component carries at least one class with PARAMS."""
    for comp, rc in registered().items():
        assert rc.classes, f"{comp} maps to no classes"
        for cls in rc.classes:
            assert hasattr(cls, "PARAMS"), f"{cls.__name__} has no PARAMS"


def test_pint_names_are_real_pint_components():
    """Cross-check the PINT-name map against PINT's own component registry.

    This is the one map with an independent oracle: every ``pint_names`` entry
    must be a real PINT component class name, so a typo or stale name is caught
    here rather than only when a specific .par exercises the bridge.
    """
    timing_model = pytest.importorskip("pint.models.timing_model")
    known = set(timing_model.Component.component_types)
    registry_names = set(R.derive_pint_component_map())
    missing = registry_names - known
    assert not missing, f"registry pint_names unknown to PINT: {sorted(missing)}"
