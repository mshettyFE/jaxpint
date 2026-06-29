"""Parameter spec: aggregated from each component's ``PARAMS`` declaration.

The native ``.par`` parser's vocabulary is owned by JaxPINT's own components.
Every component class carries a class-level ``PARAMS: tuple[ParamDecl, ...]``
(see :class:`jaxpint.components.ParamDecl`); this module aggregates them into
the tables the parser consumes:

- ``PARAM_SPEC``     -- canonical name -> {kind, unit, scale?, frozen_default?, ...}
- ``ALIAS_MAP``      -- alias -> canonical name
- ``PREFIX_MAP``     -- prefix (incl. prefix aliases) -> template name
- ``CANONICAL_PREFIX``-- template name -> its canonical prefix
- ``KNOWN_PARAMS``   -- every declared canonical parameter name
- ``TRIGGER_MAP``    -- param -> Component it activates (derived: uniquely-owned params)
- ``spec_for(name)`` -- spec for a known name (default float if undeclared-but-known)

Built lazily and cached on first access: the ``(class, owner)`` pairs come from
``jaxpint.par.registry_table``, which resolves the component classes with
function-local imports.  So this is robust to import order, ``jaxpint.par`` never
imports PINT, and there is no cycle (this module imports no component class;
components import ``ParamDecl`` from :mod:`jaxpint.components` and never import
this module).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NotRequired, Optional, TypedDict

from jaxpint.par import registry_table
from jaxpint.par.registry import Component

if TYPE_CHECKING:
    from jaxpint.components import ParamDecl

__all__ = [
    "PARAM_SPEC",
    "ALIAS_MAP",
    "PREFIX_MAP",
    "CANONICAL_PREFIX",
    "KNOWN_PARAMS",
    "TRIGGER_MAP",
    "BINARY_PARAMS",
    "BINARY_PRIORITY",
    "spec_for",
]


class ParamSpec(TypedDict):
    """Per-parameter spec aggregated from a :class:`ParamDecl`.

    Fixed schema: ``kind``/``unit`` are always present; the rest appear only
    when the declaration sets them.
    """

    kind: str
    unit: str
    scale: NotRequired[float]
    scale_threshold: NotRequired[float]
    frozen_default: NotRequired[bool]
    is_prefix: NotRequired[bool]
    prefix: NotRequired[str]
    prefix_aliases: NotRequired[tuple[str, ...]]


_DEFAULT_FLOAT: ParamSpec = {"kind": "float", "unit": ""}

# guess_binary_model priority when a .par has binary params but no BINARY line.
BINARY_PRIORITY: tuple[str, ...] = (
    "BT",
    "BT_piecewise",
    "ELL1",
    "ELL1H",
    "ELL1k",
    "DD",
    "DDK",
    "DDGR",
    "DDS",
    "DDH",
)


def _spec_of(decl: ParamDecl) -> ParamSpec:
    spec: ParamSpec = {"kind": decl.kind, "unit": decl.unit}
    if decl.scale is not None:
        spec["scale"] = decl.scale
        if decl.scale_threshold is not None:
            spec["scale_threshold"] = decl.scale_threshold
    if decl.frozen_default is False:
        spec["frozen_default"] = False
    if decl.prefix is not None:
        spec["is_prefix"] = True
        spec["prefix"] = decl.prefix
        if decl.prefix_aliases:
            spec["prefix_aliases"] = tuple(decl.prefix_aliases)
    return spec


@functools.cache
def _tables() -> dict:
    """Aggregate the registry's ``(class, owner)`` pairs into the parser tables."""
    param_spec: dict[str, ParamSpec] = {}
    alias_map: dict[str, str] = {}
    prefix_map: dict[str, str] = {}
    canonical_prefix: dict[str, str] = {}
    owners: dict[str, set] = {}  # param -> set of owning Component enums
    known: set[str] = set()
    binary_params: set[str] = set()

    _binary = registry_table.binary_components()
    for cls, comp in registry_table.derive_component_classes():
        if not cls.PARAMS:
            raise TypeError(f"{cls.__name__} declares no PARAMS")
        for decl in cls.PARAMS:
            if comp in _binary:
                binary_params.add(decl.name)
            owners.setdefault(decl.name, set()).add(comp)
            spec = _spec_of(decl)
            if decl.name in param_spec and param_spec[decl.name] != spec:
                raise ValueError(
                    f"Inconsistent ParamDecl for {decl.name!r}: "
                    f"{param_spec[decl.name]} vs {spec}"
                )
            param_spec[decl.name] = spec
            known.add(decl.name)
            for a in decl.aliases:
                alias_map.setdefault(a, decl.name)
            if decl.prefix is not None:
                prefix_map.setdefault(decl.prefix, decl.name)
                canonical_prefix[decl.name] = decl.prefix
                for pa in decl.prefix_aliases:
                    prefix_map.setdefault(pa, decl.name)

    # Triggers: a param activates its component when it is uniquely owned by
    # exactly one non-binary component (PINT's own rule).  Binary models are
    # selected from the BINARY line, and top-level/admin params (owner None)
    # never trigger.
    trigger_map: dict[str, Component] = {}
    for name, comps in owners.items():
        if len(comps) != 1:
            continue
        (comp,) = comps
        if comp is None or comp in _binary:
            continue
        trigger_map[name] = comp

    # PHOFF is modeled as a TimingModel field (phoff_name), not a dedicated
    # component class, but its presence must still activate PHASE_OFFSET.
    if "PHOFF" in known:
        trigger_map["PHOFF"] = Component.PHASE_OFFSET

    return {
        "PARAM_SPEC": param_spec,
        "ALIAS_MAP": alias_map,
        "PREFIX_MAP": prefix_map,
        "CANONICAL_PREFIX": canonical_prefix,
        "KNOWN_PARAMS": frozenset(known),
        "TRIGGER_MAP": trigger_map,
        "BINARY_PARAMS": frozenset(binary_params),
    }


def spec_for(name: str) -> Optional[ParamSpec]:
    """Spec for a canonical name, or ``None`` if unknown.

    Declared params have an explicit spec; any other declared-but-plain name
    resolves to a default float (its unit is documentation the runtime ignores).
    """
    t = _tables()
    s = t["PARAM_SPEC"].get(name)
    if s is not None:
        return s
    if name in t["KNOWN_PARAMS"]:
        return _DEFAULT_FLOAT
    return None


def __getattr__(name: str):
    t = _tables()
    if name in t:
        return t[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
