"""Component auto-detection for the native ``.par`` parser.

Detection is driven by the components' own ``PARAMS`` declarations (aggregated
in :mod:`jaxpint.par.spec`): a parameter uniquely owned by exactly one
non-binary component activates it when present (this trigger relation is
*derived* from ownership, not declared).  ``SolarSystemShapiro`` is auto-added
with astrometry,
and the binary model comes from the ``BINARY`` line.  This produces the same
``(component_set, binary_model)`` the PINT bridge reads from ``model.components``.

Also exposes ``PINT_COMPONENT_MAP`` (PINT component class name -> JaxPINT
``Component`` enum, derived from :mod:`jaxpint.par.registry_table`), used only by
the PINT bridge.  PINT-free.
"""

from __future__ import annotations

import logging
from typing import Optional

from jaxpint.par import registry_table
from jaxpint.par import spec as S
from jaxpint.par.registry import BinaryModel, Component, binary_component_for

log = logging.getLogger(__name__)


_ASTROMETRY = {Component.ASTROMETRY_EQUATORIAL, Component.ASTROMETRY_ECLIPTIC}


def __getattr__(name: str):
    # PINT component class name -> JaxPINT Component enum.  Used only by the PINT
    # bridge (which reads PINT's own model.components); the native detector uses
    # the components' trigger declarations instead.  Derived from the registry's
    # ``pint_names`` (single source of truth).
    #
    # Lazy (via module ``__getattr__``): ``derive_pint_component_map`` triggers
    # the registry's lazy table assembly (importing the component modules), which
    # must not run during the ``jaxpint.par`` import cascade — deferring it to
    # first access avoids a self-registration import cycle.
    if name == "PINT_COMPONENT_MAP":
        return registry_table.derive_pint_component_map()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _detect_binary(
    templates: set[str], binary_value: Optional[str]
) -> tuple[Optional[BinaryModel], Optional[Component]]:
    """Resolve the binary model from the BINARY line (guessed fallback if absent)."""
    name = binary_value
    if name is None:
        if not (templates & S.BINARY_PARAMS):
            return None, None
        name = S.BINARY_PRIORITY[0]  # priority-ordered fallback (rare; no BINARY line)
        log.warning("No BINARY line but binary params present; guessing %r", name)

    binary_model, comp = binary_component_for(name)
    if binary_model is None:
        log.warning("Unknown binary model %r", name)
    return binary_model, comp


def detect_components(
    templates: set[str], binary_value: Optional[str] = None
) -> tuple[set[Component], Optional[BinaryModel]]:
    """Return ``(component_set, binary_model)`` for the present parameters.

    A present (template) parameter activates its component via
    ``spec.TRIGGER_MAP``; ``SolarSystemShapiro`` is auto-added with astrometry;
    the binary model comes from the ``BINARY`` line.
    """
    trigger_map = S.TRIGGER_MAP
    component_set: set[Component] = {
        trigger_map[t] for t in templates if t in trigger_map
    }

    if component_set & _ASTROMETRY:
        component_set.add(Component.SOLAR_SYSTEM_SHAPIRO)

    binary_model, binary_comp = _detect_binary(templates, binary_value)
    if binary_comp is not None:
        component_set.add(binary_comp)

    return component_set, binary_model
