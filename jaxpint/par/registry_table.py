"""Parser/bridge derivations over the component self-registration registry.

Every component self-registers: its class carries ``@register_component`` /
``register_family`` (see :mod:`jaxpint.par._component_registry`), which records
its identity, PINT names, classes, binary flag, and builder in one place.  This
module holds the small ``derive_*`` functions that project that registry into
the shapes the parser (:mod:`jaxpint.par.spec`) and PINT bridge
(:mod:`jaxpint.par.components`) need — plus the execution-order constants.

The derivations are **lazy** (via :func:`_registry`): self-registration requires
importing the component modules, so reading the registry eagerly during
``import`` could re-enter a component mid-import (a cycle).  Deferring to first
use — after the component packages finish importing — avoids that.  A consequence
is that :func:`_validate` runs on first derivation (e.g. first parse), not at
bare ``import``.

``EXECUTION_ORDER`` / ``PRIORITY`` are the exception: they name only the
``Component`` enum (never the classes), so they stay eager, import-light module
constants that ``model_builder`` can read directly to order the delay chain
without forcing the registry to assemble.
"""

from __future__ import annotations

import functools

from jaxpint.par.registry import Component

C = Component


# ---------------------------------------------------------------------------
# Execution order.  A global arrangement (how delays chain), not a per-component
# fact: position in this tuple *is* the order, mirroring PINT's DEFAULT_ORDER.
# Import-light (only the Component enum).  Detected-but-unordered components are
# simply absent (phases are summed, so their relative order is irrelevant).
# ---------------------------------------------------------------------------

EXECUTION_ORDER: tuple[Component, ...] = (
    # --- Delay components (PINT ordering) ---
    C.ASTROMETRY_EQUATORIAL,
    C.ASTROMETRY_ECLIPTIC,
    C.TROPOSPHERE_DELAY,
    C.SOLAR_SYSTEM_SHAPIRO,
    C.SOLAR_WIND_DISPERSION,
    C.SOLAR_WIND_DISPERSION_X,
    C.DISPERSION_DM,
    C.DISPERSION_DMX,
    C.DISPERSION_JUMP,
    C.BINARY,
    C.BINARY_BT_PIECEWISE,
    C.FREQUENCY_DEPENDENT,
    C.FD_JUMP,
    C.CHROMATIC_CM,
    C.CHROMATIC_CMX,
    C.EXPONENTIAL_DIP,
    C.WAVE_X,
    C.DM_WAVE_X,
    C.CM_WAVE_X,
    # --- Phase components ---
    C.SPINDOWN,
    C.GLITCH,
    C.PIECEWISE_SPINDOWN,
    C.PHASE_JUMP,
    C.WAVE,
    C.IFUNC,
    # --- Noise components ---
    C.SCALE_TOA_ERROR,
    C.SCALE_DM_ERROR,
    C.ECORR_NOISE,
    C.PL_RED_NOISE,
    C.PL_DM_NOISE,
    C.PL_CHROM_NOISE,
    C.PL_SW_NOISE,
)

# Component -> its position in EXECUTION_ORDER.  Used by ``build_model`` to order
# the delay chain; components absent from EXECUTION_ORDER sort to the end.
# Import-light (only the enum), so it is an eager module constant.
PRIORITY: dict[Component, int] = {comp: i for i, comp in enumerate(EXECUTION_ORDER)}


# ---------------------------------------------------------------------------
# Lazy registry access -- the ONLY part that reaches the component packages.
# ---------------------------------------------------------------------------


def _validate(registry: dict) -> None:
    """Sanity (on first registry use): full enum coverage + well-formed order.

    The registry is keyed by ``Component`` (so duplicate components are
    impossible); what remains to check is that every enum member registered and
    that ``EXECUTION_ORDER`` — the single source of ordering — has no duplicates
    and references only registered components.
    """
    seen = set(registry)
    missing = set(Component) - seen
    if missing:
        raise ValueError(f"registry does not cover the Component enum: {missing}")
    if len(EXECUTION_ORDER) != len(set(EXECUTION_ORDER)):
        raise ValueError("duplicate entries in EXECUTION_ORDER")
    unknown = set(EXECUTION_ORDER) - seen
    if unknown:
        raise ValueError(f"EXECUTION_ORDER references unknown components: {unknown}")


@functools.cache
def _registry() -> dict:
    """Validated snapshot of the registry (``Component -> RegisteredComponent``).

    Lazy + cached: the first call reads the registry, whose contents come from
    importing the component modules, so it must run *after* they finish importing
    — never at module import (that could re-enter a component mid-import).  See the
    module docstring.
    """
    from jaxpint.par._component_registry import registered

    registry = registered()
    _validate(registry)
    return registry


def derive_component_classes() -> list[tuple]:
    """``(class, owner)`` pairs feeding ``spec._tables()``.

    ``owner`` is the ``Component`` a class's params belong to, except
    ``TimingModel`` (the top-level / admin params) which is paired with ``None``
    so its params never become detection triggers.  Binary is many-to-one: every
    ``Binary*`` class contributes its PARAMS to ``Component.BINARY``.
    """
    from jaxpint.model import TimingModel

    pairs: list[tuple] = [(TimingModel, None)]  # top-level/admin params
    for rc in _registry().values():
        for cls in rc.classes:
            pairs.append((cls, rc.component))
    return pairs


def derive_pint_component_map() -> dict[str, Component]:
    """PINT class name -> Component."""
    return {name: rc.component for rc in _registry().values() for name in rc.pint_names}


def binary_components() -> frozenset[Component]:
    """Components flagged ``is_binary`` (feeds spec's binary handling)."""
    return frozenset(rc.component for rc in _registry().values() if rc.is_binary)
