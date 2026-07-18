"""Single source of truth for timing-model components.

A component is declared in one of two ways, which coexist:

- **Manual** (legacy): a :class:`ComponentSpec` in ``_MANUAL_COMPONENTS`` plus a
  ``_param_classes`` entry plus a ``_build_*`` in ``model_builder``.
- **Self-registered** (preferred): the component class carries
  ``@register_component`` / ``register_family`` (see
  :mod:`jaxpint.par._component_registry`); its spec / classes / builder are
  *derived* from that one declaration.

Migrating a component moves it from the first form to the second.

The full table (``COMPONENTS`` / ``COMPONENT_SPECS``) is assembled **lazily**
(:func:`_components`): self-registration requires importing the component
modules, so eager assembly during ``import`` could re-enter a component
mid-import (a cycle).  Deferring to first use — after the component packages
finish importing — avoids that.  ``COMPONENTS`` / ``COMPONENT_SPECS`` are exposed
as module attributes via ``__getattr__`` for backward compatibility.

``EXECUTION_ORDER`` / ``PRIORITY`` are the exception: they name only the
``Component`` enum (never the classes), so they stay eager, import-light module
constants that ``model_builder`` can read directly to order the delay chain
without forcing table assembly.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from jaxpint.par.registry import Component

C = Component


@dataclass(frozen=True)
class ComponentSpec:
    """Declarative metadata for one timing-model component.

    Today the table drives the parser/order/PINT-name derivations.  A future
    phase may attach a per-component ``build`` callable here to also derive the
    model-builder dispatch (see the plan); that is intentionally not done yet.
    """

    component: Component
    # PINT component class names mapping to this Component (>=0; "FD" and
    # "SimpleExponentialDip" differ from Component.value; binary has none).
    pint_names: tuple[str, ...] = ()
    # Feeds spec.BINARY_PARAMS; excluded from TRIGGER_MAP.
    is_binary: bool = False


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
# Manual registry.  Pure metadata -- imports only the Component enum.
# Self-registered components are NOT listed here; they are merged in by
# :func:`_components` from the class decorators.
# ---------------------------------------------------------------------------

_MANUAL_COMPONENTS: tuple[ComponentSpec, ...] = (
    # --- Delay components (PINT ordering) ---
    ComponentSpec(C.ASTROMETRY_EQUATORIAL, ("AstrometryEquatorial",)),
    ComponentSpec(C.ASTROMETRY_ECLIPTIC, ("AstrometryEcliptic",)),
    ComponentSpec(C.TROPOSPHERE_DELAY, ("TroposphereDelay",)),
    ComponentSpec(C.SOLAR_SYSTEM_SHAPIRO, ("SolarSystemShapiro",)),
    ComponentSpec(C.SOLAR_WIND_DISPERSION, ("SolarWindDispersion",)),
    ComponentSpec(C.SOLAR_WIND_DISPERSION_X, ("SolarWindDispersionX",)),
    ComponentSpec(C.DISPERSION_DM, ("DispersionDM",)),
    # DISPERSION_DMX is self-registered (jaxpint/delay/dispersion_dmx.py).
    ComponentSpec(C.DISPERSION_JUMP, ("DispersionJump",)),
    # BINARY and BINARY_BT_PIECEWISE are self-registered as a family
    # (jaxpint/binary/_build.py).
    ComponentSpec(C.FREQUENCY_DEPENDENT, ("FD",)),
    ComponentSpec(C.FD_JUMP, ("FDJump",)),
    ComponentSpec(C.CHROMATIC_CM, ("ChromaticCM",)),
    ComponentSpec(C.CHROMATIC_CMX, ("ChromaticCMX",)),
    ComponentSpec(C.EXPONENTIAL_DIP, ("SimpleExponentialDip",)),
    ComponentSpec(C.WAVE_X, ("WaveX",)),
    ComponentSpec(C.DM_WAVE_X, ("DMWaveX",)),
    ComponentSpec(C.CM_WAVE_X, ("CMWaveX",)),
    # --- Phase components ---
    # SPINDOWN is self-registered (jaxpint/phase/spin.py).
    ComponentSpec(C.GLITCH, ("Glitch",)),
    ComponentSpec(C.PIECEWISE_SPINDOWN, ("PiecewiseSpindown",)),
    ComponentSpec(C.PHASE_JUMP, ("PhaseJump",)),
    ComponentSpec(C.WAVE, ("Wave",)),
    ComponentSpec(C.IFUNC, ("IFunc",)),
    # --- Noise components ---
    ComponentSpec(C.SCALE_TOA_ERROR, ("ScaleToaError",)),
    ComponentSpec(C.SCALE_DM_ERROR, ("ScaleDmError",)),
    ComponentSpec(C.ECORR_NOISE, ("EcorrNoise",)),
    # PL_RED_NOISE is self-registered (jaxpint/noise/red_noise.py).
    ComponentSpec(C.PL_DM_NOISE, ("PLDMNoise",)),
    ComponentSpec(C.PL_CHROM_NOISE, ("PLChromNoise",)),
    ComponentSpec(C.PL_SW_NOISE, ("PLSWNoise",)),
)


def _validate(comps: tuple[ComponentSpec, ...]) -> None:
    """Sanity (on assembly): unique components, full enum coverage, sane order."""
    seen = [s.component for s in comps]
    if len(seen) != len(set(seen)):
        dupes = {c for c in seen if seen.count(c) > 1}
        raise ValueError(f"duplicate ComponentSpec entries: {dupes}")
    missing = set(Component) - set(seen)
    if missing:
        raise ValueError(f"COMPONENTS does not cover the Component enum: {missing}")
    # EXECUTION_ORDER is the single source of ordering: position is the order, so
    # it may contain no duplicates and may reference only known components.
    if len(EXECUTION_ORDER) != len(set(EXECUTION_ORDER)):
        raise ValueError("duplicate entries in EXECUTION_ORDER")
    unknown = set(EXECUTION_ORDER) - set(seen)
    if unknown:
        raise ValueError(f"EXECUTION_ORDER references unknown components: {unknown}")


@functools.cache
def _components() -> tuple[ComponentSpec, ...]:
    """The full component table: manual entries + self-registered (derived).

    Lazy + cached: the first call imports the registry, whose contents come from
    importing the (migrated) component modules, so it must run *after* they
    finish importing — never at module import (that could re-enter a component
    mid-import).  See the module docstring.
    """
    from jaxpint.par._component_registry import registered

    derived = tuple(
        ComponentSpec(rc.component, rc.pint_names, is_binary=rc.is_binary)
        for rc in registered().values()
    )
    comps = _MANUAL_COMPONENTS + derived
    _validate(comps)
    return comps


@functools.cache
def _component_specs() -> dict[Component, ComponentSpec]:
    """``Component -> ComponentSpec`` view of the assembled table."""
    return {s.component: s for s in _components()}


def __getattr__(name: str):
    # Expose COMPONENTS / COMPONENT_SPECS lazily (assembled on first access) so
    # existing ``registry_table.COMPONENTS`` callers keep working.
    if name == "COMPONENTS":
        return _components()
    if name == "COMPONENT_SPECS":
        return _component_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Lazy class resolution -- the ONLY part that imports component packages.
# ---------------------------------------------------------------------------


@functools.cache
def _param_classes() -> dict[Component, tuple[type, ...]]:
    """Component -> classes whose ``PARAMS`` feed the parser spec.

    Binary is many-to-one: every ``Binary*`` model contributes its PARAMS to
    ``Component.BINARY``.  The top-level/admin params (``TimingModel``) are paired
    with ``None`` directly in :func:`derive_component_classes`, not here.

    Self-registered components contribute their class(es) via the registry merge
    at the end (so migrating a component out of the manual dict is a no-op here).
    """
    from jaxpint.phase.glitch import Glitch
    from jaxpint.phase.wave import Wave
    from jaxpint.phase.jump import PhaseJump
    from jaxpint.phase.piecewise_spindown import PiecewiseSpindown
    from jaxpint.phase.ifunc import IFunc
    from jaxpint.delay.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.delay.dispersion_dm import DispersionDM
    from jaxpint.delay.dispersion_jump import DispersionJump
    from jaxpint.delay.shapiro import SolarSystemShapiroDelay
    from jaxpint.delay.solar_wind import SolarWindDispersion
    from jaxpint.delay.solar_wind_x import SolarWindDispersionX
    from jaxpint.delay.troposphere import TroposphereDelay
    from jaxpint.delay.chromatic_cm import ChromaticCM
    from jaxpint.delay.chromatic_cmx import ChromaticCMX
    from jaxpint.delay.cmwavex import CMWaveX
    from jaxpint.delay.wavex import WaveX
    from jaxpint.delay.dmwavex import DMWaveX
    from jaxpint.delay.frequency_dependent import FrequencyDependent
    from jaxpint.delay.fdjump import FDJump
    from jaxpint.delay.exponential_dip import ExponentialDip
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.dm_white import ScaleDmError
    from jaxpint.noise.ecorr import EcorrNoise
    from jaxpint.noise.dm_noise import PLDMNoise
    from jaxpint.noise.chrom_noise import PLChromNoise
    from jaxpint.noise.sw_noise import PLSWNoise

    from jaxpint.par._component_registry import registered

    manual = {
        C.GLITCH: (Glitch,),
        C.WAVE: (Wave,),
        C.PHASE_JUMP: (PhaseJump,),
        C.PIECEWISE_SPINDOWN: (PiecewiseSpindown,),
        C.IFUNC: (IFunc,),
        C.ASTROMETRY_EQUATORIAL: (AstrometryEquatorial,),
        C.ASTROMETRY_ECLIPTIC: (AstrometryEcliptic,),
        C.DISPERSION_DM: (DispersionDM,),
        C.DISPERSION_JUMP: (DispersionJump,),
        C.SOLAR_SYSTEM_SHAPIRO: (SolarSystemShapiroDelay,),
        C.SOLAR_WIND_DISPERSION: (SolarWindDispersion,),
        C.SOLAR_WIND_DISPERSION_X: (SolarWindDispersionX,),
        C.TROPOSPHERE_DELAY: (TroposphereDelay,),
        C.CHROMATIC_CM: (ChromaticCM,),
        C.CHROMATIC_CMX: (ChromaticCMX,),
        C.CM_WAVE_X: (CMWaveX,),
        C.WAVE_X: (WaveX,),
        C.DM_WAVE_X: (DMWaveX,),
        C.FREQUENCY_DEPENDENT: (FrequencyDependent,),
        C.FD_JUMP: (FDJump,),
        C.EXPONENTIAL_DIP: (ExponentialDip,),
        C.SCALE_TOA_ERROR: (ScaleToaError,),
        C.SCALE_DM_ERROR: (ScaleDmError,),
        C.ECORR_NOISE: (EcorrNoise,),
        C.PL_DM_NOISE: (PLDMNoise,),
        C.PL_CHROM_NOISE: (PLChromNoise,),
        C.PL_SW_NOISE: (PLSWNoise,),
    }
    # Self-registered components contribute their class(es) here too.
    manual.update({rc.component: rc.classes for rc in registered().values()})
    return manual


def derive_component_classes() -> list[tuple]:
    """``(class, owner)`` pairs feeding ``spec._tables()``.

    ``owner`` is the ``Component`` enum, except ``TimingModel`` (the top-level /
    admin params) which is paired with ``None`` so its params never become
    triggers.
    """
    from jaxpint.model import TimingModel

    classes = _param_classes()
    pairs: list[tuple] = [(TimingModel, None)]  # top-level/admin params
    for s in _components():
        for cls in classes.get(s.component, ()):
            pairs.append((cls, s.component))
    return pairs


def derive_pint_component_map() -> dict[str, Component]:
    """PINT class name -> Component."""
    return {name: s.component for s in _components() for name in s.pint_names}


def binary_components() -> frozenset[Component]:
    """Components flagged ``is_binary`` (feeds spec's binary handling)."""
    return frozenset(s.component for s in _components() if s.is_binary)
