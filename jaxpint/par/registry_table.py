"""Single source of truth for timing-model components.

Every component is declared once here as a :class:`ComponentSpec`.

Adding a component therefore means adding one ``ComponentSpec`` entry (plus the
component class itself).

The *metadata* (identity / order / PINT names / flags) is a
plain module-level tuple that imports only the ``Component`` enum, so deriving
the order and PINT-name maps stays import-light.

The component *classes* are resolved lazily in :func:`_param_classes`
(function-local imports), so importing this module never forces the heavy
component packages and cannot create a cycle.
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


# ---------------------------------------------------------------------------
# The registry.  Pure metadata -- imports only the Component enum.
# ---------------------------------------------------------------------------

COMPONENTS: tuple[ComponentSpec, ...] = (
    # --- Delay components (PINT ordering) ---
    ComponentSpec(C.ASTROMETRY_EQUATORIAL, ("AstrometryEquatorial",)),
    ComponentSpec(C.ASTROMETRY_ECLIPTIC, ("AstrometryEcliptic",)),
    ComponentSpec(C.TROPOSPHERE_DELAY, ("TroposphereDelay",)),
    ComponentSpec(C.SOLAR_SYSTEM_SHAPIRO, ("SolarSystemShapiro",)),
    ComponentSpec(C.SOLAR_WIND_DISPERSION, ("SolarWindDispersion",)),
    ComponentSpec(C.SOLAR_WIND_DISPERSION_X, ("SolarWindDispersionX",)),
    ComponentSpec(C.DISPERSION_DM, ("DispersionDM",)),
    ComponentSpec(C.DISPERSION_DMX, ("DispersionDMX",)),
    ComponentSpec(C.DISPERSION_JUMP, ("DispersionJump",)),
    ComponentSpec(C.BINARY, (), is_binary=True),
    ComponentSpec(C.BINARY_BT_PIECEWISE, (), is_binary=True),
    ComponentSpec(C.FREQUENCY_DEPENDENT, ("FD",)),
    ComponentSpec(C.FD_JUMP, ("FDJump",)),
    ComponentSpec(C.CHROMATIC_CM, ("ChromaticCM",)),
    ComponentSpec(C.CHROMATIC_CMX, ("ChromaticCMX",)),
    ComponentSpec(C.EXPONENTIAL_DIP, ("SimpleExponentialDip",)),
    ComponentSpec(C.WAVE_X, ("WaveX",)),
    ComponentSpec(C.DM_WAVE_X, ("DMWaveX",)),
    ComponentSpec(C.CM_WAVE_X, ("CMWaveX",)),
    # --- Phase components ---
    ComponentSpec(C.SPINDOWN, ("Spindown",)),
    ComponentSpec(C.GLITCH, ("Glitch",)),
    ComponentSpec(C.PIECEWISE_SPINDOWN, ("PiecewiseSpindown",)),
    ComponentSpec(C.PHASE_JUMP, ("PhaseJump",)),
    ComponentSpec(C.WAVE, ("Wave",)),
    ComponentSpec(C.IFUNC, ("IFunc",)),
    # --- Noise components ---
    ComponentSpec(C.SCALE_TOA_ERROR, ("ScaleToaError",)),
    ComponentSpec(C.SCALE_DM_ERROR, ("ScaleDmError",)),
    ComponentSpec(C.ECORR_NOISE, ("EcorrNoise",)),
    ComponentSpec(C.PL_RED_NOISE, ("PLRedNoise",)),
    ComponentSpec(C.PL_DM_NOISE, ("PLDMNoise",)),
    ComponentSpec(C.PL_CHROM_NOISE, ("PLChromNoise",)),
    ComponentSpec(C.PL_SW_NOISE, ("PLSWNoise",)),
)

COMPONENT_SPECS: dict[Component, ComponentSpec] = {s.component: s for s in COMPONENTS}


def _validate() -> None:
    """Import-time sanity: unique components, full enum coverage, sane order."""
    seen = [s.component for s in COMPONENTS]
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


_validate()


# ---------------------------------------------------------------------------
# Lazy class resolution -- the ONLY part that imports component packages.
# ---------------------------------------------------------------------------


@functools.cache
def _param_classes() -> dict[Component, tuple[type, ...]]:
    """Component -> classes whose ``PARAMS`` feed the parser spec.

    Binary is many-to-one: every ``Binary*`` model contributes its PARAMS to
    ``Component.BINARY``.  The top-level/admin params (``TimingModel``) are paired
    with ``None`` directly in :func:`derive_component_classes`, not here.
    """
    from jaxpint.phase.spin import Spindown
    from jaxpint.phase.glitch import Glitch
    from jaxpint.phase.wave import Wave
    from jaxpint.phase.jump import PhaseJump
    from jaxpint.phase.piecewise_spindown import PiecewiseSpindown
    from jaxpint.phase.ifunc import IFunc
    from jaxpint.delay.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.delay.dispersion_dm import DispersionDM
    from jaxpint.delay.dispersion_dmx import DispersionDMX
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
    from jaxpint.binary.bt import BinaryBT
    from jaxpint.binary.bt_piecewise import BinaryBTPiecewise
    from jaxpint.binary.dd import BinaryDD
    from jaxpint.binary.ddk import BinaryDDK
    from jaxpint.binary.ddgr import BinaryDDGR
    from jaxpint.binary.ell1 import BinaryELL1
    from jaxpint.noise.white import ScaleToaError
    from jaxpint.noise.dm_white import ScaleDmError
    from jaxpint.noise.ecorr import EcorrNoise
    from jaxpint.noise.red_noise import PLRedNoise
    from jaxpint.noise.dm_noise import PLDMNoise
    from jaxpint.noise.chrom_noise import PLChromNoise
    from jaxpint.noise.sw_noise import PLSWNoise

    return {
        C.SPINDOWN: (Spindown,),
        C.GLITCH: (Glitch,),
        C.WAVE: (Wave,),
        C.PHASE_JUMP: (PhaseJump,),
        C.PIECEWISE_SPINDOWN: (PiecewiseSpindown,),
        C.IFUNC: (IFunc,),
        C.ASTROMETRY_EQUATORIAL: (AstrometryEquatorial,),
        C.ASTROMETRY_ECLIPTIC: (AstrometryEcliptic,),
        C.DISPERSION_DM: (DispersionDM,),
        C.DISPERSION_DMX: (DispersionDMX,),
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
        C.BINARY: (BinaryBT, BinaryDD, BinaryDDK, BinaryDDGR, BinaryELL1),
        C.BINARY_BT_PIECEWISE: (BinaryBTPiecewise,),
        C.SCALE_TOA_ERROR: (ScaleToaError,),
        C.SCALE_DM_ERROR: (ScaleDmError,),
        C.ECORR_NOISE: (EcorrNoise,),
        C.PL_RED_NOISE: (PLRedNoise,),
        C.PL_DM_NOISE: (PLDMNoise,),
        C.PL_CHROM_NOISE: (PLChromNoise,),
        C.PL_SW_NOISE: (PLSWNoise,),
    }


def derive_component_classes() -> list[tuple]:
    """``(class, owner)`` pairs feeding ``spec._tables()``.

    ``owner`` is the ``Component`` enum, except ``TimingModel`` (the top-level /
    admin params) which is paired with ``None`` so its params never become
    triggers.
    """
    from jaxpint.model import TimingModel

    classes = _param_classes()
    pairs: list[tuple] = [(TimingModel, None)]  # top-level/admin params
    for s in COMPONENTS:
        for cls in classes.get(s.component, ()):
            pairs.append((cls, s.component))
    return pairs


def derive_default_order() -> tuple[Component, ...]:
    """Ordered components, mirroring PINT's DEFAULT_ORDER"""
    return EXECUTION_ORDER


def derive_pint_component_map() -> dict[str, Component]:
    """PINT class name -> Component."""
    return {name: s.component for s in COMPONENTS for name in s.pint_names}


def binary_components() -> frozenset[Component]:
    """Components flagged ``is_binary`` (feeds spec's binary handling)."""
    return frozenset(s.component for s in COMPONENTS if s.is_binary)
