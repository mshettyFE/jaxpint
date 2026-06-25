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
from typing import Optional

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
    # Execution order (mirrors PINT's DEFAULT_ORDER).  None => not ordered.
    order: Optional[int] = None
    # Feeds spec.BINARY_PARAMS; excluded from TRIGGER_MAP.
    is_binary: bool = False
    # The top-level/admin-params entry (TimingModel; paired with None, not the
    # enum, in the (class, owner) stream so its params never become triggers).
    top_level: bool = False


# ---------------------------------------------------------------------------
# The registry.  Pure metadata -- imports only the Component enum.
# ---------------------------------------------------------------------------

COMPONENTS: tuple[ComponentSpec, ...] = (
    # --- Delay components (PINT ordering) ---
    ComponentSpec(C.ASTROMETRY_EQUATORIAL, ("AstrometryEquatorial",), order=0),
    ComponentSpec(C.ASTROMETRY_ECLIPTIC, ("AstrometryEcliptic",), order=1),
    ComponentSpec(C.TROPOSPHERE_DELAY, ("TroposphereDelay",), order=2),
    ComponentSpec(C.SOLAR_SYSTEM_SHAPIRO, ("SolarSystemShapiro",), order=3),
    ComponentSpec(C.SOLAR_WIND_DISPERSION, ("SolarWindDispersion",), order=4),
    ComponentSpec(C.SOLAR_WIND_DISPERSION_X, ("SolarWindDispersionX",), order=5),
    ComponentSpec(C.DISPERSION_DM, ("DispersionDM",), order=6),
    ComponentSpec(C.DISPERSION_DMX, ("DispersionDMX",), order=7),
    ComponentSpec(C.DISPERSION_JUMP, ("DispersionJump",), order=8),
    ComponentSpec(C.BINARY, (), order=9, is_binary=True),
    ComponentSpec(C.BINARY_BT_PIECEWISE, (), order=10, is_binary=True),
    ComponentSpec(C.FREQUENCY_DEPENDENT, ("FD",), order=11),
    ComponentSpec(C.FD_JUMP, ("FDJump",), order=12),
    ComponentSpec(C.CHROMATIC_CM, ("ChromaticCM",), order=13),
    ComponentSpec(C.CHROMATIC_CMX, ("ChromaticCMX",), order=14),
    ComponentSpec(C.EXPONENTIAL_DIP, ("SimpleExponentialDip",), order=15),
    ComponentSpec(C.WAVE_X, ("WaveX",), order=16),
    ComponentSpec(C.DM_WAVE_X, ("DMWaveX",), order=17),
    ComponentSpec(C.CM_WAVE_X, ("CMWaveX",), order=18),
    # --- Phase components ---
    ComponentSpec(C.SPINDOWN, ("Spindown",), order=19),
    ComponentSpec(C.GLITCH, ("Glitch",), order=20),
    ComponentSpec(C.PIECEWISE_SPINDOWN, ("PiecewiseSpindown",), order=21),
    ComponentSpec(C.PHASE_JUMP, ("PhaseJump",), order=22),
    ComponentSpec(C.WAVE, ("Wave",), order=23),
    ComponentSpec(C.IFUNC, ("IFunc",), order=24),
    # --- Noise components ---
    ComponentSpec(C.SCALE_TOA_ERROR, ("ScaleToaError",), order=25),
    ComponentSpec(C.SCALE_DM_ERROR, ("ScaleDmError",), order=26),
    ComponentSpec(C.ECORR_NOISE, ("EcorrNoise",), order=27),
    ComponentSpec(C.PL_RED_NOISE, ("PLRedNoise",), order=28),
    ComponentSpec(C.PL_DM_NOISE, ("PLDMNoise",), order=29),
    ComponentSpec(C.PL_CHROM_NOISE, ("PLChromNoise",), order=30),
    ComponentSpec(C.PL_SW_NOISE, ("PLSWNoise",), order=31),
    # --- Unordered: detected/activated but not in DEFAULT_ORDER ---
    ComponentSpec(
        C.PHASE_OFFSET, ("PhaseOffset",)
    ),  # modeled as TimingModel.phoff_name
    ComponentSpec(C.NONE, top_level=True),  # admin/top-level params (TimingModel)
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
    orders = [s.order for s in COMPONENTS if s.order is not None]
    if len(orders) != len(set(orders)):
        raise ValueError("duplicate order values in COMPONENTS")


_validate()


# ---------------------------------------------------------------------------
# Lazy class resolution -- the ONLY part that imports component packages.
# ---------------------------------------------------------------------------


@functools.cache
def _param_classes() -> dict[Component, tuple[type, ...]]:
    """Component -> classes whose ``PARAMS`` feed the parser spec.

    Binary is many-to-one: every ``Binary*`` model contributes its PARAMS to
    ``Component.BINARY``.  ``TimingModel`` carries the top-level/admin params.
    Components without a dedicated class (``PHASE_OFFSET``) are absent here.
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
    from jaxpint.model import TimingModel

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
        C.NONE: (TimingModel,),  # top-level/admin params
    }


def derive_component_classes() -> list[tuple]:
    """``(class, owner)`` pairs feeding ``spec._tables()``.

    ``owner`` is the ``Component`` enum, except the ``top_level`` entry which is
    paired with ``None`` so its params never become triggers (matching the old
    ``(TimingModel, None)`` tuple).
    """
    classes = _param_classes()
    pairs: list[tuple] = []
    for s in COMPONENTS:
        owner = None if s.top_level else s.component
        for cls in classes.get(s.component, ()):
            pairs.append((cls, owner))
    return pairs


def derive_default_order() -> tuple[Component, ...]:
    """Ordered components, mirroring PINT's DEFAULT_ORDER"""
    ordered = sorted(
        (s for s in COMPONENTS if s.order is not None),
        key=lambda s: s.order if s.order is not None else 0,
    )
    return tuple(s.component for s in ordered)


def derive_pint_component_map() -> dict[str, Component]:
    """PINT class name -> Component."""
    return {name: s.component for s in COMPONENTS for name in s.pint_names}


def binary_components() -> frozenset[Component]:
    """Components flagged ``is_binary`` (feeds spec's binary handling)."""
    return frozenset(s.component for s in COMPONENTS if s.is_binary)
