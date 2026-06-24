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
- ``TRIGGER_MAP``    -- param -> Component it activates (from ``triggers=True``)
- ``spec_for(name)`` -- spec for a known name (default float if undeclared-but-known)

Built lazily on first access (importing the component classes only then), so it
is robust to import order and ``jaxpint.par`` never imports PINT.  Components
import ``ParamDecl`` from :mod:`jaxpint.components` and never import this module,
so there is no cycle.
"""

from __future__ import annotations

import functools
from typing import Optional

from jaxpint.par.registry import Component

__all__ = [
    "PARAM_SPEC", "ALIAS_MAP", "PREFIX_MAP", "CANONICAL_PREFIX",
    "KNOWN_PARAMS", "TRIGGER_MAP", "BINARY_PARAMS", "BINARY_PRIORITY", "spec_for",
]

_DEFAULT_FLOAT: dict = {"kind": "float", "unit": ""}

# guess_binary_model priority when a .par has binary params but no BINARY line.
BINARY_PRIORITY: tuple[str, ...] = (
    "BT", "BT_piecewise", "ELL1", "ELL1H", "ELL1k", "DD", "DDK", "DDGR", "DDS", "DDH",
)


def _component_classes():
    """(class, Component) for every component that declares ``PARAMS``.

    Imported lazily so this module never forces a heavy import at load time and
    cannot create a cycle with the component packages.
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

    C = Component
    return [
        (Spindown, C.SPINDOWN), (Glitch, C.GLITCH), (Wave, C.WAVE),
        (PhaseJump, C.PHASE_JUMP), (PiecewiseSpindown, C.PIECEWISE_SPINDOWN),
        (IFunc, C.IFUNC),
        (AstrometryEquatorial, C.ASTROMETRY_EQUATORIAL),
        (AstrometryEcliptic, C.ASTROMETRY_ECLIPTIC),
        (DispersionDM, C.DISPERSION_DM), (DispersionDMX, C.DISPERSION_DMX),
        (DispersionJump, C.DISPERSION_JUMP),
        (SolarSystemShapiroDelay, C.SOLAR_SYSTEM_SHAPIRO),
        (SolarWindDispersion, C.SOLAR_WIND_DISPERSION),
        (SolarWindDispersionX, C.SOLAR_WIND_DISPERSION_X),
        (TroposphereDelay, C.TROPOSPHERE_DELAY),
        (ChromaticCM, C.CHROMATIC_CM), (ChromaticCMX, C.CHROMATIC_CMX),
        (CMWaveX, C.CM_WAVE_X), (WaveX, C.WAVE_X), (DMWaveX, C.DM_WAVE_X),
        (FrequencyDependent, C.FREQUENCY_DEPENDENT), (FDJump, C.FD_JUMP),
        (ExponentialDip, C.EXPONENTIAL_DIP),
        (BinaryBT, C.BINARY), (BinaryBTPiecewise, C.BINARY_BT_PIECEWISE),
        (BinaryDD, C.BINARY), (BinaryDDK, C.BINARY), (BinaryDDGR, C.BINARY),
        (BinaryELL1, C.BINARY),
        (ScaleToaError, C.SCALE_TOA_ERROR), (ScaleDmError, C.SCALE_DM_ERROR),
        (EcorrNoise, C.ECORR_NOISE), (PLRedNoise, C.PL_RED_NOISE),
        (PLDMNoise, C.PL_DM_NOISE), (PLChromNoise, C.PL_CHROM_NOISE),
        (PLSWNoise, C.PL_SW_NOISE),
        (TimingModel, None),  # top-level/admin params (incl. PHOFF); see _tables
    ]


def _spec_of(decl) -> dict:
    spec = {"kind": decl.kind, "unit": decl.unit}
    if decl.scale is not None:
        spec["scale"] = decl.scale
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
    param_spec: dict[str, dict] = {}
    alias_map: dict[str, str] = {}
    prefix_map: dict[str, str] = {}
    canonical_prefix: dict[str, str] = {}
    owners: dict[str, set] = {}        # param -> set of owning Component enums
    known: set[str] = set()
    binary_params: set[str] = set()

    _binary = {Component.BINARY, Component.BINARY_BT_PIECEWISE}
    for cls, comp in _component_classes():
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


def spec_for(name: str) -> Optional[dict]:
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
