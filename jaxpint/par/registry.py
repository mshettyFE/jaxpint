"""Component and binary model enumerations for JaxPINT.

These enums identify which timing model components are active and which
binary orbital model is in use.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class BinaryModel(Enum):
    """Supported binary orbital models."""

    BT = "BT"
    DD = "DD"
    DDS = "DDS"
    DDH = "DDH"
    DDK = "DDK"
    DDGR = "DDGR"
    ELL1 = "ELL1"
    ELL1H = "ELL1H"
    ELL1k = "ELL1k"
    BT_PIECEWISE = "BT_piecewise"


class Component(Enum):
    """JaxPINT timing model component that owns a parameter."""

    SPINDOWN = "Spindown"
    ASTROMETRY_EQUATORIAL = "AstrometryEquatorial"
    ASTROMETRY_ECLIPTIC = "AstrometryEcliptic"
    DISPERSION_DM = "DispersionDM"
    DISPERSION_DMX = "DispersionDMX"
    DISPERSION_JUMP = "DispersionJump"
    BINARY = "Binary"
    BINARY_BT_PIECEWISE = "BinaryBTPiecewise"
    SOLAR_SYSTEM_SHAPIRO = "SolarSystemShapiro"
    SOLAR_WIND_DISPERSION = "SolarWindDispersion"
    SOLAR_WIND_DISPERSION_X = "SolarWindDispersionX"
    TROPOSPHERE_DELAY = "TroposphereDelay"
    CHROMATIC_CM = "ChromaticCM"
    CHROMATIC_CMX = "ChromaticCMX"
    WAVE_X = "WaveX"
    DM_WAVE_X = "DMWaveX"
    CM_WAVE_X = "CMWaveX"
    WAVE = "Wave"
    IFUNC = "IFunc"
    EXPONENTIAL_DIP = "ExponentialDip"
    FREQUENCY_DEPENDENT = "FrequencyDependent"
    FD_JUMP = "FDJump"
    PHASE_JUMP = "PhaseJump"
    PIECEWISE_SPINDOWN = "PiecewiseSpindown"
    GLITCH = "Glitch"
    SCALE_TOA_ERROR = "ScaleToaError"
    SCALE_DM_ERROR = "ScaleDmError"
    ECORR_NOISE = "EcorrNoise"
    PL_RED_NOISE = "PLRedNoise"
    PL_DM_NOISE = "PLDMNoise"
    PL_CHROM_NOISE = "PLChromNoise"
    PL_SW_NOISE = "PLSWNoise"


def binary_component_for(
    name: Optional[str],
) -> tuple[Optional[BinaryModel], Optional[Component]]:
    """Map a binary-model name to its ``(BinaryModel, Component)`` pair.

    ``BT_piecewise`` has its own component; every other recognized model shares
    :attr:`Component.BINARY`.  Returns ``(None, None)`` when *name* is not a
    known :class:`BinaryModel` (the caller decides whether to warn).

    Single source of truth for the binary model -> component rule, shared by the
    native detector (:func:`jaxpint.par.components._detect_binary`) and the PINT
    bridge (:func:`jaxpint.bridge.model_conversion._pint_detect_components`).
    """
    try:
        model = BinaryModel(name)
    except ValueError:
        return None, None
    comp = (
        Component.BINARY_BT_PIECEWISE
        if model is BinaryModel.BT_PIECEWISE
        else Component.BINARY
    )
    return model, comp
