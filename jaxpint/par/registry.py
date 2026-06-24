"""Component and binary model enumerations for JaxPINT.

These enums identify which timing model components are active and which
binary orbital model is in use.  They are the shared vocabulary between
the parameter adapters (the native ``.par`` parser :mod:`jaxpint.par.parser`
and the PINT bridge :mod:`jaxpint.bridge`) and the model
builder (:mod:`jaxpint.model_builder`).

This module is PINT-free.
"""

from __future__ import annotations

from enum import Enum


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

    NONE = ""
    SPINDOWN = "Spindown"
    PHASE_OFFSET = "PhaseOffset"
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
