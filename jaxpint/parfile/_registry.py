"""Declarative parameter metadata registry.

Replaces PINT's class hierarchy with a flat lookup table that maps
parameter names to their types, units, owning components, and conversion
rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ParamType(Enum):
    """Type of a timing-model parameter."""
    FLOAT = "float"
    MJD = "MJD"
    ANGLE_HMS = "angle_hms"
    ANGLE_DMS = "angle_dms"
    MASK = "mask"
    PAIR = "pair"
    STR = "str"
    INT = "int"
    BOOL = "bool"


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


@dataclass(frozen=True)
class ParamMeta:
    """Metadata for a single timing-model parameter."""

    param_type: ParamType
    """Parameter value type."""

    default_unit: str
    """
        Unit string stored in ParameterVector (e.g. "rad", "Hz", "s").
        See ParameterVector for unit convention. ParamMeta should follow this contract
    """

    component: Component
    """Owning JaxPINT component name (e.g. "Spindown", Component.BINARY)."""

    repeatable: bool = False
    """True for mask parameters that can appear multiple times."""

    is_epoch: bool = False
    """True for MJD parameters that get int/frac split."""

    convert: Optional[str] = None
    """Name of conversion function in _converters.CONVERTERS to apply
    after parsing (e.g. "deg_to_rad", "us_to_s")."""

    tcb2tdb_n: Optional[int] = None
    """Power *n* for TCB→TDB scaling: x_tdb = x_tcb * IFTE_K^n.
    None means the parameter is not converted.  For MJD/epoch
    parameters, the MJD transform formula is used instead of scaling."""

    scale_factor: Optional[float] = None
    """Implicit scale factor applied when abs(value) > scale_threshold.
    PINT uses this for PBDOT, A1DOT, EDOT (all 1e-12)."""

    scale_threshold: float = 1e-7
    """Threshold for applying scale_factor."""


# =========================================================================
# Explicit registry: known non-prefix parameters
# =========================================================================

PARAM_REGISTRY: dict[str, ParamMeta] = {
    # -- Spindown --
    "F0": ParamMeta(ParamType.FLOAT, "Hz", Component.SPINDOWN, tcb2tdb_n=1),
    "PEPOCH": ParamMeta(ParamType.MJD, "day", Component.SPINDOWN, is_epoch=True, tcb2tdb_n=0),
    "PHOFF": ParamMeta(ParamType.FLOAT, "cycle", Component.PHASE_OFFSET),

    # -- Astrometry Equatorial --
    "RAJ": ParamMeta(ParamType.ANGLE_HMS, "rad", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=0),
    "DECJ": ParamMeta(ParamType.ANGLE_DMS, "rad", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=0),
    "PMRA": ParamMeta(ParamType.FLOAT, "mas/yr", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=1),
    "PMDEC": ParamMeta(ParamType.FLOAT, "mas/yr", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=1),
    "PX": ParamMeta(ParamType.FLOAT, "mas", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=-1),
    "POSEPOCH": ParamMeta(ParamType.MJD, "day", Component.ASTROMETRY_EQUATORIAL, is_epoch=True, tcb2tdb_n=0),
    "PMDD": ParamMeta(ParamType.FLOAT, "mas/yr^2", Component.ASTROMETRY_EQUATORIAL, tcb2tdb_n=2),

    # -- Astrometry Ecliptic --
    "ELONG": ParamMeta(ParamType.FLOAT, "rad", Component.ASTROMETRY_ECLIPTIC, convert="deg_to_rad", tcb2tdb_n=0),
    "ELAT": ParamMeta(ParamType.FLOAT, "rad", Component.ASTROMETRY_ECLIPTIC, convert="deg_to_rad", tcb2tdb_n=0),
    "PMELONG": ParamMeta(ParamType.FLOAT, "mas/yr", Component.ASTROMETRY_ECLIPTIC, tcb2tdb_n=1),
    "PMELAT": ParamMeta(ParamType.FLOAT, "mas/yr", Component.ASTROMETRY_ECLIPTIC, tcb2tdb_n=1),
    "ECL": ParamMeta(ParamType.STR, "", Component.ASTROMETRY_ECLIPTIC),

    # -- Dispersion --
    "DM": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DISPERSION_DM, tcb2tdb_n=1),
    "DMEPOCH": ParamMeta(ParamType.MJD, "day", Component.DISPERSION_DM, is_epoch=True, tcb2tdb_n=0),

    # -- DispersionDMX base --
    "DMX": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DISPERSION_DMX, tcb2tdb_n=1),

    # -- Solar wind --
    "NE_SW": ParamMeta(ParamType.FLOAT, "cm^-3", Component.SOLAR_WIND_DISPERSION, tcb2tdb_n=1),
    "SWM": ParamMeta(ParamType.INT, "", Component.SOLAR_WIND_DISPERSION),
    "SWP": ParamMeta(ParamType.FLOAT, "", Component.SOLAR_WIND_DISPERSION),
    "SWEPOCH": ParamMeta(ParamType.MJD, "day", Component.SOLAR_WIND_DISPERSION, is_epoch=True, tcb2tdb_n=0),

    # -- Solar system Shapiro --
    "PLANET_SHAPIRO": ParamMeta(ParamType.BOOL, "", Component.SOLAR_SYSTEM_SHAPIRO),

    # -- Troposphere --
    "CORRECT_TROPOSPHERE": ParamMeta(ParamType.BOOL, "", Component.TROPOSPHERE_DELAY),

    # -- Binary common --
    "BINARY": ParamMeta(ParamType.STR, "", Component.BINARY),
    "PB": ParamMeta(ParamType.FLOAT, "day", Component.BINARY, tcb2tdb_n=-1),
    "PBDOT": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0, scale_factor=1e-12),
    "T0": ParamMeta(ParamType.MJD, "day", Component.BINARY, is_epoch=True, tcb2tdb_n=0),
    "TASC": ParamMeta(ParamType.MJD, "day", Component.BINARY, is_epoch=True, tcb2tdb_n=0),
    "A1": ParamMeta(ParamType.FLOAT, "lsec", Component.BINARY, tcb2tdb_n=-1),
    "A1DOT": ParamMeta(ParamType.FLOAT, "lsec/s", Component.BINARY, tcb2tdb_n=0, scale_factor=1e-12),
    "ECC": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "EDOT": ParamMeta(ParamType.FLOAT, "1/s", Component.BINARY, tcb2tdb_n=1, scale_factor=1e-12),
    "OM": ParamMeta(ParamType.FLOAT, "rad", Component.BINARY, convert="deg_to_rad", tcb2tdb_n=0),
    "OMDOT": ParamMeta(ParamType.FLOAT, "rad/s", Component.BINARY, convert="deg_per_yr_to_rad_per_s", tcb2tdb_n=1),
    "XPBDOT": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "GAMMA": ParamMeta(ParamType.FLOAT, "s", Component.BINARY, tcb2tdb_n=-1),
    "M2": ParamMeta(ParamType.FLOAT, "Msun", Component.BINARY, tcb2tdb_n=-1),
    "SINI": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "DR": ParamMeta(ParamType.FLOAT, "", Component.BINARY),
    "DTH": ParamMeta(ParamType.FLOAT, "", Component.BINARY),
    "A0": ParamMeta(ParamType.FLOAT, "s", Component.BINARY),
    "B0": ParamMeta(ParamType.FLOAT, "s", Component.BINARY),
    "EPS1": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "EPS2": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "EPS1DOT": ParamMeta(ParamType.FLOAT, "1/s", Component.BINARY, tcb2tdb_n=1),
    "EPS2DOT": ParamMeta(ParamType.FLOAT, "1/s", Component.BINARY, tcb2tdb_n=1),
    "LNEDOT": ParamMeta(ParamType.FLOAT, "1/s", Component.BINARY, tcb2tdb_n=1),
    "H3": ParamMeta(ParamType.FLOAT, "s", Component.BINARY, tcb2tdb_n=-1),
    "H4": ParamMeta(ParamType.FLOAT, "s", Component.BINARY, tcb2tdb_n=-1),
    "STIGMA": ParamMeta(ParamType.FLOAT, "", Component.BINARY, tcb2tdb_n=0),
    "SHAPMAX": ParamMeta(ParamType.FLOAT, "", Component.BINARY),
    "KIN": ParamMeta(ParamType.FLOAT, "rad", Component.BINARY, convert="deg_to_rad"),
    "KOM": ParamMeta(ParamType.FLOAT, "rad", Component.BINARY, convert="deg_to_rad"),
    "K96": ParamMeta(ParamType.BOOL, "", Component.BINARY),
    "MTOT": ParamMeta(ParamType.FLOAT, "Msun", Component.BINARY, tcb2tdb_n=-1),
    "XOMDOT": ParamMeta(ParamType.FLOAT, "rad/s", Component.BINARY, convert="deg_per_yr_to_rad_per_s"),

    # -- Chromatic measure --
    "CM": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.CHROMATIC_CM, tcb2tdb_n=1),
    "CMEPOCH": ParamMeta(ParamType.MJD, "day", Component.CHROMATIC_CM, is_epoch=True, tcb2tdb_n=0),
    "TNCHROMIDX": ParamMeta(ParamType.FLOAT, "", Component.CHROMATIC_CM),

    # -- WaveX --
    "WXEPOCH": ParamMeta(ParamType.MJD, "day", Component.WAVE_X, is_epoch=True, tcb2tdb_n=0),

    # -- DMWaveX --
    "DMWXEPOCH": ParamMeta(ParamType.MJD, "day", Component.DM_WAVE_X, is_epoch=True, tcb2tdb_n=0),

    # -- CMWaveX --
    "CMWXEPOCH": ParamMeta(ParamType.MJD, "day", Component.CM_WAVE_X, is_epoch=True, tcb2tdb_n=0),

    # -- Wave --
    "WAVE_OM": ParamMeta(ParamType.FLOAT, "1/day", Component.WAVE),
    "WAVEEPOCH": ParamMeta(ParamType.MJD, "day", Component.WAVE, is_epoch=True),

    # -- IFunc --
    "SIFUNC": ParamMeta(ParamType.INT, "", Component.IFUNC),

    # -- Exponential dip --
    "EXPDIPEPS": ParamMeta(ParamType.FLOAT, "", Component.EXPONENTIAL_DIP),
    "EXPDIPFREF": ParamMeta(ParamType.FLOAT, "MHz", Component.EXPONENTIAL_DIP),

    # -- FDJump --
    "FDJUMPLOG": ParamMeta(ParamType.BOOL, "", Component.FD_JUMP),

    # -- Mask parameters (repeatable) --
    "JUMP": ParamMeta(ParamType.MASK, "s", Component.PHASE_JUMP, repeatable=True, tcb2tdb_n=-1),
    "EFAC": ParamMeta(ParamType.MASK, "", Component.SCALE_TOA_ERROR, repeatable=True),
    "EQUAD": ParamMeta(ParamType.MASK, "s", Component.SCALE_TOA_ERROR, repeatable=True, convert="us_to_s"),
    "ECORR": ParamMeta(ParamType.MASK, "s", Component.ECORR_NOISE, repeatable=True, convert="us_to_s"),
    "DMEFAC": ParamMeta(ParamType.MASK, "", Component.SCALE_DM_ERROR, repeatable=True),
    "DMEQUAD": ParamMeta(ParamType.MASK, "s", Component.SCALE_DM_ERROR, repeatable=True, convert="us_to_s"),
    "DMJUMP": ParamMeta(ParamType.MASK, "pc cm^-3", Component.DISPERSION_JUMP, repeatable=True),

    # -- Noise (power-law) --
    "TNREDAMP": ParamMeta(ParamType.FLOAT, "", Component.PL_RED_NOISE),
    "TNREDGAM": ParamMeta(ParamType.FLOAT, "", Component.PL_RED_NOISE),
    "TNREDC": ParamMeta(ParamType.INT, "", Component.PL_RED_NOISE),
    "TNREDTSPAN": ParamMeta(ParamType.FLOAT, "day", Component.PL_RED_NOISE),
    "TNDMAMP": ParamMeta(ParamType.FLOAT, "", Component.PL_DM_NOISE),
    "TNDMGAM": ParamMeta(ParamType.FLOAT, "", Component.PL_DM_NOISE),
    "TNDMC": ParamMeta(ParamType.INT, "", Component.PL_DM_NOISE),
    "TNDMTSPAN": ParamMeta(ParamType.FLOAT, "day", Component.PL_DM_NOISE),
    "TNCHROMAMP": ParamMeta(ParamType.FLOAT, "", Component.PL_CHROM_NOISE),
    "TNCHROMGAM": ParamMeta(ParamType.FLOAT, "", Component.PL_CHROM_NOISE),
    "TNCHROMC": ParamMeta(ParamType.INT, "", Component.PL_CHROM_NOISE),
    "TNCHROMTSPAN": ParamMeta(ParamType.FLOAT, "day", Component.PL_CHROM_NOISE),
    "TNSWAMP": ParamMeta(ParamType.FLOAT, "", Component.PL_SW_NOISE),
    "TNSWGAM": ParamMeta(ParamType.FLOAT, "", Component.PL_SW_NOISE),
    "TNSWC": ParamMeta(ParamType.INT, "", Component.PL_SW_NOISE),

    # -- Metadata (non-numeric, not in ParameterVector) --
    "PSR": ParamMeta(ParamType.STR, "", Component.NONE),
    "EPHEM": ParamMeta(ParamType.STR, "", Component.NONE),
    "UNITS": ParamMeta(ParamType.STR, "", Component.NONE),
    "CLOCK": ParamMeta(ParamType.STR, "", Component.NONE),
    "START": ParamMeta(ParamType.MJD, "day", Component.NONE, is_epoch=True),
    "FINISH": ParamMeta(ParamType.MJD, "day", Component.NONE, is_epoch=True),
    "TZRMJD": ParamMeta(ParamType.MJD, "day", Component.NONE, is_epoch=True),
    "TZRSITE": ParamMeta(ParamType.STR, "", Component.NONE),
    "TZRFRQ": ParamMeta(ParamType.FLOAT, "MHz", Component.NONE),
    "TRACK": ParamMeta(ParamType.STR, "", Component.NONE),
    "INFO": ParamMeta(ParamType.STR, "", Component.NONE),
    "NTOA": ParamMeta(ParamType.INT, "", Component.NONE),
    "CHI2": ParamMeta(ParamType.FLOAT, "", Component.NONE),
    "CHI2R": ParamMeta(ParamType.FLOAT, "", Component.NONE),
    "TRES": ParamMeta(ParamType.FLOAT, "us", Component.NONE),
    "DMDATA": ParamMeta(ParamType.INT, "", Component.NONE),
    "TIMEEPH": ParamMeta(ParamType.STR, "", Component.NONE),
    "T2CMETHOD": ParamMeta(ParamType.STR, "", Component.NONE),
    "DILATEFREQ": ParamMeta(ParamType.BOOL, "", Component.NONE),
    "PLANET_SHAPIRO": ParamMeta(ParamType.BOOL, "", Component.SOLAR_SYSTEM_SHAPIRO),
    "MODE": ParamMeta(ParamType.INT, "", Component.NONE),
    "NITS": ParamMeta(ParamType.INT, "", Component.NONE),
    "RNAMP": ParamMeta(ParamType.FLOAT, "", Component.PL_RED_NOISE),
    "RNIDX": ParamMeta(ParamType.FLOAT, "", Component.PL_RED_NOISE),
}


# =========================================================================
# Prefix registry: for indexed parameter families
# =========================================================================

# Entries keyed by canonical prefix (e.g. "F" matches F0, F1, F2, ...).
# The component field identifies the owning component for all members.

PREFIX_REGISTRY: dict[str, ParamMeta] = {
    # -- Spindown --
    "F": ParamMeta(ParamType.FLOAT, "Hz", Component.SPINDOWN, tcb2tdb_n=1),
    # Note: F0 is in PARAM_REGISTRY too; explicit entries take priority.
    # For F1, n=2; F2, n=3; etc. -- handled by adding index to tcb2tdb_n.

    # -- Dispersion Taylor --
    "DM": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DISPERSION_DM, tcb2tdb_n=1),

    # -- DispersionDMX --
    "DMX_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DISPERSION_DMX, tcb2tdb_n=1),
    "DMXR1_": ParamMeta(ParamType.MJD, "day", Component.DISPERSION_DMX, is_epoch=True, tcb2tdb_n=0),
    "DMXR2_": ParamMeta(ParamType.MJD, "day", Component.DISPERSION_DMX, is_epoch=True, tcb2tdb_n=0),
    # DMXEP_, DMXF1_, DMXF2_ are DMX metadata (not in ParameterVector)
    "DMXEP_": ParamMeta(ParamType.MJD, "day", Component.NONE, is_epoch=True),
    "DMXF1_": ParamMeta(ParamType.FLOAT, "MHz", Component.NONE),
    "DMXF2_": ParamMeta(ParamType.FLOAT, "MHz", Component.NONE),

    # -- WaveX --
    "WXFREQ_": ParamMeta(ParamType.FLOAT, "1/day", Component.WAVE_X, tcb2tdb_n=-1),
    "WXSIN_": ParamMeta(ParamType.FLOAT, "s", Component.WAVE_X, tcb2tdb_n=-1),
    "WXCOS_": ParamMeta(ParamType.FLOAT, "s", Component.WAVE_X, tcb2tdb_n=-1),

    # -- DMWaveX --
    "DMWXFREQ_": ParamMeta(ParamType.FLOAT, "1/day", Component.DM_WAVE_X, tcb2tdb_n=-1),
    "DMWXSIN_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DM_WAVE_X, tcb2tdb_n=1),
    "DMWXCOS_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.DM_WAVE_X, tcb2tdb_n=1),

    # -- CMWaveX --
    "CMWXFREQ_": ParamMeta(ParamType.FLOAT, "1/day", Component.CM_WAVE_X, tcb2tdb_n=-1),
    "CMWXSIN_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.CM_WAVE_X, tcb2tdb_n=1),
    "CMWXCOS_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.CM_WAVE_X, tcb2tdb_n=1),

    # -- ChromaticCM Taylor --
    "CM": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.CHROMATIC_CM, tcb2tdb_n=1),

    # -- ChromaticCMX --
    "CMX_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.CHROMATIC_CMX, tcb2tdb_n=1),
    "CMXR1_": ParamMeta(ParamType.MJD, "day", Component.CHROMATIC_CMX, is_epoch=True, tcb2tdb_n=0),
    "CMXR2_": ParamMeta(ParamType.MJD, "day", Component.CHROMATIC_CMX, is_epoch=True, tcb2tdb_n=0),

    # -- Glitch --
    "GLEP_": ParamMeta(ParamType.MJD, "day", Component.GLITCH, is_epoch=True, tcb2tdb_n=0),
    "GLPH_": ParamMeta(ParamType.FLOAT, "cycle", Component.GLITCH, tcb2tdb_n=0),
    "GLF0_": ParamMeta(ParamType.FLOAT, "Hz", Component.GLITCH, tcb2tdb_n=1),
    "GLF1_": ParamMeta(ParamType.FLOAT, "Hz/s", Component.GLITCH, tcb2tdb_n=2),
    "GLF2_": ParamMeta(ParamType.FLOAT, "Hz/s^2", Component.GLITCH, tcb2tdb_n=3),
    "GLF0D_": ParamMeta(ParamType.FLOAT, "Hz", Component.GLITCH, tcb2tdb_n=1),
    "GLTD_": ParamMeta(ParamType.FLOAT, "day", Component.GLITCH, tcb2tdb_n=-1),

    # -- Piecewise spindown --
    "PWSTART_": ParamMeta(ParamType.MJD, "day", Component.PIECEWISE_SPINDOWN, is_epoch=True),
    "PWSTOP_": ParamMeta(ParamType.MJD, "day", Component.PIECEWISE_SPINDOWN, is_epoch=True),
    "PWEP_": ParamMeta(ParamType.MJD, "day", Component.PIECEWISE_SPINDOWN, is_epoch=True),
    "PWPH_": ParamMeta(ParamType.FLOAT, "cycle", Component.PIECEWISE_SPINDOWN),
    "PWF0_": ParamMeta(ParamType.FLOAT, "Hz", Component.PIECEWISE_SPINDOWN),
    "PWF1_": ParamMeta(ParamType.FLOAT, "Hz/s", Component.PIECEWISE_SPINDOWN),
    "PWF2_": ParamMeta(ParamType.FLOAT, "Hz/s^2", Component.PIECEWISE_SPINDOWN),

    # -- Frequency dependent --
    "FD": ParamMeta(ParamType.FLOAT, "s", Component.FREQUENCY_DEPENDENT),

    # -- Exponential dip --
    "EXPDIPEPOCH_": ParamMeta(ParamType.MJD, "day", Component.EXPONENTIAL_DIP, is_epoch=True),
    "EXPDIPAMP_": ParamMeta(ParamType.FLOAT, "s", Component.EXPONENTIAL_DIP),
    "EXPDIPIDX_": ParamMeta(ParamType.FLOAT, "", Component.EXPONENTIAL_DIP),
    "EXPDIPTAU_": ParamMeta(ParamType.FLOAT, "day", Component.EXPONENTIAL_DIP),

    # -- BT piecewise --
    "T0X_": ParamMeta(ParamType.MJD, "day", Component.BINARY_BT_PIECEWISE, is_epoch=True, tcb2tdb_n=0),
    "A1X_": ParamMeta(ParamType.FLOAT, "lsec", Component.BINARY_BT_PIECEWISE, tcb2tdb_n=-1),
    "XR1_": ParamMeta(ParamType.MJD, "day", Component.BINARY_BT_PIECEWISE, is_epoch=True, tcb2tdb_n=0),
    "XR2_": ParamMeta(ParamType.MJD, "day", Component.BINARY_BT_PIECEWISE, is_epoch=True, tcb2tdb_n=0),

    # -- FDJump (special prefix: FDnJUMPm) --
    # Handled as a special case in the tokenizer/builder.

    # -- Wave (pair parameters) --
    "WAVE": ParamMeta(ParamType.PAIR, "s", Component.WAVE),

    # -- IFunc (pair parameters) --
    "IFUNC": ParamMeta(ParamType.PAIR, "", Component.IFUNC),

    # -- Solar wind --
    "NE_SW": ParamMeta(ParamType.FLOAT, "cm^-3", Component.SOLAR_WIND_DISPERSION, tcb2tdb_n=1),

    # -- Solar wind X --
    "SWXDM_": ParamMeta(ParamType.FLOAT, "pc cm^-3", Component.SOLAR_WIND_DISPERSION_X, tcb2tdb_n=1),
    "SWXP_": ParamMeta(ParamType.FLOAT, "", Component.SOLAR_WIND_DISPERSION_X),
    "SWXR1_": ParamMeta(ParamType.MJD, "day", Component.SOLAR_WIND_DISPERSION_X, is_epoch=True, tcb2tdb_n=0),
    "SWXR2_": ParamMeta(ParamType.MJD, "day", Component.SOLAR_WIND_DISPERSION_X, is_epoch=True, tcb2tdb_n=0),

    # -- Binary FB --
    "FB": ParamMeta(ParamType.FLOAT, "1/s", Component.BINARY, tcb2tdb_n=1),

    # -- Orbital wave (rarely used) --
    "ORBWAVEC": ParamMeta(ParamType.FLOAT, "", Component.BINARY),
    "ORBWAVES": ParamMeta(ParamType.FLOAT, "", Component.BINARY),
}


# =========================================================================
# Prefix detection regex patterns (from PINT's split_prefixed_name)
# =========================================================================

# Can test out new patterns at: https://regex101.com/
PREFIX_PATTERNS = [
    re.compile(r"^([a-zA-Z]*\d+[a-zA-Z]+)(\d+)$"),   # e.g. T2EFAC2, FD1JUMP2
    re.compile(r"^([a-zA-Z]+)0*(\d+)$"),               # e.g. F12, DM1, FD1
    re.compile(r"^([a-zA-Z0-9]+_)(\d+)$"),             # e.g. DMX_0001, GLEP_1
    re.compile(r"^([a-zA-Z]+_[a-zA-Z]+)(\d+)$"),       # e.g. NE_SW2
]


def split_prefixed_name(name: str) -> tuple[str, str, int] | None:
    """Split an indexed parameter name into (prefix, index_str, index_int).

    Returns None if the name does not match any prefix pattern.
    """
    for pat in PREFIX_PATTERNS:
        m = pat.match(name)
        if m:
            prefix = m.group(1)
            idx_str = m.group(2)
            return prefix, idx_str, int(idx_str)
    return None

def lookup(name: str) -> tuple[ParamMeta, str] | None:
    """Look up parameter metadata by name.

    Returns (ParamMeta, canonical_name) or None if unknown.
    The canonical_name may differ from the input for prefix params.
    """
    # Direct lookup first
    if name in PARAM_REGISTRY:
        return PARAM_REGISTRY[name], name

    # Try prefix splitting
    split = split_prefixed_name(name)
    if split is not None:
        prefix, idx_str, idx_int = split

        # Try prefix with underscore (e.g. "DMX_")
        if prefix in PREFIX_REGISTRY:
            return PREFIX_REGISTRY[prefix], name

        # Try prefix without underscore (e.g. "F" for F0, F1, ...)
        # but only if the prefix itself isn't already a full param
        if prefix in PREFIX_REGISTRY:
            return PREFIX_REGISTRY[prefix], name

    return None
