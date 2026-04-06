"""Value parsing and unit conversion functions.

All functions are pure (no external dependencies beyond math/re) so that
the parser has zero dependency on astropy or PINT.
"""

from __future__ import annotations

import math
import re
from jaxpint.constants import SECS_PER_JULIAN_YEAR 

# ---------------------------------------------------------------------------
# Float parsing
# ---------------------------------------------------------------------------

_FORTRAN_D = re.compile(r"[Dd]")


def parse_float(s: str) -> float:
    """Parse a float string, handling Fortran D-notation (e.g. '1.23D-04')."""
    return float(_FORTRAN_D.sub("E", s))


# ---------------------------------------------------------------------------
# Angle parsing
# ---------------------------------------------------------------------------


def parse_hms_to_rad(s: str) -> float:
    """Parse 'HH:MM:SS.sss' right ascension string to radians."""
    parts = s.split(":")
    h = float(parts[0])
    m = float(parts[1]) if len(parts) > 1 else 0.0
    sec = float(parts[2]) if len(parts) > 2 else 0.0
    # RA: hours → degrees → radians  (1h = 15°)
    degrees = (abs(h) + m / 60.0 + sec / 3600.0) * 15.0
    if h < 0:
        degrees = -degrees
    return degrees * (math.pi / 180.0)


def parse_dms_to_rad(s: str) -> float:
    """Parse 'DD:MM:SS.sss' declination string to radians."""
    parts = s.split(":")
    d_str = parts[0]
    sign = -1.0 if d_str.lstrip().startswith("-") else 1.0
    d = abs(float(d_str))
    m = float(parts[1]) if len(parts) > 1 else 0.0
    sec = float(parts[2]) if len(parts) > 2 else 0.0
    degrees = sign * (d + m / 60.0 + sec / 3600.0)
    return degrees * (math.pi / 180.0)


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def deg_to_rad(value: float) -> float:
    """Convert degrees to radians."""
    return value * (math.pi / 180.0)


def deg_per_yr_to_rad_per_s(value: float) -> float:
    """Convert deg/yr to rad/s."""
    return value * (math.pi / 180.0) / (SECS_PER_JULIAN_YEAR)


def us_to_s(value: float) -> float:
    """Convert microseconds to seconds."""
    return value * 1e-6


# ---------------------------------------------------------------------------
# MJD splitting
# ---------------------------------------------------------------------------


def split_mjd(value: float) -> tuple[float, float]:
    """Split an MJD value into integer day + fractional day.

    Returns (int_day, frac_day) where int_day is a whole number and
    0 <= frac_day < 1.
    """
    int_part = float(int(value))
    return int_part, value - int_part


# ---------------------------------------------------------------------------
# TCB → TDB conversion
# ---------------------------------------------------------------------------

# See Irwin, A. W. & Fukushima, T. (1999) — "A numerical time ephemeris of the Earth." Astronomy & Astrophysics, 348, 642–652.
# Basic reason this exists is because TEMPO1 uses TCB, while moern timing packages use TDB time convention 
# This conversion is not necessary for moern timing software 

IFTE_MJD0 = 43144.0003725
IFTE_KM1 = 1.55051979176e-8
IFTE_K = 1.0 + IFTE_KM1


def tcb_scale_parameter(value: float, n: int) -> float:
    """Scale a parameter value from TCB to TDB: x_tdb = x_tcb * IFTE_K^n.

    Parameters
    ----------
    value : float
        Parameter value in TCB.
    n : int
        Effective time dimensionality (e.g. 1 for frequency, -1 for
        light-seconds, 2 for F1).
    """
    return value * (IFTE_K ** n)


def tcb_transform_mjd(value: float) -> float:
    """Transform an MJD from TCB to TDB.

    t_tdb = (t_tcb - IFTE_MJD0) / IFTE_K + IFTE_MJD0
    """
    return (value - IFTE_MJD0) / IFTE_K + IFTE_MJD0


# ---------------------------------------------------------------------------
# Conversion dispatcher
# ---------------------------------------------------------------------------

# Map of conversion names used in the registry to callables.
# Each callable takes a single float value and returns the converted value.
CONVERTERS: dict[str, callable] = {
    "deg_to_rad": deg_to_rad,
    "deg_per_yr_to_rad_per_s": deg_per_yr_to_rad_per_s,
    "us_to_s": us_to_s,
}
