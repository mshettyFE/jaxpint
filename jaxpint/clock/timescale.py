"""Clock-corrected (UTC-scale) MJD -> TDB conversion (PINT-free; astropy/erfa).

The clock chain (:mod:`jaxpint.clock.correction`) leaves each TOA as a UTC-scale
``pulsar_mjd`` value nudged by the (microsecond-level) clock corrections (which
target the TT(BIPM) realization); this module performs the remaining time-scale
step UTC -> TAI -> TT -> TDB via astropy, returning the longdouble TDB MJD split
into ``(int, frac)`` days.

``pulsar_mjd`` (the leap-second-day-aware MJD convention TEMPO/TEMPO2 use) is a
PINT-registered astropy format, *not* built into astropy.  Rather than depend on
PINT, we port the small ``mjds_to_jds_pulsar`` conversion (pure erfa) and hand
the resulting ``jd1/jd2`` to a standard ``Time(..., format="jd", scale="utc")``.
The two conventions differ only on the ~27 leap-second days (UTC days of 86401
s); observatory MJDs count uniform 86400-s days, so this is the correct
interpretation and matches PINT bit-for-bit.
"""

from __future__ import annotations

import erfa
import numpy as np
from astropy.time import Time
from astropy.time.utils import day_frac


def mjds_to_jds_pulsar(mjd1, mjd2):
    """Port of PINT's ``pulsar_mjd`` MJD->(jd1, jd2), leap-second-day aware.

    Converts an MJD (split into two parts for precision) interpreted in the
    TEMPO/TEMPO2 "ignore leap seconds" convention into astropy/erfa ``jd1, jd2``
    with leap seconds correctly placed.  Mirrors
    ``pint.pulsar_mjd.mjds_to_jds_pulsar``.
    """
    v1, v2 = day_frac(mjd1, mjd2)
    y, mo, d, f = erfa.jd2cal(erfa.DJM0 + v1, v2)
    # Fractional day -> H:M:S on a uniform 86400-second day (stable; avoids
    # np.remainder issues, matching PINT's comment).
    f = f * 24.0
    h = np.floor(f).astype(int)
    f -= h
    f = f * 60.0
    m = np.floor(f).astype(int)
    f -= m
    s = f * 60.0
    return erfa.dtf2d("UTC", y, mo, d, h, m, s)


def to_tdb(mjd_int, mjd_frac, itrf_xyz):
    """Convert clock-corrected UTC-scale MJDs to TDB ``(int, frac)``.

    Parameters
    ----------
    mjd_int, mjd_frac : array_like
        The corrected MJD, integer day + fractional day, as produced by the
        clock chain (UTC-scale ``pulsar_mjd``).
    itrf_xyz : array_like, shape (n, 3)
        Per-TOA geocentric ITRF xyz of the observatory in metres (the location
        affects the TDB topocentric correction; PINT attaches it before
        ``.tdb``).  Rows may be NaN/0 for the barycentre.

    Returns
    -------
    (tdb_int, tdb_frac) : tuple of float64 arrays
        TDB MJD split into integer + fractional day (frac in [0, 1)).
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    mjd_int = np.asarray(mjd_int, dtype=np.float64)
    mjd_frac = np.asarray(mjd_frac, dtype=np.float64)
    xyz = np.asarray(itrf_xyz, dtype=np.float64)

    jd1, jd2 = mjds_to_jds_pulsar(mjd_int, mjd_frac)
    loc = EarthLocation.from_geocentric(
        xyz[..., 0] * u.m, xyz[..., 1] * u.m, xyz[..., 2] * u.m
    )
    # Off load conversion to astropy
    t = Time(jd1, jd2, format="jd", scale="utc", location=loc)

    # Longdouble TDB MJD, split to (int, frac).  (jd1 is a half-integer JD, so
    # jd1 - DJM0 is the integer-ish MJD day and jd2 the fraction.)
    from ..utils import split_longdouble_days

    tdb = t.tdb
    ld = (np.longdouble(tdb.jd1) - np.longdouble(erfa.DJM0)) + np.longdouble(tdb.jd2)
    return split_longdouble_days(ld)
