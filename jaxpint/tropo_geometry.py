"""Per-TOA troposphere elevation geometry (astropy ``AltAz``), PINT-free.

Shared by the native loader (:mod:`jaxpint.loaders.native`) and the PINT bridge
(:mod:`jaxpint.bridge.toa_conversion`): each resolves the genuinely
source-specific inputs — the target sky position ``radec`` (a *fixed* SkyCoord,
no proper motion, matching PINT's ``_get_target_skycoord``) and its per-observatory
``EarthLocation`` — then hands them here.  This owns the shared part: the
elevation-angle transform and the physical-validity clamp, so the two loaders
can't drift on it.
"""

from __future__ import annotations

import numpy as np


def tropo_fields(radec, obs_groups: dict, mjd, n_toas: int) -> dict:
    """Elevation angle + observatory lat/height per TOA, with the validity clamp.

    Parameters
    ----------
    radec : astropy.coordinates.SkyCoord
        Fixed target sky position (no proper motion).
    obs_groups : dict[str, tuple[EarthLocation, indices]]
        One entry per **topocentric** observatory (non-topocentric ones are the
        caller's to skip): its ``EarthLocation`` and the indices of its TOAs.
        The indices may be a list, array, or slice.
    mjd : (n_toas,) array
        UTC MJD of every TOA.
    n_toas : int
        Total number of TOAs (sets the output length).

    Returns
    -------
    dict
        ``tropo_alt`` (rad, elevation; invalid entries clamped to ``pi/2``),
        ``tropo_alt_valid`` (bool), ``obs_geodetic_lat`` (rad),
        ``obs_height_km`` (km).  Entries for non-topocentric TOAs stay zero /
        ``valid=False``.
    """
    from astropy import units as u
    from astropy.coordinates import AltAz
    from astropy.time import Time

    alt = np.zeros(n_toas, dtype=np.float64)
    lat = np.zeros(n_toas, dtype=np.float64)
    height = np.zeros(n_toas, dtype=np.float64)
    valid = np.zeros(n_toas, dtype=bool)

    for loc, idx in obs_groups.values():
        obstime = Time(mjd[idx], format="mjd", scale="utc")
        # astropy SkyCoord / transform_to are under-typed (return Optional in stubs).
        a = radec.transform_to(AltAz(location=loc, obstime=obstime)).alt.to_value(u.rad)  # pyright: ignore[reportCallIssue, reportOptionalCall, reportOptionalMemberAccess]
        alt[idx] = a
        lat[idx] = loc.lat.to_value(u.rad)
        height[idx] = loc.height.to_value(u.km)
        valid[idx] = True

    # Validate altitudes: must be in [0, pi/2]; clamp invalid to zenith.
    bad = (alt < 0.0) | (alt > np.pi / 2.0)
    valid[bad] = False
    alt[bad] = np.pi / 2.0

    return {
        "tropo_alt": alt,
        "tropo_alt_valid": valid,
        "obs_geodetic_lat": lat,
        "obs_height_km": height,
    }
