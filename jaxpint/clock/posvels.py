"""Barycentric observatory position/velocity (PINT-free; astropy/erfa).

Reproduces PINT's ``compute_posvels`` for one observatory: the observatory's
position/velocity relative to the solar-system barycentre, plus the
observatory->Sun and (optionally) observatory->planet vectors, all from a TDB
time, the observatory ITRF xyz, and a JPL ephemeris.

    ssb_obs    = gcrs_posvel_from_itrf(loc, tdb)        [erfa, origin=earth]
               + objPosVel_wrt_SSB("earth", tdb, ephem) [astropy + JPL .bsp]
    obs_sun    = objPosVel_wrt_SSB("sun", tdb)  - ssb_obs
    obs_planet = objPosVel_wrt_SSB(planet, tdb) - ssb_obs

Positions are km, velocities km/s, matching PINT's table columns.
"""

from __future__ import annotations

import os

import numpy as np
from astropy import units as u
from astropy.coordinates import (
    EarthLocation,
    get_body_barycentric_posvel,
    solar_system_ephemeris,
)

from astropy.time import Time
from astropy.utils.data import download_file

from ..constants import PLANETS, SolarSystemBody
from . import config

# Mirror list for JPL SPK kernels. Order matters: FTP is first so the fallback
# below skips TLS entirely -- works in HPC containers whose CA bundles are
# missing, where astropy's default HTTPS URL fails cert verification. (Same
# trick PINT uses in pint.solar_system_ephemerides.)
_EPHEMERIS_MIRRORS: tuple[str, ...] = (
    "ftp://ssd.jpl.nasa.gov/pub/eph/planets/bsp/",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/",
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
)

# location of already loaded ephems
_LOADED_EPHEMS: dict[str, str] = {}


def _ensure_ephemeris(ephem: str) -> str:
    """Resolve ``ephem`` to a value usable in ``solar_system_ephemeris.set(...)``.

    Tries, in order, returning at the first success:

    1. In-process cache (``_LOADED_EPHEMS``) — repeat calls are O(1).
    2. ``$JAXPINT_EPHEM_PATH`` — a pre-staged ``.bsp`` file or directory (lets
       you avoid the network on locked-down nodes).
    3. astropy's name-based resolver — handles built-in ephemerides without
       network and reuses astropy's own download cache.
    4. ``download_file(sources=mirrors)`` with the FTP mirror first, so
       containers without a working CA bundle still succeed via plain FTP.

    Returns the value to pass to ``solar_system_ephemeris.set(...)`` -- either
    the original name (when astropy resolved it) or a local file path.
    """
    key = ephem.lower()
    cached = _LOADED_EPHEMS.get(key)
    if cached is not None:
        return cached

    local = config.get("JAXPINT_EPHEM_PATH")
    if local:
        p = local if os.path.isfile(local) else os.path.join(local, f"{key}.bsp")
        if os.path.isfile(p):
            _LOADED_EPHEMS[key] = p
            return p

    # astropy's own resolver: populates its URL cache as a side effect. The
    # `with ... : pass` restores the previous global setting once validated, so
    # we don't leak state if the caller has its own scoping.
    try:
        with solar_system_ephemeris.set(ephem):
            pass
        _LOADED_EPHEMS[key] = ephem
        return ephem
    except (ValueError, OSError):
        pass

    sources = [f"{m}{key}.bsp" for m in _EPHEMERIS_MIRRORS]
    p = download_file(sources[0], cache=True, sources=sources)
    _LOADED_EPHEMS[key] = p
    return p


def _body_ssb_posvel(body: SolarSystemBody, tdb: Time):
    """(pos_km, vel_kms) of ``body`` wrt the SSB at the given TDB times."""
    pos, vel = get_body_barycentric_posvel(body, tdb)
    assert pos is not None and vel is not None
    return (
        pos.xyz.to_value(u.km).T,  # (n, 3)
        vel.xyz.to_value(u.km / u.s).T,  # (n, 3)
    )


def compute_posvels(
    tdb_int,
    tdb_frac,
    itrf_xyz,
    *,
    ephem: str = "DE440",
    planets: bool = False,
):
    """Compute barycentric posvels for one observatory group.

    Parameters
    ----------
    tdb_int, tdb_frac : array_like, shape (n,)
        TDB MJD (integer + fractional day) for the group's TOAs.
    itrf_xyz : tuple/array of 3 floats, or None
        The observatory's geocentric ITRF xyz in metres.  ``None`` (the
        barycentre) yields a zero observatory term.
    ephem : str
        JPL ephemeris name; ``"DE440"`` by default.  Commonly used names are
        ``DE200``, ``DE405``, ``DE421``, ``DE430``, ``DE436``, ``DE438``,
        ``DE440``, ``DE440s``, and ``DE441`` (DE421/DE430/DE436/DE440 are the
        usual NANOGrav/IPTA choices).  The set is *not* closed: any name astropy
        can resolve works, as does any ``<name>.bsp`` reachable via
        ``$JAXPINT_EPHEM_PATH`` or the download mirrors -- so newer JPL releases
        are accepted without a code change.  Resolution + caching is handled by
        :func:`_ensure_ephemeris`.
    planets : bool
        Also compute observatory->planet vectors.

    Returns
    -------
    dict with ``ssb_obs_pos`` (n,3 km), ``ssb_obs_vel`` (n,3 km/s),
    ``obs_sun_pos`` (n,3 km), and (if ``planets``) ``planet_positions``:
    ``{f"obs_{p}_pos": (n,3) km}``.
    """
    tdb_int = np.asarray(tdb_int, dtype=np.float64)
    tdb_frac = np.asarray(tdb_frac, dtype=np.float64)
    n = tdb_int.shape[0]
    # astropy Time in TDB from the two-part MJD (jd1/jd2 for precision).
    tdb = Time(tdb_int, tdb_frac, format="mjd", scale="tdb")

    resolved = _ensure_ephemeris(ephem)
    with solar_system_ephemeris.set(resolved):
        earth_pos, earth_vel = _body_ssb_posvel("earth", tdb)

        if itrf_xyz is None:
            # Barycentre: the observatory *is* the SSB, so its SSB-relative
            # posvel is zero (matches the native TZR path and PINT).  The
            # geocentre is NOT this case -- it carries itrf_xyz == (0, 0, 0) and
            # goes through the else branch (giving ssb_obs == earth_pos).
            ssb_obs_pos = np.zeros((n, 3))
            ssb_obs_vel = np.zeros((n, 3))
        else:
            xyz = np.asarray(itrf_xyz, dtype=np.float64)
            loc = EarthLocation.from_geocentric(
                xyz[0] * u.m, xyz[1] * u.m, xyz[2] * u.m
            )
            # erfa: observatory posvel wrt geocentre in GCRS.
            gpos, gvel = loc.get_gcrs_posvel(tdb)
            obs_geo_pos = gpos.xyz.to_value(u.km).T
            obs_geo_vel = gvel.xyz.to_value(u.km / u.s).T
            ssb_obs_pos = obs_geo_pos + earth_pos
            ssb_obs_vel = obs_geo_vel + earth_vel

        sun_pos, _ = _body_ssb_posvel("sun", tdb)
        obs_sun_pos = sun_pos - ssb_obs_pos

        out = {
            "ssb_obs_pos": ssb_obs_pos,
            "ssb_obs_vel": ssb_obs_vel,
            "obs_sun_pos": obs_sun_pos,
        }
        if planets:
            pp: dict[str, np.ndarray] = {}
            for p in PLANETS:
                p_pos, _ = _body_ssb_posvel(p, tdb)
                pp[f"obs_{p}_pos"] = p_pos - ssb_obs_pos
            out["planet_positions"] = pp
    return out
