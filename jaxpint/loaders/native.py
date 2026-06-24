"""Native (PINT-free) ``.tim`` -> :class:`~jaxpint.types.TOAData`.

Chains the native stages built across this effort:

    read_tim  ->  clock.correct  ->  clock.timescale.to_tdb
              ->  clock.posvels.compute_posvels  ->  TOAData

The result reproduces what ``bridge.pint_toas_to_jax(get_TOAs(...))`` produces,
without PINT, including ``flag_masks`` (per masked-parameter TOA selection), the
TZR absolute-phase anchor, and troposphere geometry when a ``.par`` is supplied;
wideband DM is read straight from the ``.tim`` flags.

A parsed ``.par`` (:class:`~jaxpint.par.result.ParResult`) is optional: it
supplies the barycentric-frequency astrometry direction and can override the run
config (ephemeris / BIPM / planets).  With no ``.par`` the frequency stays
topocentric -- matching PINT's no-model behaviour -- which is the clean,
parameter-independent comparison target (see :func:`topocentric_core`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ..clock.correction import correct
from ..clock.observatory import resolve_observatory
from ..clock.posvels import compute_posvels
from ..clock.timescale import to_tdb
from ..constants import PLANETS
from ..delay._barycentric import barycentric_radio_freq
from ..par.components import Component
from ..par.result import ParResult
from ..tim import RawTOA, read_tim, select_toa_mask
from ..types import TOAData


@dataclass
class _Core:
    """Parameter-independent time/geometry intermediate (the `.par`-free target)."""

    mjd_int: np.ndarray
    mjd_frac: np.ndarray
    tdb_int: np.ndarray
    tdb_frac: np.ndarray
    error_s: np.ndarray
    freq_mhz: np.ndarray  # topocentric
    delta_pulse_number: np.ndarray
    ssb_obs_pos: np.ndarray
    ssb_obs_vel: np.ndarray
    obs_sun_pos: np.ndarray
    planet_positions: Optional[dict]
    dm_values: Optional[np.ndarray]
    dm_errors: Optional[np.ndarray]
    obs_names: tuple
    obs_indices: np.ndarray
    n_toas: int
    # For native flag-mask matching (select_toa_mask): the parsed TOAs and their
    # per-TOA canonical observatory names.
    raw_toas: list
    obs_canonical: list


def _flag_floats(toas, key):
    """Per-TOA float value of a flag, or None if no TOA carries it."""
    vals = [t.flags.get(key) for t in toas]
    if all(v is None for v in vals):
        return None
    return np.array(
        [float(v) if v is not None else np.nan for v in vals], dtype=np.float64
    )


def topocentric_core(
    tim_path,
    *,
    ephem: str = "DE440",
    include_bipm: bool = True,
    bipm_version: str = "BIPM2023",
    planets: bool = False,
    limits: str = "warn",
) -> _Core:
    """Build the parameter-independent time/geometry core from a ``.tim`` file.

    This is exactly the content of PINT's ``TOAs`` table (topocentric freq +
    TDB + barycentric posvels), so it is diffable against PINT with no model.
    """
    toas = read_tim(tim_path).toas
    n = len(toas)

    # 1. clock corrections -> TT(BIPM) MJD
    corrected = correct(
        toas, include_bipm=include_bipm, bipm_version=bipm_version, limits=limits
    )

    # 2. per-TOA observatory ITRF xyz (for TDB topo term + posvels)
    obs_tokens = [t.obs for t in toas]
    cfgs = {tok: resolve_observatory(tok) for tok in set(obs_tokens)}
    xyz_rows = np.array(
        [
            cfgs[tok].itrf_xyz
            if cfgs[tok].itrf_xyz is not None
            else (np.nan, np.nan, np.nan)
            for tok in obs_tokens
        ],
        dtype=np.float64,
    )

    # 3. TT(BIPM) -> TDB
    tdb_int, tdb_frac = to_tdb(corrected.mjd_int, corrected.mjd_frac, xyz_rows)

    # 4. posvels, per observatory group (each group shares one xyz / barycenter)
    ssb_obs_pos = np.zeros((n, 3))
    ssb_obs_vel = np.zeros((n, 3))
    obs_sun_pos = np.zeros((n, 3))
    planet_positions = (
        {f"obs_{p}_pos": np.zeros((n, 3)) for p in PLANETS} if planets else None
    )
    groups: dict[str, list[int]] = {}
    for i, tok in enumerate(obs_tokens):
        groups.setdefault(tok, []).append(i)
    for tok, idx in groups.items():
        idx_a = np.array(idx)
        pv = compute_posvels(
            tdb_int[idx_a],
            tdb_frac[idx_a],
            cfgs[tok].itrf_xyz,
            ephem=ephem,
            planets=planets,
        )
        ssb_obs_pos[idx_a] = pv["ssb_obs_pos"]
        ssb_obs_vel[idx_a] = pv["ssb_obs_vel"]
        obs_sun_pos[idx_a] = pv["obs_sun_pos"]
        if planets:
            assert planet_positions is not None
            for k, v in pv["planet_positions"].items():
                planet_positions[k][idx_a] = v

    # 5. raw per-TOA scalars + wideband DM from flags
    error_s = np.array([t.error_s for t in toas], dtype=np.float64)
    freq_mhz = np.array([t.freq_mhz for t in toas], dtype=np.float64)
    dpn = np.array([t.delta_pulse_number for t in toas], dtype=np.float64)
    dm_values = _flag_floats(toas, "pp_dm")
    dm_errors = _flag_floats(toas, "pp_dme")

    # observatory canonical names + integer indices
    obs_canon = [cfgs[tok].canonical for tok in obs_tokens]
    uniq = tuple(dict.fromkeys(obs_canon))
    idx_of = {name: i for i, name in enumerate(uniq)}
    obs_indices = np.array([idx_of[c] for c in obs_canon], dtype=np.int32)

    return _Core(
        mjd_int=corrected.mjd_int,
        mjd_frac=corrected.mjd_frac,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error_s=error_s,
        freq_mhz=freq_mhz,
        delta_pulse_number=dpn,
        ssb_obs_pos=ssb_obs_pos,
        ssb_obs_vel=ssb_obs_vel,
        obs_sun_pos=obs_sun_pos,
        planet_positions=planet_positions,
        dm_values=dm_values,
        dm_errors=dm_errors,
        obs_names=uniq,
        obs_indices=obs_indices,
        n_toas=n,
        raw_toas=toas,
        obs_canonical=obs_canon,
    )


def native_toas_to_jax(
    tim_path,
    par_result: Optional[ParResult] = None,
    *,
    ephem: Optional[str] = None,
    include_bipm: Optional[bool] = None,
    bipm_version: Optional[str] = None,
    planets: Optional[bool] = None,
    limits: str = "warn",
) -> TOAData:
    """Build a :class:`TOAData` natively from a ``.tim`` (+ optional ``.par``)."""
    # Config: explicit args win, else from .par, else PINT-like defaults.
    md = par_result.metadata if par_result else {}
    bp = par_result.bool_params if par_result else {}
    if ephem is None:
        ephem = md.get("EPHEM", "DE440")
    if planets is None:
        planets = bool(bp.get("PLANET_SHAPIRO", False))
    if include_bipm is None:
        include_bipm = True
    if bipm_version is None:
        bipm_version = "BIPM2023"

    core = topocentric_core(
        tim_path,
        ephem=ephem,
        include_bipm=include_bipm,
        bipm_version=bipm_version,
        planets=planets,
        limits=limits,
    )

    to_jnp = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)  # noqa: E731

    freq = core.freq_mhz
    if par_result is not None and _has_astrometry(par_result):
        toa_data_topo = _assemble(
            core, freq, to_jnp, planet_positions=core.planet_positions
        )
        ecl = Component.ASTROMETRY_ECLIPTIC in par_result.component_set
        freq = np.asarray(
            barycentric_radio_freq(
                toa_data_topo,
                par_result.params,
                ecliptic=ecl,
                pmra_name=_opt(par_result, "PMRA"),
                pmdec_name=_opt(par_result, "PMDEC"),
                pmelong_name=_opt(par_result, "PMELONG"),
                pmelat_name=_opt(par_result, "PMELAT"),
                posepoch_name=_posepoch(par_result),
                obliquity_arcsec=_obliquity(par_result),
            )
        )

    flag_masks = _build_flag_masks(core, par_result)
    tzr = (
        None
        if par_result is None
        else _build_tzr_fields(
            core,
            par_result,
            ephem=ephem,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            planets=planets,
        )
    )
    tropo = None if par_result is None else _build_tropo_fields(core, par_result)
    return _assemble(
        core,
        freq,
        to_jnp,
        planet_positions=core.planet_positions,
        flag_masks=flag_masks,
        tzr=tzr,
        tropo=tropo,
    )


def _build_flag_masks(core, par_result: Optional[ParResult]) -> dict:
    """Boolean TOA masks for every masked parameter in ``par_result.mask_info``.

    An entry is produced for *every* masked parameter (not just those with
    matches): the noise/jump components index ``flag_masks[name]`` directly, so a
    missing key is a KeyError.
    """
    if par_result is None or not par_result.mask_info:
        return {}
    mjd_corrected = core.mjd_int + core.mjd_frac
    return {
        name: select_toa_mask(
            info,
            core.raw_toas,
            obs_canonical=core.obs_canonical,
            mjd_corrected=mjd_corrected,
        )
        for name, info in par_result.mask_info.items()
    }


def _assemble(
    core, freq, to_jnp, *, planet_positions, flag_masks=None, tzr=None, tropo=None
):
    jnp_planets = (
        None
        if planet_positions is None
        else {k: to_jnp(v) for k, v in planet_positions.items()}
    )
    tzr_planets = (
        None
        if tzr is None or tzr["tzr_planet_positions"] is None
        else {k: to_jnp(v) for k, v in tzr["tzr_planet_positions"].items()}
    )
    return TOAData(
        mjd_int=to_jnp(core.mjd_int),
        mjd_frac=to_jnp(core.mjd_frac),
        tdb_int=to_jnp(core.tdb_int),
        tdb_frac=to_jnp(core.tdb_frac),
        error=to_jnp(core.error_s),
        freq=to_jnp(freq),
        delta_pulse_number=to_jnp(core.delta_pulse_number),
        flag_masks=(
            {}
            if not flag_masks
            else {k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()}
        ),
        ssb_obs_pos=to_jnp(core.ssb_obs_pos),
        ssb_obs_vel=to_jnp(core.ssb_obs_vel),
        obs_sun_pos=to_jnp(core.obs_sun_pos),
        planet_positions=jnp_planets,
        dm_values=(None if core.dm_values is None else to_jnp(core.dm_values)),
        dm_errors=(None if core.dm_errors is None else to_jnp(core.dm_errors)),
        # Troposphere geometry: None unless CORRECT_TROPOSPHERE is set (matches
        # PINT, which leaves these unset otherwise).
        tropo_alt=(None if tropo is None else to_jnp(tropo["tropo_alt"])),
        tropo_alt_valid=(
            None
            if tropo is None
            else jnp.asarray(tropo["tropo_alt_valid"], dtype=jnp.bool_)
        ),
        obs_geodetic_lat=(None if tropo is None else to_jnp(tropo["obs_geodetic_lat"])),
        obs_height_km=(None if tropo is None else to_jnp(tropo["obs_height_km"])),
        n_toas=core.n_toas,
        obs_names=core.obs_names,
        obs_indices=jnp.asarray(core.obs_indices, dtype=jnp.int32),
        tzr_tdb_int=(None if tzr is None else tzr["tzr_tdb_int"]),
        tzr_tdb_frac=(None if tzr is None else tzr["tzr_tdb_frac"]),
        tzr_freq=(None if tzr is None else tzr["tzr_freq"]),
        tzr_ssb_obs_pos=(None if tzr is None else to_jnp(tzr["tzr_ssb_obs_pos"])),
        tzr_obs_sun_pos=(None if tzr is None else to_jnp(tzr["tzr_obs_sun_pos"])),
        tzr_planet_positions=tzr_planets,
    )


def _build_tzr_fields(
    core,
    par_result: ParResult,
    *,
    ephem: str,
    include_bipm: bool,
    bipm_version: str,
    planets: bool,
) -> Optional[dict]:
    """Synthesize the TZR (time-zero reference) TOA and return its ``tzr_*`` data.

    Mirrors the PINT bridge's ``extract_tzr_toa`` but builds the single TZR TOA
    from the parsed ``.par`` (TZRMJD/TZRSITE/TZRFRQ, or auto from PEPOCH) and runs
    it through the native clock -> TDB -> posvel chain.  Returns ``None`` when no
    absolute-phase anchor can be determined (no TZRMJD and no PEPOCH), in which
    case ``compute_phase`` leaves the phase un-subtracted.
    """
    names = par_result.params.names

    # --- TZRMJD (raw site MJD int/frac to feed the chain) -------------------
    if "TZRMJD" in names:
        tzr_int, tzr_frac = par_result.params.epoch_value("TZRMJD")
        tzr_int = float(tzr_int)
        tzr_frac = float(np.asarray(tzr_frac))
    elif "PEPOCH" in names:
        # Auto, mirroring PINT make_TZR_toa: first corrected MJD after PEPOCH,
        # else max <= PEPOCH (the chosen corrected value is then re-corrected
        # through the chain below, exactly as PINT does).
        pe_int, pe_frac = par_result.params.epoch_value("PEPOCH")
        pepoch = float(pe_int) + float(np.asarray(pe_frac))
        cand = core.mjd_int + core.mjd_frac
        later = cand[cand > pepoch]
        chosen = float(later.min()) if later.size else float(cand.max())
        tzr_int = float(np.floor(chosen))
        tzr_frac = chosen - tzr_int
    else:
        return None

    # --- site + frequency ---------------------------------------------------
    tzr_site = par_result.metadata.get("TZRSITE", "ssb")
    cfg = resolve_observatory(tzr_site)
    raw_frq = par_result.metadata.get("TZRFRQ")
    if raw_frq is None or str(raw_frq).strip().lower() == "inf":
        tzr_frq = np.inf
    else:
        f = float(raw_frq)
        tzr_frq = np.inf if f == 0.0 else f

    is_bary = cfg.itrf_xyz is None or cfg.canonical == "barycenter"
    if is_bary:
        # TZRMJD is already TDB at the barycentre; no clock / TT->TDB step.
        tdb_int, tdb_frac = tzr_int, tzr_frac
        ssb_obs_pos = np.zeros(3, dtype=np.float64)
        ssb_obs_vel = np.zeros(3, dtype=np.float64)
        obs_sun_pos = np.zeros(3, dtype=np.float64)
        planet_positions = None
    else:
        raw = RawTOA(
            mjd_int=tzr_int,
            mjd_frac=tzr_frac,
            error_s=1.0,
            freq_mhz=tzr_frq,
            obs=tzr_site,
            flags={},
        )
        corrected = correct(
            [raw],
            include_bipm=include_bipm,
            bipm_version=bipm_version,
        )
        xyz = np.array([cfg.itrf_xyz], dtype=np.float64)
        tdb_int_a, tdb_frac_a = to_tdb(corrected.mjd_int, corrected.mjd_frac, xyz)
        pv = compute_posvels(
            tdb_int_a,
            tdb_frac_a,
            cfg.itrf_xyz,
            ephem=ephem,
            planets=planets,
        )
        tdb_int = float(tdb_int_a[0])
        tdb_frac = float(tdb_frac_a[0])
        ssb_obs_pos = np.asarray(pv["ssb_obs_pos"][0], dtype=np.float64)
        ssb_obs_vel = np.asarray(pv["ssb_obs_vel"][0], dtype=np.float64)
        obs_sun_pos = np.asarray(pv["obs_sun_pos"][0], dtype=np.float64)
        planet_positions = (
            {
                k: np.asarray(v[0], dtype=np.float64)
                for k, v in pv["planet_positions"].items()
            }
            if planets
            else None
        )

    # --- barycentric TZR frequency (Doppler applied once, at build) ---------
    tzr_freq = tzr_frq
    if _has_astrometry(par_result) and np.isfinite(tzr_frq):
        to_jnp = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)  # noqa: E731
        tzr_topo = TOAData(
            mjd_int=to_jnp([tdb_int]),
            mjd_frac=to_jnp([tdb_frac]),
            tdb_int=to_jnp([tdb_int]),
            tdb_frac=to_jnp([tdb_frac]),
            error=to_jnp([1.0]),
            freq=to_jnp([tzr_frq]),
            delta_pulse_number=to_jnp([0.0]),
            flag_masks={},
            ssb_obs_pos=to_jnp(ssb_obs_pos[None, :]),
            ssb_obs_vel=to_jnp(ssb_obs_vel[None, :]),
            obs_sun_pos=to_jnp(obs_sun_pos[None, :]),
            planet_positions=None,
            dm_values=None,
            dm_errors=None,
            tropo_alt=None,
            tropo_alt_valid=None,
            obs_geodetic_lat=None,
            obs_height_km=None,
            n_toas=1,
            obs_names=("",),
            obs_indices=jnp.zeros(1, dtype=jnp.int32),
        )
        ecl = Component.ASTROMETRY_ECLIPTIC in par_result.component_set
        tzr_freq = float(
            np.asarray(
                barycentric_radio_freq(
                    tzr_topo,
                    par_result.params,
                    ecliptic=ecl,
                    pmra_name=_opt(par_result, "PMRA"),
                    pmdec_name=_opt(par_result, "PMDEC"),
                    pmelong_name=_opt(par_result, "PMELONG"),
                    pmelat_name=_opt(par_result, "PMELAT"),
                    posepoch_name=_posepoch(par_result),
                    obliquity_arcsec=_obliquity(par_result),
                )
            )[0]
        )

    return {
        "tzr_tdb_int": float(tdb_int),
        "tzr_tdb_frac": float(tdb_frac),
        "tzr_freq": float(tzr_freq),
        "tzr_ssb_obs_pos": ssb_obs_pos,
        "tzr_obs_sun_pos": obs_sun_pos,
        "tzr_planet_positions": planet_positions,
    }


def _build_tropo_fields(core, par_result: ParResult) -> Optional[dict]:
    """Per-TOA troposphere geometry, or ``None`` when the correction is off.

    Mirrors the PINT bridge's troposphere block: gated on
    ``TroposphereDelay`` + ``CORRECT_TROPOSPHERE``; computes the pulsar's
    elevation angle (astropy ``AltAz``) at each TOA plus the observatory geodetic
    latitude/height.  ``radec`` is a *fixed* sky position (no proper motion),
    matching PINT's ``_get_target_skycoord``.
    """
    if Component.TROPOSPHERE_DELAY not in par_result.component_set:
        return None
    if not par_result.bool_params.get("CORRECT_TROPOSPHERE", False):
        return None

    from astropy import units as u
    from astropy.coordinates import AltAz, EarthLocation, SkyCoord
    from astropy.time import Time

    params = par_result.params
    if Component.ASTROMETRY_ECLIPTIC in par_result.component_set:
        radec = SkyCoord(
            float(params.param_value("ELONG")) * u.rad,
            float(params.param_value("ELAT")) * u.rad,
            frame="barycentricmeanecliptic",
        )
    else:
        radec = SkyCoord(
            float(params.param_value("RAJ")) * u.rad,
            float(params.param_value("DECJ")) * u.rad,
        )

    n = core.n_toas
    alt = np.zeros(n, dtype=np.float64)
    lat = np.zeros(n, dtype=np.float64)
    height = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)

    mjd = core.mjd_int + core.mjd_frac
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(core.obs_canonical):
        groups.setdefault(name, []).append(i)

    for name, idx in groups.items():
        cfg = resolve_observatory(name)
        if cfg.itrf_xyz is None:  # non-topocentric (barycenter/geocenter): skip
            continue
        xyz = np.asarray(cfg.itrf_xyz, dtype=np.float64)
        loc = EarthLocation.from_geocentric(xyz[0] * u.m, xyz[1] * u.m, xyz[2] * u.m)
        obstime = Time(mjd[idx], format="mjd", scale="utc")
        # astropy SkyCoord / transform_to are under-typed (return Optional in stubs).
        a = radec.transform_to(AltAz(location=loc, obstime=obstime)).alt.to_value(u.rad)  # pyright: ignore[reportCallIssue, reportOptionalCall, reportOptionalMemberAccess]
        idx_a = np.asarray(idx)
        alt[idx_a] = a
        lat[idx_a] = loc.lat.to_value(u.rad)
        height[idx_a] = loc.height.to_value(u.km)
        valid[idx_a] = True

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


def _has_astrometry(par: ParResult) -> bool:
    return bool(
        {Component.ASTROMETRY_EQUATORIAL, Component.ASTROMETRY_ECLIPTIC}
        & par.component_set
    )


def _opt(par: ParResult, name: str) -> Optional[str]:
    return name if name in par.params.names else None


def _posepoch(par: ParResult) -> Optional[str]:
    """Proper-motion reference epoch: POSEPOCH, else PEPOCH (PINT convention)."""
    if "POSEPOCH" in par.params.names:
        return "POSEPOCH"
    if "PEPOCH" in par.params.names:
        return "PEPOCH"
    return None


def _obliquity(par: ParResult) -> float:
    from ..constants import OBLIQUITY_ARCSEC

    ecl = par.metadata.get("ECL", "IERS2010")
    return float(OBLIQUITY_ARCSEC.get(ecl, OBLIQUITY_ARCSEC["IERS2010"]))
