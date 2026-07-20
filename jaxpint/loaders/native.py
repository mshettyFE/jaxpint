"""Native (PINT-free) ``.tim`` -> :class:`~jaxpint.types.TOAData`.

    read_tim  ->  clock.correct  ->  clock.timescale.to_tdb
              ->  clock.posvels.compute_posvels  ->  TOAData

A parsed ``.par`` (:class:`~jaxpint.par.result.ParResult`) is optional: it
supplies the barycentric-frequency astrometry direction and can override the run
config (ephemeris / BIPM / planets).  With no ``.par`` the frequency stays
topocentric -- matching PINT's no-model behaviour -- which is the clean,
parameter-independent comparison target (see :func:`topocentric_core`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..clock.correction import correct, resolve_clock_config
from ..clock.observatory import resolve_observatory
from ..clock.posvels import compute_posvels
from ..clock.timescale import to_tdb
from ..tropo_geometry import tropo_fields
from ..constants import PLANETS
from ..utils import barycentric_radio_freq
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
    """Build a :class:`TOAData` natively from a ``.tim`` (+ optional ``.par``).

    When *par_result* is supplied the returned TOAData is a *complete
    producer* of the GP basis time coordinate, mirroring the bridge: it
    carries barycentered TOAs in ``basis_seconds`` (``basis_coord =
    "barycentric"``, the enterprise/discovery convention), evaluated at the
    par-file parameter values.  Without a par there is no delay chain to
    evaluate, so the field stays unset and any GP-component build raises.
    """
    # Config: explicit args first, else from .par, else PINT-like defaults.
    md = par_result.metadata if par_result else {}
    bp = par_result.bool_params if par_result else {}
    if ephem is None:
        ephem = md.get("EPHEM", "DE440")
    if planets is None:
        planets = bool(bp.get("PLANET_SHAPIRO", False))
    # CLK/CLOCK selects the clock realization the same way EPHEM selects the
    # ephemeris above; hardcoding it silently overrode every par file (no local
    # corpus file asks for the old BIPM2023 default). A None version means
    # "packaged default", resolved inside correct().
    include_bipm, bipm_version = resolve_clock_config(
        md.get("CLOCK", md.get("CLK")),
        include_bipm=include_bipm,
        bipm_version=bipm_version,
    )

    core = topocentric_core(
        tim_path,
        ephem=ephem,
        include_bipm=include_bipm,
        bipm_version=bipm_version,
        planets=planets,
        limits=limits,
    )

    freq = core.freq_mhz
    if par_result is not None and _has_astrometry(par_result):
        toa_data_topo = _assemble(core, freq, planet_positions=core.planet_positions)
        freq = np.asarray(_barycentric_freq(toa_data_topo, par_result))

    flag_masks = _build_flag_masks(core, par_result)
    tzr = (
        None
        if par_result is None
        else _extract_tzr_fields(
            core,
            par_result,
            ephem=ephem,
            include_bipm=include_bipm,
            bipm_version=bipm_version,
            planets=planets,
        )
    )
    tropo = None if par_result is None else _build_tropo_fields(core, par_result)
    toa_data = _assemble(
        core,
        freq,
        planet_positions=core.planet_positions,
        flag_masks=flag_masks,
        tzr=tzr,
        tropo=tropo,
    )

    if par_result is not None:
        # GP basis time coordinate: barycentered TOAs (see the docstring).
        # Needs the delay chain, so build a TOA-independent timing model
        # (TOA-dependent noise components are skipped) and evaluate the
        # delays ahead of the binary component.  Stamped here — at
        # production — so every downstream build_model precomputes its noise
        # bases on the same coordinate the PTA injectors will use.
        from jaxpint.model_builder import build_model

        tm_only, _ = build_model(par_result, None)
        bary = tm_only.compute_barycentric_toas(toa_data, par_result.params)
        toa_data = toa_data.with_basis_seconds(
            bary.int * 86400.0 + bary.frac * 86400.0, "barycentric"
        )
    return toa_data


def _build_flag_masks(core, par_result: Optional[ParResult]) -> dict:
    """Boolean TOA masks for every masked parameter in ``par_result.mask_info``.

    An entry is produced for *every* masked parameter (not just those with
    matches): the noise/jump components access ``toa_data.flag_mask(name)``
    (no default), so a missing key is a KeyError.
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


def _assemble(core, freq, *, planet_positions, flag_masks=None, tzr=None, tropo=None):
    """Map the native ``_Core`` (+ optional tzr/tropo dicts) onto TOAData.

    Field-name mapping only; all dtype coercion lives in
    :meth:`TOAData.from_arrays`.  ``tropo`` is ``None`` unless CORRECT_TROPOSPHERE
    is set (matching PINT, which leaves those fields unset otherwise).
    """
    return TOAData.from_arrays(
        mjd_int=core.mjd_int,
        mjd_frac=core.mjd_frac,
        tdb_int=core.tdb_int,
        tdb_frac=core.tdb_frac,
        error=core.error_s,
        freq=freq,
        delta_pulse_number=core.delta_pulse_number,
        ssb_obs_pos=core.ssb_obs_pos,
        ssb_obs_vel=core.ssb_obs_vel,
        obs_sun_pos=core.obs_sun_pos,
        obs_indices=core.obs_indices,
        n_toas=core.n_toas,
        obs_names=core.obs_names,
        flag_masks=flag_masks,
        planet_positions=planet_positions,
        dm_values=core.dm_values,
        dm_errors=core.dm_errors,
        tropo_alt=None if tropo is None else tropo["tropo_alt"],
        tropo_alt_valid=None if tropo is None else tropo["tropo_alt_valid"],
        obs_geodetic_lat=None if tropo is None else tropo["obs_geodetic_lat"],
        obs_height_km=None if tropo is None else tropo["obs_height_km"],
        tzr_tdb_int=None if tzr is None else tzr["tzr_tdb_int"],
        tzr_tdb_frac=None if tzr is None else tzr["tzr_tdb_frac"],
        tzr_freq=None if tzr is None else tzr["tzr_freq"],
        tzr_ssb_obs_pos=None if tzr is None else tzr["tzr_ssb_obs_pos"],
        tzr_obs_sun_pos=None if tzr is None else tzr["tzr_obs_sun_pos"],
        tzr_planet_positions=None if tzr is None else tzr["tzr_planet_positions"],
    )


def _extract_tzr_fields(
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
        # Zero vectors, not None: an SSB-referenced TOA has no solar-system
        # delays, and r = 0 is the established "no delay" convention
        planet_positions = (
            {f"obs_{p}_pos": np.zeros(3, dtype=np.float64) for p in PLANETS}
            if planets
            else None
        )
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
        tzr_topo = TOAData.single(
            tdb_int=tdb_int,
            tdb_frac=tdb_frac,
            freq=tzr_frq,
            ssb_obs_pos=ssb_obs_pos,
            ssb_obs_vel=ssb_obs_vel,
            obs_sun_pos=obs_sun_pos,
        )
        tzr_freq = float(np.asarray(_barycentric_freq(tzr_topo, par_result))[0])

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
    from astropy.coordinates import EarthLocation, SkyCoord

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

    groups: dict[str, list[int]] = {}
    for i, name in enumerate(core.obs_canonical):
        groups.setdefault(name, []).append(i)

    # Resolve each topocentric observatory to an EarthLocation (native's
    # source-specific part); tropo_fields owns the shared AltAz + clamp.
    obs_groups: dict[str, tuple] = {}
    for name, idx in groups.items():
        cfg = resolve_observatory(name)
        if cfg.itrf_xyz is None:  # non-topocentric (barycenter/geocenter): skip
            continue
        xyz = np.asarray(cfg.itrf_xyz, dtype=np.float64)
        loc = EarthLocation.from_geocentric(xyz[0] * u.m, xyz[1] * u.m, xyz[2] * u.m)
        obs_groups[name] = (loc, np.asarray(idx))

    mjd = core.mjd_int + core.mjd_frac
    return tropo_fields(radec, obs_groups, mjd, core.n_toas)


def _has_astrometry(par: ParResult) -> bool:
    return bool(
        {Component.ASTROMETRY_EQUATORIAL, Component.ASTROMETRY_ECLIPTIC}
        & par.component_set
    )


def _opt(par: ParResult, name: str) -> Optional[str]:
    return name if name in par.params else None


def _posepoch(par: ParResult) -> Optional[str]:
    """Proper-motion reference epoch: POSEPOCH, else PEPOCH (PINT convention)."""
    if "POSEPOCH" in par.params:
        return "POSEPOCH"
    if "PEPOCH" in par.params:
        return "PEPOCH"
    return None


def _obliquity(par: ParResult) -> float:
    from ..constants import OBLIQUITY_ARCSEC

    ecl = par.metadata.get("ECL", "IERS2010")
    return float(OBLIQUITY_ARCSEC.get(ecl, OBLIQUITY_ARCSEC["IERS2010"]))


def _barycentric_freq(toa_data: TOAData, par: ParResult):
    """Doppler-corrected barycentric radio frequency from the .par astrometry.

    Resolves the astrometry parameter names (equatorial vs ecliptic, proper
    motion, POSEPOCH, obliquity) from *par* and applies the Doppler factor to
    the topocentric ``toa_data.freq`` -- the one place the loader needs them.
    """
    return barycentric_radio_freq(
        toa_data,
        par.params,
        ecliptic=Component.ASTROMETRY_ECLIPTIC in par.component_set,
        pmra_name=_opt(par, "PMRA"),
        pmdec_name=_opt(par, "PMDEC"),
        pmelong_name=_opt(par, "PMELONG"),
        pmelat_name=_opt(par, "PMELAT"),
        posepoch_name=_posepoch(par),
        obliquity_arcsec=_obliquity(par),
    )
