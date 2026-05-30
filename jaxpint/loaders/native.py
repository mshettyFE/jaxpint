"""Native (PINT-free) ``.tim`` -> :class:`~jaxpint.types.TOAData`.

Chains the native stages built across this effort:

    read_tim  ->  clock.correct  ->  clock.timescale.to_tdb
              ->  clock.posvels.compute_posvels  ->  TOAData

The result reproduces what ``bridge.pint_toas_to_jax(get_TOAs(...))`` produces,
without PINT.  ``flag_masks``, TZR, and troposphere are left empty/None for now
(separate follow-ups); wideband DM is read straight from the ``.tim`` flags.

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
from ..tim import read_tim
from ..types import TOAData


@dataclass
class _Core:
    """Parameter-independent time/geometry intermediate (the `.par`-free target)."""

    mjd_int: np.ndarray
    mjd_frac: np.ndarray
    tdb_int: np.ndarray
    tdb_frac: np.ndarray
    error_s: np.ndarray
    freq_mhz: np.ndarray          # topocentric
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


def _flag_floats(toas, key):
    """Per-TOA float value of a flag, or None if no TOA carries it."""
    vals = [t.flags.get(key) for t in toas]
    if all(v is None for v in vals):
        return None
    return np.array([float(v) if v is not None else np.nan for v in vals], dtype=np.float64)


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
            cfgs[tok].itrf_xyz if cfgs[tok].itrf_xyz is not None else (np.nan, np.nan, np.nan)
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
            tdb_int[idx_a], tdb_frac[idx_a], cfgs[tok].itrf_xyz,
            ephem=ephem, planets=planets,
        )
        ssb_obs_pos[idx_a] = pv["ssb_obs_pos"]
        ssb_obs_vel[idx_a] = pv["ssb_obs_vel"]
        obs_sun_pos[idx_a] = pv["obs_sun_pos"]
        if planets:
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
        mjd_int=corrected.mjd_int, mjd_frac=corrected.mjd_frac,
        tdb_int=tdb_int, tdb_frac=tdb_frac,
        error_s=error_s, freq_mhz=freq_mhz, delta_pulse_number=dpn,
        ssb_obs_pos=ssb_obs_pos, ssb_obs_vel=ssb_obs_vel, obs_sun_pos=obs_sun_pos,
        planet_positions=planet_positions, dm_values=dm_values, dm_errors=dm_errors,
        obs_names=uniq, obs_indices=obs_indices, n_toas=n,
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
        tim_path, ephem=ephem, include_bipm=include_bipm,
        bipm_version=bipm_version, planets=planets, limits=limits,
    )

    to_jnp = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)  # noqa: E731

    freq = core.freq_mhz
    if par_result is not None and _has_astrometry(par_result):
        toa_data_topo = _assemble(core, freq, to_jnp, planet_positions=core.planet_positions)
        ecl = Component.ASTROMETRY_ECLIPTIC in par_result.component_set
        freq = np.asarray(
            barycentric_radio_freq(
                toa_data_topo, par_result.params,
                ecliptic=ecl,
                pmra_name=_opt(par_result, "PMRA"), pmdec_name=_opt(par_result, "PMDEC"),
                pmelong_name=_opt(par_result, "PMELONG"), pmelat_name=_opt(par_result, "PMELAT"),
                posepoch_name=_opt(par_result, "POSEPOCH"),
                obliquity_arcsec=_obliquity(par_result),
            )
        )

    return _assemble(core, freq, to_jnp, planet_positions=core.planet_positions)


def _assemble(core, freq, to_jnp, *, planet_positions):
    jnp_planets = (
        None if planet_positions is None
        else {k: to_jnp(v) for k, v in planet_positions.items()}
    )
    return TOAData(
        mjd_int=to_jnp(core.mjd_int),
        mjd_frac=to_jnp(core.mjd_frac),
        tdb_int=to_jnp(core.tdb_int),
        tdb_frac=to_jnp(core.tdb_frac),
        error=to_jnp(core.error_s),
        freq=to_jnp(freq),
        delta_pulse_number=to_jnp(core.delta_pulse_number),
        flag_masks={},
        ssb_obs_pos=to_jnp(core.ssb_obs_pos),
        ssb_obs_vel=to_jnp(core.ssb_obs_vel),
        obs_sun_pos=to_jnp(core.obs_sun_pos),
        planet_positions=jnp_planets,
        dm_values=(None if core.dm_values is None else to_jnp(core.dm_values)),
        dm_errors=(None if core.dm_errors is None else to_jnp(core.dm_errors)),
        # Troposphere geometry is a deferred follow-up; left None (PINT also
        # leaves these None unless CORRECT_TROPOSPHERE is set).
        tropo_alt=None,
        tropo_alt_valid=None,
        obs_geodetic_lat=None,
        obs_height_km=None,
        n_toas=core.n_toas,
        obs_names=core.obs_names,
        obs_indices=jnp.asarray(core.obs_indices, dtype=jnp.int32),
    )


def _has_astrometry(par: ParResult) -> bool:
    return bool(
        {Component.ASTROMETRY_EQUATORIAL, Component.ASTROMETRY_ECLIPTIC}
        & par.component_set
    )


def _opt(par: ParResult, name: str) -> Optional[str]:
    return name if name in par.params.names else None


def _obliquity(par: ParResult) -> float:
    from ..constants import OBLIQUITY_ARCSEC

    ecl = par.metadata.get("ECL", "IERS2010")
    return float(OBLIQUITY_ARCSEC.get(ecl, OBLIQUITY_ARCSEC["IERS2010"]))
