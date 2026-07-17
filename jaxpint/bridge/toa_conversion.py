"""TOA conversion: PINT TOAs → JaxPINT TOAData.

Converts PINT TOA tables into JAX-native float64 arrays following the
conventions documented in :class:`jaxpint.types.TOAData`.  All unit
conversion and validation happens here.
"""

from __future__ import annotations

import logging
from typing import Optional

import astropy.units as u
import numpy as np
from pint.models.parameter import maskParameter
from pint.models.timing_model import TimingModel as PINTTimingModel
from pint.observatory import get_observatory
from pint.observatory.topo_obs import TopoObs
from pint.toa import TOAs

from jaxpint.constants import JD_MJD_OFFSET, PLANETS
from jaxpint.utils import split_longdouble_days
from jaxpint.types import TOAData

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _check_column_unit(table, colname: str, expected_unit) -> None:
    """Assert a table column has the expected physical dimension.

    Raises ``astropy.units.UnitConversionError`` if the column's unit is
    not convertible to *expected_unit*.  Silently passes if the column has
    no unit metadata (some PINT columns are plain numpy arrays).
    """
    col = table[colname]
    if hasattr(col, "unit") and col.unit is not None:
        # .to() raises UnitConversionError if dimensions don't match
        col.unit.to(expected_unit)


def _split_mjd_time(
    time_col,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an astropy Time into float64 ``(int_day, frac_day)``.

    Uses the internal ``jd1`` / ``jd2`` representation for maximum precision.
    ``jd1`` is typically a half-integer (e.g. 2459000.5), so
    ``jd1 - 2400000.5`` is the integer MJD day.

    *time_col* may be a scalar ``Time``, a vectorized ``Time`` array, or an
    astropy ``Column`` of individual ``Time`` objects (as stored in a PINT
    TOA table).  For scalar inputs the returned arrays are 0-d.
    """
    from astropy.time import Time

    # PINT stores Time objects per-row in an object Column; coalesce.
    if not isinstance(time_col, Time):
        time_col = Time(list(time_col))

    jd1 = np.asarray(time_col.jd1, dtype=np.float64)
    jd2 = np.asarray(time_col.jd2, dtype=np.float64)
    # Convert JD pair to MJD pair: MJD = (jd1 - 2400000.5) + jd2
    # Compute combined MJD, then split into integer day + fraction in [0, 1)
    mjd1 = jd1 - JD_MJD_OFFSET
    full_mjd = mjd1 + jd2
    mjd_int = np.floor(full_mjd)
    mjd_frac = full_mjd - mjd_int
    return mjd_int, mjd_frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_tzr_toa(
    model: PINTTimingModel,
    toas: TOAs,
) -> dict:
    """Extract the TZR TOA data from PINT's AbsPhase component.

    If the model does not already have an AbsPhase component, one is
    auto-generated from the TOAs (first TOA after PEPOCH), matching
    PINT's guarantee in ``timing_model.phase()``.

    Parameters
    ----------
    model : pint.models.TimingModel
        PINT timing model, which should contain an ``AbsPhase`` component
        (or one will be auto-generated).
    toas : pint.toa.TOAs
        The TOA set used to generate the TZR TOA if ``AbsPhase`` is absent.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``tdb_int`` (float) -- Integer MJD day of the TZR TOA in TDB.
        - ``tdb_frac`` (float) -- Fractional MJD day of the TZR TOA in TDB.
        - ``freq`` (float) -- Observing frequency in MHz.
        - ``ssb_obs_pos`` (numpy.ndarray, shape (3,)) -- SSB observer
          position in km.
        - ``obs_sun_pos`` (numpy.ndarray, shape (3,)) -- Observer-to-Sun
          position in km (zeros for barycentric observations).
    """
    if "AbsPhase" not in model.components:
        log.info("No AbsPhase in model; auto-generating TZR TOA from TOAs.")
        model.add_tzr_toa(toas)

    abs_phase = model.components["AbsPhase"]
    tz_toas = abs_phase.get_TZR_toa(toas)

    # Force planet position columns on the TZR TOA whenever the model needs them.
    need_planets = "PLANET_SHAPIRO" in model.params and bool(
        getattr(model.PLANET_SHAPIRO, "value", False)
    )
    if need_planets and not any(
        f"obs_{p}_pos" in tz_toas.table.colnames for p in PLANETS
    ):
        tz_toas.compute_posvels(planets=True)

    tz_tbl = tz_toas.table

    tdb_int, tdb_frac = split_longdouble_days(np.asarray(tz_tbl["tdbld"]))

    try:
        freq = float(model.barycentric_radio_freq(tz_toas).to(u.MHz).value[0])
    except AttributeError:
        log.warning(
            "Model has no barycentric_radio_freq; using topocentric TZR frequency"
        )
        freq = float(tz_toas.get_freqs().to(u.MHz).value[0])

    ssb_obs_pos = np.asarray(tz_tbl["ssb_obs_pos"], dtype=np.float64)[0]

    # PINT skips Shapiro delay for barycentered TOAs (obs == "barycenter").
    # Mirror this by setting obs_sun_pos to zeros for barycentric TZR,
    # which causes the Shapiro delay guard to return 0.
    is_bary = np.all(tz_toas.get_obss() == "barycenter")
    if is_bary:
        obs_sun_pos = np.zeros(3, dtype=np.float64)
    else:
        obs_sun_pos = np.asarray(tz_tbl["obs_sun_pos"], dtype=np.float64)[0]

    # Copy over planet positions from PINT TOA if present
    tz_planet_positions: Optional[dict[str, np.ndarray]] = None
    for planet in PLANETS:
        col_name = f"obs_{planet}_pos"
        if col_name in tz_tbl.colnames:
            if tz_planet_positions is None:
                tz_planet_positions = {}
            _check_column_unit(tz_tbl, col_name, u.km)
            tz_planet_positions[col_name] = np.asarray(
                tz_tbl[col_name], dtype=np.float64
            )[0]

    return {
        "tdb_int": float(tdb_int[0]),
        "tdb_frac": float(tdb_frac[0]),
        "freq": freq,
        "ssb_obs_pos": ssb_obs_pos,
        "obs_sun_pos": obs_sun_pos,
        "planet_positions": tz_planet_positions,
    }


def pint_toas_to_jax(
    toas: TOAs,
    model: Optional[PINTTimingModel] = None,
) -> TOAData:
    """Convert PINT TOAs to a JaxPINT :class:`TOAData`.

    All unit conversion and validation happens here.  After this call
    everything is raw float64 JAX arrays following the conventions
    documented in :class:`jaxpint.types.TOAData`.

    Parameters
    ----------
    toas : pint.toa.TOAs
        Must already have ``compute_TDBs()`` and ``compute_posvels()``
        called (or this function calls them).
    model : pint.models.TimingModel, optional
        If provided, pre-computes boolean flag masks for all
        ``maskParameter`` instances (JUMP, EFAC, EQUAD, DMX, etc.)
        and extracts TZR TOA data for absolute phase computation.

    Returns
    -------
    TOAData
        A frozen container of JAX float64 arrays holding MJD times,
        TDB times, uncertainties, frequencies, SSB positions/velocities,
        observatory indices, flag masks, and optional fields (planet
        positions, wideband DM, troposphere data, TZR TOA).
    """
    n_toas = toas.ntoas

    # PINT's get_TOAs doesn't always populate planet positions  even when planets=True is passed
    # Force compute_posvels(planets=True) whenever the model needs it.
    need_planets = (
        model is not None
        and "PLANET_SHAPIRO" in model.params
        and bool(getattr(model.PLANET_SHAPIRO, "value", False))
    )

    # -- Ensure computed columns exist -----------------------------------
    if "tdbld" not in toas.table.colnames:
        log.info("Computing TDBs (not yet present on TOAs)")
        toas.compute_TDBs()
    if "ssb_obs_pos" not in toas.table.colnames:
        log.info("Computing posvels (not yet present on TOAs)")
        toas.compute_posvels(planets=need_planets)
    elif need_planets and not any(
        f"obs_{p}_pos" in toas.table.colnames for p in PLANETS
    ):
        log.info("Recomputing posvels with planets=True (model has PLANET_SHAPIRO Y)")
        toas.compute_posvels(planets=True)

    tbl = toas.table

    # -- MJD split (UTC) -------------------------------------------------
    mjd_int, mjd_frac = _split_mjd_time(tbl["mjd"])

    # -- TDB split -------------------------------------------------------
    tdb_int, tdb_frac = split_longdouble_days(np.asarray(tbl["tdbld"]))

    # -- Unit-validated scalar columns -----------------------------------
    error_s = toas.get_errors().to(u.s).value
    if model is not None:
        try:
            freq_mhz = np.asarray(
                model.barycentric_radio_freq(toas).to(u.MHz).value,
                dtype=np.float64,
            )
        except AttributeError:
            log.warning(
                "Model has no barycentric_radio_freq; using topocentric frequency"
            )
            freq_mhz = toas.get_freqs().to(u.MHz).value
    else:
        freq_mhz = toas.get_freqs().to(u.MHz).value

    # -- Position / velocity (validate units, then extract) --------------
    _check_column_unit(tbl, "ssb_obs_pos", u.km)
    _check_column_unit(tbl, "ssb_obs_vel", u.km / u.s)
    _check_column_unit(tbl, "obs_sun_pos", u.km)

    ssb_obs_pos = np.asarray(tbl["ssb_obs_pos"], dtype=np.float64)
    ssb_obs_vel = np.asarray(tbl["ssb_obs_vel"], dtype=np.float64)
    obs_sun_pos = np.asarray(tbl["obs_sun_pos"], dtype=np.float64)

    # -- Delta pulse number ----------------------------------------------
    delta_pulse_number = np.asarray(tbl["delta_pulse_number"], dtype=np.float64)

    # -- Observatory indices ---------------------------------------------
    obs_array = np.asarray(toas.get_obss(), dtype=str)
    obs_names = tuple(str(s) for s in sorted(set(obs_array)))
    obs_name_to_idx = {name: i for i, name in enumerate(obs_names)}
    obs_indices = np.array([obs_name_to_idx[o] for o in obs_array], dtype=np.int32)

    # -- Flag masks (requires model) -------------------------------------
    flag_masks: dict[str, np.ndarray] = {}
    if model is not None:
        for pname in model.params:
            param = getattr(model, pname)
            if isinstance(param, maskParameter):
                idx = param.select_toa_mask(toas)
                mask = np.zeros(n_toas, dtype=bool)
                if len(idx) > 0:
                    mask[idx] = True
                flag_masks[pname] = mask

    # -- Optional planet positions ---------------------------------------
    planet_positions: Optional[dict[str, np.ndarray]] = None
    for planet in PLANETS:
        col_name = f"obs_{planet}_pos"
        if col_name in tbl.colnames:
            if planet_positions is None:
                planet_positions = {}
            _check_column_unit(tbl, col_name, u.km)
            planet_positions[col_name] = np.asarray(tbl[col_name], dtype=np.float64)

    # -- Optional wideband DM --------------------------------------------
    dm_values: Optional[np.ndarray] = None
    dm_errors: Optional[np.ndarray] = None
    if toas.is_wideband():
        dm_values = toas.get_dms().to(u.pc / u.cm**3).value
        dm_errors = toas.get_dm_errors().to(u.pc / u.cm**3).value

    # -- Optional troposphere data ------------------------------------------
    tropo_alt: Optional[np.ndarray] = None
    tropo_alt_valid: Optional[np.ndarray] = None
    obs_geodetic_lat: Optional[np.ndarray] = None
    obs_height_km: Optional[np.ndarray] = None

    if model is not None and "TroposphereDelay" in model.components:
        tropo_comp = model.components["TroposphereDelay"]
        if tropo_comp.CORRECT_TROPOSPHERE.value:
            radec = tropo_comp._get_target_skycoord()

            alt_arr = np.zeros(n_toas, dtype=np.float64)
            lat_arr = np.zeros(n_toas, dtype=np.float64)
            height_arr = np.zeros(n_toas, dtype=np.float64)
            valid_arr = np.zeros(n_toas, dtype=bool)

            for key, grp in toas.get_obs_groups():
                obsobj = get_observatory(key)
                if not isinstance(obsobj, TopoObs):
                    # Non-topocentric: leave as zeros, valid=False
                    continue

                obs = obsobj.earth_location_itrf()
                alt = tropo_comp._get_target_altitude(obs, tbl[grp], radec)

                alt_arr[grp] = alt.to(u.rad).value
                lat_arr[grp] = obs.lat.to(u.rad).value
                height_arr[grp] = obs.height.to(u.km).value
                valid_arr[grp] = True

            # Validate altitudes: must be in [0, pi/2]
            bad = (alt_arr < 0.0) | (alt_arr > np.pi / 2.0)
            valid_arr[bad] = False
            alt_arr[bad] = np.pi / 2.0  # replace invalid with zenith

            tropo_alt = alt_arr
            tropo_alt_valid = valid_arr
            obs_geodetic_lat = lat_arr
            obs_height_km = height_arr

    # -- GP basis time coordinate: barycentered TOAs -------------------------
    # Model-dependent, so left unset when no model was supplied — building GP
    # noise components from such a TOAData raises rather than guessing.
    basis_seconds = None
    if model is not None:
        basis_seconds = (
            np.array(model.get_barycentric_toas(toas).value, dtype=np.float64) * 86400.0
        )

    # -- TZR TOA for absolute phase -----------------------------------------
    tzr_tdb_int = None
    tzr_tdb_frac = None
    tzr_freq = None
    tzr_ssb_obs_pos = None
    tzr_obs_sun_pos = None
    tzr_planet_positions = None
    if model is not None:
        tzr_info = extract_tzr_toa(model, toas)
        tzr_tdb_int = tzr_info["tdb_int"]
        tzr_tdb_frac = tzr_info["tdb_frac"]
        tzr_freq = tzr_info["freq"]
        tzr_ssb_obs_pos = tzr_info["ssb_obs_pos"]
        tzr_obs_sun_pos = tzr_info["obs_sun_pos"]
        tzr_planet_positions = tzr_info["planet_positions"]

    # -- Assemble TOAData (from_arrays owns the dtype coercion) ----------
    return TOAData.from_arrays(
        mjd_int=mjd_int,
        mjd_frac=mjd_frac,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=error_s,
        freq=freq_mhz,
        delta_pulse_number=delta_pulse_number,
        ssb_obs_pos=ssb_obs_pos,
        ssb_obs_vel=ssb_obs_vel,
        obs_sun_pos=obs_sun_pos,
        obs_indices=obs_indices,
        flag_masks=flag_masks,
        planet_positions=planet_positions,
        dm_values=dm_values,
        dm_errors=dm_errors,
        tropo_alt=tropo_alt,
        tropo_alt_valid=tropo_alt_valid,
        obs_geodetic_lat=obs_geodetic_lat,
        obs_height_km=obs_height_km,
        tzr_tdb_int=tzr_tdb_int,
        tzr_tdb_frac=tzr_tdb_frac,
        tzr_freq=tzr_freq,
        tzr_ssb_obs_pos=tzr_ssb_obs_pos,
        tzr_obs_sun_pos=tzr_obs_sun_pos,
        tzr_planet_positions=tzr_planet_positions,
        n_toas=n_toas,
        obs_names=obs_names,
        basis_seconds=basis_seconds,
        basis_coord="barycentric" if basis_seconds is not None else None,
    )
