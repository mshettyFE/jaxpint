"""Bridge layer: converts PINT objects to JaxPINT JAX-native types.

PINT's role is purely I/O: .par/.tim parsing, observatory database, clock
corrections, ephemeris lookups, and coordinate transforms.  JaxPINT owns all
numerical computation.  This module is the boundary -- the **only** place that
touches astropy units.  It runs once per fit setup; after conversion everything
is convention-based float64 arrays (see Plans/Units.md for the unit contract).
"""

from __future__ import annotations

import logging
from math import floor
from typing import Optional

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from pint.models.parameter import (
    AngleParameter,
    MJDParameter,
    boolParameter,
    intParameter,
    maskParameter,
    strParameter,
)
from pint.models.timing_model import TimingModel as PINTTimingModel
from pint.observatory import get_observatory
from pint.observatory.topo_obs import TopoObs
from pint.toa import TOAs

from jaxpint.types import ParameterVector, TOAData

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_PARAM_TYPES = frozenset({"floatParameter", "MJDParameter", "AngleParameter"})

_PLANETS = ("jupiter", "saturn", "venus", "uranus", "neptune", "earth")


def _convert_deg_to_rad(quantity):
    """Convert a quantity with degree-based units to radian-based units.

    Uses astropy's ``dimensionless_angles()`` equivalency to replace
    degrees with radians in any compound unit (e.g. deg → rad, deg/yr → rad/s).

    Returns (value, unit_string) if conversion was applied, or None if the
    quantity does not contain degrees.
    """
    try:
        unit = quantity.unit
        if u.deg not in unit.bases:
            return None
    except (AttributeError, TypeError):
        return None

    # Build the target unit by replacing deg with rad in the decomposition
    rad_unit = unit
    for base, power in zip(unit.bases, unit.powers):
        if base == u.deg:
            rad_unit = rad_unit * (u.rad / u.deg) ** power

    converted = quantity.to(rad_unit, equivalencies=u.dimensionless_angles())
    return float(converted.value), str(converted.unit)

# JD to MJD offset
_JD_MJD_OFFSET = 2_400_000.5


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


def _split_mjd_longdouble(
    ld_array: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a longdouble MJD array into float64 ``(int_day, frac_day)``.

    ``frac_day`` is in [0, 1).
    """
    int_part = np.floor(ld_array)
    frac_part = ld_array - int_part
    return int_part.astype(np.float64), frac_part.astype(np.float64)


def _split_mjd_time(
    time_col,
) -> tuple[np.ndarray, np.ndarray]:
    """Split an astropy Time column into float64 ``(int_day, frac_day)``.

    Uses the internal ``jd1`` / ``jd2`` representation for maximum precision.
    ``jd1`` is typically a half-integer (e.g. 2459000.5), so
    ``jd1 - 2400000.5`` is the integer MJD day.

    *time_col* may be an astropy ``Column`` of individual ``Time`` objects
    (as stored in a PINT TOA table) or a single vectorized ``Time`` array.
    """
    from astropy.time import Time

    # PINT stores Time objects per-row in an object Column; coalesce.
    if not isinstance(time_col, Time):
        time_col = Time(list(time_col))

    jd1 = np.asarray(time_col.jd1, dtype=np.float64)
    jd2 = np.asarray(time_col.jd2, dtype=np.float64)
    # Convert JD pair to MJD pair: MJD = (jd1 - 2400000.5) + jd2
    # Compute combined MJD, then split into integer day + fraction in [0, 1)
    mjd1 = jd1 - _JD_MJD_OFFSET
    full_mjd = mjd1 + jd2
    mjd_int = np.floor(full_mjd)
    mjd_frac = full_mjd - mjd_int
    return mjd_int, mjd_frac


def _split_epoch_jd(quantity) -> tuple[float, float]:
    """Split a single astropy Time (from an MJDParameter) into MJD int + frac."""
    jd1 = float(quantity.jd1)
    jd2 = float(quantity.jd2)
    full_mjd = (jd1 - _JD_MJD_OFFSET) + jd2
    mjd_int = floor(full_mjd)
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

    Returns a dict with keys:
        ``tdb_int``, ``tdb_frac`` — TDB time in days (MJD, float64)
        ``freq`` — observing frequency in MHz (float; inf = no dispersion)
        ``ssb_obs_pos`` — SSB observer position in km, shape (3,)
    """
    if "AbsPhase" not in model.components:
        log.info("No AbsPhase in model; auto-generating TZR TOA from TOAs.")
        model.add_tzr_toa(toas)

    abs_phase = model.components["AbsPhase"]
    tz_toas = abs_phase.get_TZR_toa(toas)
    tz_tbl = tz_toas.table

    tdb_int, tdb_frac = _split_mjd_longdouble(
        np.asarray(tz_tbl["tdbld"])
    )

    try:
        freq = float(model.barycentric_radio_freq(tz_toas).to(u.MHz).value[0])
    except AttributeError:
        log.warning("Model has no barycentric_radio_freq; using topocentric TZR frequency")
        freq = float(tz_toas.get_freqs().to(u.MHz).value[0])

    ssb_obs_pos = np.asarray(tz_tbl["ssb_obs_pos"], dtype=np.float64)[0]

    return {
        "tdb_int": float(tdb_int[0]),
        "tdb_frac": float(tdb_frac[0]),
        "freq": freq,
        "ssb_obs_pos": ssb_obs_pos,
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
        ``maskParameter`` instances (JUMP, EFAC, EQUAD, DMX, etc.).
    """
    n_toas = toas.ntoas

    # -- Ensure computed columns exist -----------------------------------
    if "tdbld" not in toas.table.colnames:
        log.info("Computing TDBs (not yet present on TOAs)")
        toas.compute_TDBs()
    if "ssb_obs_pos" not in toas.table.colnames:
        log.info("Computing posvels (not yet present on TOAs)")
        toas.compute_posvels()

    tbl = toas.table

    # -- MJD split (UTC) -------------------------------------------------
    mjd_int, mjd_frac = _split_mjd_time(tbl["mjd"])

    # -- TDB split -------------------------------------------------------
    tdb_int, tdb_frac = _split_mjd_longdouble(np.asarray(tbl["tdbld"]))

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
    obs_names = tuple(sorted(set(obs_array)))
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
    for planet in _PLANETS:
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
            from astropy.coordinates import AltAz, SkyCoord

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

    # -- TZR TOA for absolute phase -----------------------------------------
    tzr_tdb_int = None
    tzr_tdb_frac = None
    tzr_freq = None
    tzr_ssb_obs_pos = None
    if model is not None:
        tzr_info = extract_tzr_toa(model, toas)
        tzr_tdb_int = tzr_info["tdb_int"]
        tzr_tdb_frac = tzr_info["tdb_frac"]
        tzr_freq = tzr_info["freq"]
        tzr_ssb_obs_pos = jnp.asarray(tzr_info["ssb_obs_pos"], dtype=jnp.float64)

    # -- Assemble TOAData ------------------------------------------------
    to_jnp = lambda arr: jnp.asarray(arr, dtype=jnp.float64)
    jnp_flag_masks = {k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()}
    jnp_planets = (
        {k: to_jnp(v) for k, v in planet_positions.items()}
        if planet_positions is not None
        else None
    )

    return TOAData(
        mjd_int=to_jnp(mjd_int),
        mjd_frac=to_jnp(mjd_frac),
        tdb_int=to_jnp(tdb_int),
        tdb_frac=to_jnp(tdb_frac),
        error=to_jnp(error_s),
        freq=to_jnp(freq_mhz),
        delta_pulse_number=to_jnp(delta_pulse_number),
        ssb_obs_pos=to_jnp(ssb_obs_pos),
        ssb_obs_vel=to_jnp(ssb_obs_vel),
        obs_sun_pos=to_jnp(obs_sun_pos),
        obs_indices=jnp.asarray(obs_indices, dtype=jnp.int32),
        flag_masks=jnp_flag_masks,
        planet_positions=jnp_planets,
        dm_values=to_jnp(dm_values) if dm_values is not None else None,
        dm_errors=to_jnp(dm_errors) if dm_errors is not None else None,
        tropo_alt=to_jnp(tropo_alt) if tropo_alt is not None else None,
        tropo_alt_valid=jnp.asarray(tropo_alt_valid, dtype=jnp.bool_) if tropo_alt_valid is not None else None,
        obs_geodetic_lat=to_jnp(obs_geodetic_lat) if obs_geodetic_lat is not None else None,
        obs_height_km=to_jnp(obs_height_km) if obs_height_km is not None else None,
        tzr_tdb_int=tzr_tdb_int,
        tzr_tdb_frac=tzr_tdb_frac,
        tzr_freq=tzr_freq,
        tzr_ssb_obs_pos=tzr_ssb_obs_pos,
        n_toas=n_toas,
        obs_names=obs_names,
    )


def pint_model_to_params(model: PINTTimingModel) -> ParameterVector:
    """Convert a PINT TimingModel to a JaxPINT :class:`ParameterVector`.

    Iterates all parameters, skipping non-numeric types (str, bool, int,
    func).  MJD epochs are split into a static integer day and a dynamic
    fractional day.  Angles are converted to radians.

    Parameters
    ----------
    model : pint.models.TimingModel
        The timing model to extract parameters from.
    """
    param_map = model.get_params_mapping()

    names: list[str] = []
    values: list[float] = []
    units: list[str] = []
    frozen_mask: list[bool] = []
    components: list[str] = []
    bounds: list[tuple[Optional[float], Optional[float]]] = []
    epoch_int_values: dict[str, float] = {}

    for pname in model.params:
        param = getattr(model, pname)

        # Skip non-numeric parameter types
        if isinstance(param, (strParameter, boolParameter, intParameter)):
            continue
        # funcParameter has no .value — skip anything without it
        if not hasattr(param, "value") or not hasattr(param, "quantity"):
            continue
        # Skip unset parameters
        if param.quantity is None:
            continue

        if isinstance(param, MJDParameter):
            mjd_int, mjd_frac = _split_epoch_jd(param.quantity)
            epoch_int_values[pname] = mjd_int
            values.append(mjd_frac)
            units.append("day")

        elif isinstance(param, AngleParameter):
            val_rad = float(param.quantity.to(u.rad).value)
            values.append(val_rad)
            units.append("rad")

        elif isinstance(param, maskParameter) and (
            pname.startswith("EQUAD") or pname.startswith("ECORR")
        ):
            # EQUAD/ECORR are stored in microseconds in PINT; convert to
            # seconds to match TOAData.error convention.
            values.append(float(param.quantity.to(u.s).value))
            units.append("s")

        else:
            # floatParameter, prefixParameter, maskParameter, etc.
            # Convert degree-based units to radian-based (e.g. OM deg → rad,
            # OMDOT deg/yr → rad/s) so binary models get radians throughout.
            deg_result = _convert_deg_to_rad(param.quantity)
            if deg_result is not None:
                val, unit_str = deg_result
                values.append(val)
                units.append(unit_str)
            else:
                values.append(float(param.value))
                units.append(str(param.units))

        names.append(pname)
        frozen_mask.append(param.frozen)
        components.append(param_map.get(pname, "Unknown"))

        # Bounds — PINT does not store these consistently
        param_bounds: tuple[Optional[float], Optional[float]] = (None, None)
        bounds.append(param_bounds)

    name_to_index = {n: i for i, n in enumerate(names)}

    return ParameterVector(
        values=jnp.asarray(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen_mask),
        names=tuple(names),
        units=tuple(units),
        components=tuple(components),
        _name_to_index=name_to_index,
        bounds=tuple(bounds),
        epoch_int_values=epoch_int_values,
    )


def params_to_pint_model(
    params: ParameterVector,
    model: PINTTimingModel,
) -> PINTTimingModel:
    """Write JaxPINT parameter values back into a PINT TimingModel.

    Modifies *model* in-place and returns it.  The caller should copy
    the model first (``copy.deepcopy(model)``) if the original must be
    preserved.

    Parameters
    ----------
    params : ParameterVector
        The (possibly fitted) parameter values.
    model : pint.models.TimingModel
        The PINT model to update.
    """
    for i, pname in enumerate(params.names):
        param = getattr(model, pname)
        val = float(params.values[i])

        if isinstance(param, MJDParameter):
            # Reconstruct full MJD from integer + fractional day
            full_mjd = params.epoch_int_values[pname] + val
            param.value = full_mjd

        elif isinstance(param, AngleParameter):
            # Convert radians back to the parameter's native angle unit
            native_value = float((val * u.rad).to(param.units).value)
            param.value = native_value

        elif isinstance(param, maskParameter) and (
            pname.startswith("EQUAD") or pname.startswith("ECORR")
        ):
            # Convert seconds back to the parameter's native unit (microseconds)
            native_value = float((val * u.s).to(param.units).value)
            param.value = native_value

        else:
            # If we converted deg→rad on the way in, convert back using
            # the stored unit string to reconstruct the radian-based unit.
            stored_unit_str = params.units[i]
            native_unit = param.units
            if native_unit is not None and stored_unit_str != str(native_unit):
                stored_unit = u.Unit(stored_unit_str)
                native_value = float(
                    (val * stored_unit).to(
                        native_unit, equivalencies=u.dimensionless_angles()
                    ).value
                )
                param.value = native_value
            else:
                param.value = val

    return model


def _param_is_set(pint_model, name):
    """Check if a PINT parameter is set (non-None, non-zero)."""
    if not hasattr(pint_model, name):
        return False
    p = getattr(pint_model, name)
    return p.value is not None and p.value != 0.0


def _opt_name(pint_model, name):
    """Return parameter name if set, else None."""
    return name if _param_is_set(pint_model, name) else None


def _build_binary_component(comp, pint_model):
    """Construct the appropriate JaxPINT binary DelayComponent from a PINT binary component."""
    from jaxpint.binary.bt import BinaryBT
    from jaxpint.binary.dd import BinaryDD, BinaryDDS, BinaryDDH
    from jaxpint.binary.ell1 import BinaryELL1, BinaryELL1H, BinaryELL1k

    bname = comp.binary_model_name

    if bname == "BT":
        return BinaryBT(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
        )

    elif bname == "DD":
        return BinaryDD(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            dr_name=_opt_name(pint_model, "DR"),
            dth_name=_opt_name(pint_model, "DTH"),
            a0_name=_opt_name(pint_model, "A0"),
            b0_name=_opt_name(pint_model, "B0"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard",
        )

    elif bname == "DDS":
        return BinaryDDS(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            dr_name=_opt_name(pint_model, "DR"),
            dth_name=_opt_name(pint_model, "DTH"),
            a0_name=_opt_name(pint_model, "A0"),
            b0_name=_opt_name(pint_model, "B0"),
            m2_name=_opt_name(pint_model, "M2"),
            shapmax_name="SHAPMAX",
        )

    elif bname == "DDH":
        return BinaryDDH(
            pb_name="PB", t0_name="T0", a1_name="A1",
            ecc_name="ECC", om_name="OM",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            edot_name=_opt_name(pint_model, "EDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            gamma_name=_opt_name(pint_model, "GAMMA"),
            dr_name=_opt_name(pint_model, "DR"),
            dth_name=_opt_name(pint_model, "DTH"),
            a0_name=_opt_name(pint_model, "A0"),
            b0_name=_opt_name(pint_model, "B0"),
            h3_name="H3",
            stigma_name="STIGMA",
        )

    elif bname == "ELL1":
        return BinaryELL1(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
        )

    elif bname == "ELL1H":
        # Determine Shapiro mode: H3+STIGMA or H3+H4
        if _param_is_set(pint_model, "STIGMA"):
            shapiro_mode = "h3stigma"
        elif _param_is_set(pint_model, "H4"):
            shapiro_mode = "h3h4"
        else:
            shapiro_mode = "h3stigma"
        return BinaryELL1H(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            eps1dot_name=_opt_name(pint_model, "EPS1DOT"),
            eps2dot_name=_opt_name(pint_model, "EPS2DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            h3_name="H3",
            stigma_name=_opt_name(pint_model, "STIGMA"),
            h4_name=_opt_name(pint_model, "H4"),
            shapiro_mode=shapiro_mode,
        )

    elif bname == "ELL1k":
        return BinaryELL1k(
            pb_name="PB", tasc_name="TASC", a1_name="A1",
            eps1_name="EPS1", eps2_name="EPS2",
            pbdot_name=_opt_name(pint_model, "PBDOT"),
            a1dot_name=_opt_name(pint_model, "A1DOT"),
            xpbdot_name=_opt_name(pint_model, "XPBDOT"),
            omdot_name=_opt_name(pint_model, "OMDOT"),
            lnedot_name=_opt_name(pint_model, "LNEDOT"),
            m2_name=_opt_name(pint_model, "M2"),
            sini_name=_opt_name(pint_model, "SINI"),
            shapiro_mode="standard" if _param_is_set(pint_model, "M2") else "none",
        )

    else:
        raise NotImplementedError(
            f"Binary model {bname!r} is not yet ported to JaxPINT"
        )


def build_timing_model(pint_model: PINTTimingModel):
    """Construct a JaxPINT :class:`~jaxpint.model.TimingModel` from a PINT model.

    Inspects the PINT model's component list and creates the corresponding
    JaxPINT delay, phase, and noise components.  Unrecognised components are
    logged as warnings and skipped.

    Parameters
    ----------
    pint_model : pint.models.TimingModel
        The PINT timing model to convert.

    Returns
    -------
    (jaxpint.model.TimingModel, Optional[jaxpint.noise.ScaleToaError])
        The timing model and, if the PINT model contains a ``ScaleToaError``
        component, the corresponding JaxPINT noise model.
    """
    from pint.models.spindown import Spindown as PINTSpindown
    from pint.models.dispersion_model import DispersionDM as PINTDispersionDM
    from pint.models.astrometry import AstrometryEquatorial as PINTAstrometryEquatorial
    from pint.models.astrometry import AstrometryEcliptic as PINTAstrometryEcliptic
    from pint.models.noise_model import ScaleToaError as PINTScaleToaError
    from pint.models.pulsar_binary import PulsarBinary as PINTPulsarBinary
    from pint.models.solar_system_shapiro import SolarSystemShapiro as PINTSolarSystemShapiro
    from pint.models.troposphere_delay import TroposphereDelay as PINTTroposphereDelay

    from jaxpint.model import TimingModel
    from jaxpint.spin import Spindown
    from jaxpint.dispersion_dm import DispersionDM
    from jaxpint.astrometry import AstrometryEquatorial, AstrometryEcliptic
    from jaxpint.noise import ScaleToaError
    from jaxpint.shapiro import SolarSystemShapiroDelay
    from jaxpint.troposphere import TroposphereDelay

    delay_components = []
    phase_components = []
    noise_model = None

    # Cached astrometry param names (reused by Shapiro component).
    _astro_raj = "RAJ"
    _astro_decj = "DECJ"
    _astro_pmra = None
    _astro_pmdec = None
    _astro_posepoch = None
    _astro_obliquity_arcsec = None

    # Components that are handled implicitly (not mapped to JaxPINT components)
    _IMPLICIT = {"AbsPhase"}

    for name, comp in pint_model.components.items():
        if name in _IMPLICIT:
            continue

        if isinstance(comp, PINTSpindown):
            spin_names = tuple(comp.F_terms)
            phase_components.append(Spindown(spin_param_names=spin_names))

        elif isinstance(comp, PINTAstrometryEquatorial):
            if hasattr(comp, "PMRA") and comp.PMRA.value is not None and comp.PMRA.value != 0.0:
                _astro_pmra = "PMRA"
            if hasattr(comp, "PMDEC") and comp.PMDEC.value is not None and comp.PMDEC.value != 0.0:
                _astro_pmdec = "PMDEC"

            px_name = None
            if hasattr(comp, "PX") and comp.PX.value is not None and comp.PX.value != 0.0:
                px_name = "PX"

            # POSEPOCH needed only when proper motion is active
            if _astro_pmra is not None or _astro_pmdec is not None:
                _astro_posepoch = "POSEPOCH"
                if comp.POSEPOCH.value is None:
                    _astro_posepoch = "PEPOCH"

            delay_components.append(
                AstrometryEquatorial(
                    raj_name=_astro_raj,
                    decj_name=_astro_decj,
                    pmra_name=_astro_pmra,
                    pmdec_name=_astro_pmdec,
                    px_name=px_name,
                    posepoch_name=_astro_posepoch,
                )
            )

        elif isinstance(comp, PINTAstrometryEcliptic):
            from jaxpint.utils import OBLIQUITY_ARCSEC

            _astro_raj = "ELONG"
            _astro_decj = "ELAT"

            if hasattr(comp, "PMELONG") and comp.PMELONG.value is not None and comp.PMELONG.value != 0.0:
                _astro_pmra = "PMELONG"
            if hasattr(comp, "PMELAT") and comp.PMELAT.value is not None and comp.PMELAT.value != 0.0:
                _astro_pmdec = "PMELAT"

            px_name = None
            if hasattr(comp, "PX") and comp.PX.value is not None and comp.PX.value != 0.0:
                px_name = "PX"

            # POSEPOCH needed only when proper motion is active
            if _astro_pmra is not None or _astro_pmdec is not None:
                _astro_posepoch = "POSEPOCH"
                if comp.POSEPOCH.value is None:
                    _astro_posepoch = "PEPOCH"

            # Resolve obliquity from ECL parameter
            ecl_name = comp.ECL.value if comp.ECL.value else "IERS2010"
            _astro_obliquity_arcsec = OBLIQUITY_ARCSEC[ecl_name]

            delay_components.append(
                AstrometryEcliptic(
                    elong_name=_astro_raj,
                    elat_name=_astro_decj,
                    pmelong_name=_astro_pmra,
                    pmelat_name=_astro_pmdec,
                    px_name=px_name,
                    posepoch_name=_astro_posepoch,
                    obliquity_arcsec=_astro_obliquity_arcsec,
                )
            )

        elif isinstance(comp, PINTDispersionDM):
            # Collect DM Taylor terms that are set (value not None)
            dm_names = ["DM"]
            for idx in sorted(pint_model.get_prefix_mapping("DM")):
                pname = pint_model.get_prefix_mapping("DM")[idx]
                param = getattr(pint_model, pname)
                if param.value is not None and param.value != 0.0:
                    dm_names.append(pname)

            # Determine epoch name: use DMEPOCH if set, else fall back to PEPOCH
            dmepoch_name = "DMEPOCH"
            if comp.DMEPOCH.value is None:
                dmepoch_name = "PEPOCH"

            delay_components.append(
                DispersionDM(
                    dm_param_names=tuple(dm_names),
                    dmepoch_name=dmepoch_name,
                )
            )

        elif isinstance(comp, PINTPulsarBinary):
            delay_components.append(_build_binary_component(comp, pint_model))

        elif isinstance(comp, PINTSolarSystemShapiro):
            delay_components.append(
                SolarSystemShapiroDelay(
                    raj_name=_astro_raj,
                    decj_name=_astro_decj,
                    pmra_name=_astro_pmra,
                    pmdec_name=_astro_pmdec,
                    posepoch_name=_astro_posepoch,
                    planet_shapiro=bool(comp.PLANET_SHAPIRO.value),
                    obliquity_arcsec=_astro_obliquity_arcsec,
                )
            )

        elif isinstance(comp, PINTTroposphereDelay):
            if comp.CORRECT_TROPOSPHERE.value:
                delay_components.append(TroposphereDelay())

        elif isinstance(comp, PINTScaleToaError):
            # Extract EFAC and EQUAD parameter names from the PINT component
            comp.setup()
            efac_names = tuple(sorted(comp.EFACs.keys()))
            equad_names = tuple(sorted(comp.EQUADs.keys()))
            noise_model = ScaleToaError(
                efac_names=efac_names,
                equad_names=equad_names,
            )

        else:
            log.warning(
                "Skipping PINT component %r (%s) — not yet ported to JaxPINT",
                name,
                type(comp).__name__,
            )

    timing_model = TimingModel(
        delay_components=tuple(delay_components),
        phase_components=tuple(phase_components),
    )
    return timing_model, noise_model
