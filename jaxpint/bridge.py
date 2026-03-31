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
from pint.models.timing_model import TimingModel
from pint.toa import TOAs

from jaxpint.types import ParameterVector, TOAData

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NUMERIC_PARAM_TYPES = frozenset({"floatParameter", "MJDParameter", "AngleParameter"})

_PLANETS = ("jupiter", "saturn", "venus", "uranus", "neptune", "earth")

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


def pint_toas_to_jax(
    toas: TOAs,
    model: Optional[TimingModel] = None,
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
        n_toas=n_toas,
        obs_names=obs_names,
    )


def pint_model_to_params(model: TimingModel) -> ParameterVector:
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

        else:
            # floatParameter, prefixParameter, maskParameter, etc.
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
    model: TimingModel,
) -> TimingModel:
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

        else:
            param.value = val

    return model
