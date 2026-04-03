"""Shared test helpers for constructing TOAData and ParameterVector."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxpint.types import TOAData, ParameterVector


def make_toa_data(
    n_toas=5,
    *,
    t_mjd=None,
    tdb_int=59000.0,
    tdb_frac=None,
    error=1e-6,
    freq=1400.0,
    flag_masks=None,
    planet_positions=None,
    dm_values=None,
    dm_errors=None,
    tropo_alt=None,
    tropo_alt_valid=None,
    obs_geodetic_lat=None,
    obs_height_km=None,
    obs_names=("fake",),
    tzr_tdb_int=None,
    tzr_tdb_frac=None,
    tzr_freq=None,
    tzr_ssb_obs_pos=None,
):
    """Build a minimal TOAData for tests.

    Two modes:
    - Pass ``t_mjd`` (array of MJD values) to split into int/frac automatically.
    - Or pass ``n_toas`` with optional ``tdb_int``/``tdb_frac`` for linspace-style.
    """
    if t_mjd is not None:
        t_np = np.asarray(t_mjd)
        n_toas = len(t_np)
        tdb_int_arr = jnp.array(np.floor(t_np))
        tdb_frac_arr = jnp.array(t_np - np.floor(t_np))
    else:
        if tdb_frac is None:
            tdb_frac_arr = jnp.linspace(0.1, 0.9, n_toas)
        else:
            tdb_frac_arr = jnp.broadcast_to(jnp.asarray(tdb_frac), (n_toas,))
        tdb_int_arr = jnp.full(n_toas, tdb_int)

    error_arr = jnp.broadcast_to(jnp.asarray(error, dtype=jnp.float64), (n_toas,))
    freq_arr = jnp.broadcast_to(jnp.asarray(freq), (n_toas,))

    if flag_masks is None:
        flag_masks = {}
    else:
        flag_masks = {
            k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()
        }

    if planet_positions is None:
        planet_positions = {}

    return TOAData(
        mjd_int=tdb_int_arr,
        mjd_frac=tdb_frac_arr,
        tdb_int=tdb_int_arr,
        tdb_frac=tdb_frac_arr,
        error=error_arr,
        freq=freq_arr,
        delta_pulse_number=jnp.zeros(n_toas),
        ssb_obs_pos=jnp.zeros((n_toas, 3)),
        ssb_obs_vel=jnp.zeros((n_toas, 3)),
        obs_sun_pos=jnp.zeros((n_toas, 3)),
        obs_indices=jnp.zeros(n_toas, dtype=jnp.int32),
        flag_masks=flag_masks,
        planet_positions=planet_positions,
        dm_values=dm_values,
        dm_errors=dm_errors,
        tropo_alt=tropo_alt,
        tropo_alt_valid=tropo_alt_valid,
        obs_geodetic_lat=obs_geodetic_lat,
        obs_height_km=obs_height_km,
        n_toas=n_toas,
        obs_names=obs_names,
        tzr_tdb_int=tzr_tdb_int,
        tzr_tdb_frac=tzr_tdb_frac,
        tzr_freq=tzr_freq,
        tzr_ssb_obs_pos=tzr_ssb_obs_pos,
    )


def make_params(
    names,
    values,
    *,
    frozen_mask=None,
    units=None,
    components=None,
    epoch_int_values=None,
    bounds=None,
):
    """Build a minimal ParameterVector for tests.

    Parameters
    ----------
    names : tuple of str
    values : sequence of float
    frozen_mask : tuple of bool, optional
        Defaults to all False.
    units : tuple of str, optional
        Defaults to all empty strings.
    components : str or tuple of str, optional
        If a single string, applied to all params.  Defaults to "test".
    epoch_int_values : dict, optional
    bounds : tuple, optional
        Defaults to (None, None) for each param.
    """
    names = tuple(names)
    n = len(names)

    if frozen_mask is None:
        frozen_mask = (False,) * n
    if units is None:
        units = ("",) * n
    if components is None:
        components = ("test",) * n
    elif isinstance(components, str):
        components = (components,) * n
    else:
        components = tuple(components)
    if epoch_int_values is None:
        epoch_int_values = {}
    if bounds is None:
        bounds = ((None, None),) * n

    return ParameterVector(
        values=jnp.array(values, dtype=jnp.float64),
        frozen_mask=tuple(frozen_mask),
        names=names,
        units=tuple(units),
        components=components,
        _name_to_index={name: i for i, name in enumerate(names)},
        bounds=tuple(bounds),
        epoch_int_values=epoch_int_values,
    )
