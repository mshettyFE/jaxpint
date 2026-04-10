"""Padding and layout utilities for batched PTA likelihood.

Provides:
- Universal parameter layout: superset of all pulsar parameter names
- TOAData padding: pad arrays to n_max for consistent shapes
- Noise model padding: pad stored basis arrays row-wise to n_max
- Parameter reindexing: map per-pulsar values to universal layout
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.noise import NoiseModel
from jaxpint.noise.ecorr import EcorrNoise
from jaxpint.noise.red_noise import PLRedNoise


# ---------------------------------------------------------------------------
# Universal parameter layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniversalParamLayout:
    """Superset parameter layout shared by all pulsars.

    After reindexing, every pulsar's ParameterVector has these names
    in this order.  Missing parameters get default values (0.0 for most,
    1.0 for EFAC, 0.0 for EQUAD/ECORR).
    """

    names: tuple[str, ...]
    frozen_mask: tuple[bool, ...]
    units: tuple[str, ...]
    epoch_int_values: dict[str, float]
    _name_to_index: dict[str, int]
    n_params: int

    # Template ParameterVector (values are zeros, but static metadata is set)
    template_values: Float[Array, " n_params"]


def build_universal_param_layout(
    pulsar_params: tuple[ParameterVector, ...],
) -> UniversalParamLayout:
    """Build a universal parameter layout from all pulsars' ParameterVectors.

    The universal layout is the union of all parameter names across pulsars.
    Parameter ordering: parameters appear in the order they're first seen,
    iterating over pulsars in order.
    """
    seen_names: dict[str, int] = {}  # name -> first-seen index
    all_units: dict[str, str] = {}
    all_frozen: dict[str, bool] = {}
    all_epoch_ints: dict[str, float] = {}

    for pp in pulsar_params:
        for i, name in enumerate(pp.names):
            if name not in seen_names:
                seen_names[name] = len(seen_names)
                all_units[name] = pp.units[i]
                # Frozen in universal layout only if frozen in ALL pulsars
                all_frozen[name] = pp.frozen_mask[i]
            else:
                # If free in any pulsar, it's free in universal layout
                if not pp.frozen_mask[i]:
                    all_frozen[name] = False

            # Merge epoch_int_values
            if name in pp.epoch_int_values:
                all_epoch_ints[name] = pp.epoch_int_values[name]

    # Build ordered tuples
    ordered_names = sorted(seen_names.keys(), key=lambda n: seen_names[n])
    names = tuple(ordered_names)
    frozen_mask = tuple(all_frozen[n] for n in names)
    units = tuple(all_units[n] for n in names)
    name_to_index = {n: i for i, n in enumerate(names)}
    n_params = len(names)

    # Default values: 0.0 for most, 1.0 for EFAC parameters
    defaults = np.zeros(n_params, dtype=np.float64)
    for name, idx in name_to_index.items():
        if name.startswith("EFAC"):
            defaults[idx] = 1.0
    template_values = jnp.array(defaults)

    return UniversalParamLayout(
        names=names,
        frozen_mask=frozen_mask,
        units=units,
        epoch_int_values=all_epoch_ints,
        _name_to_index=name_to_index,
        n_params=n_params,
        template_values=template_values,
    )


def reindex_param_values(
    params: ParameterVector,
    layout: UniversalParamLayout,
) -> Float[Array, " n_params_universal"]:
    """Map a pulsar's parameter values to the universal layout.

    Returns a flat array of shape (layout.n_params,) with this pulsar's
    values placed at the correct universal indices.  Missing parameters
    get default values (0.0, or 1.0 for EFAC).

    JIT/grad compatible: uses jnp scatter operations, not Python floats.
    """
    # Build index mapping (static, computed once)
    indices = jnp.array(
        [layout._name_to_index[name] for name in params.names],
        dtype=jnp.int32,
    )
    # Scatter this pulsar's values into the universal layout
    return layout.template_values.at[indices].set(params.values)


def make_universal_parameter_vector(
    param_values: Float[Array, " n_params_universal"],
    layout: UniversalParamLayout,
) -> ParameterVector:
    """Create a ParameterVector with universal layout and given values."""
    return ParameterVector(
        values=param_values,
        frozen_mask=layout.frozen_mask,
        names=layout.names,
        units=layout.units,
        epoch_int_values=layout.epoch_int_values,
    )


# ---------------------------------------------------------------------------
# TOAData padding
# ---------------------------------------------------------------------------


def _pad_1d(arr: Float[Array, " n"], n_max: int, pad_value: float = 0.0):
    """Pad a 1D array from (n,) to (n_max,)."""
    n = arr.shape[0]
    if n >= n_max:
        return arr
    pad_width = n_max - n
    return jnp.concatenate([arr, jnp.full(pad_width, pad_value, dtype=arr.dtype)])


def _pad_2d(arr: Float[Array, "n k"], n_max: int, pad_value: float = 0.0):
    """Pad a 2D array from (n, k) to (n_max, k) along axis 0."""
    n = arr.shape[0]
    if n >= n_max:
        return arr
    pad_width = n_max - n
    k = arr.shape[1]
    return jnp.concatenate(
        [arr, jnp.full((pad_width, k), pad_value, dtype=arr.dtype)], axis=0
    )


def _pad_int_1d(arr, n_max: int, pad_value: int = 0):
    """Pad a 1D integer array from (n,) to (n_max,)."""
    n = arr.shape[0]
    if n >= n_max:
        return arr
    pad_width = n_max - n
    return jnp.concatenate([arr, jnp.full(pad_width, pad_value, dtype=arr.dtype)])


def _pad_bool_1d(arr, n_max: int, pad_value: bool = False):
    """Pad a 1D boolean array from (n,) to (n_max,)."""
    n = arr.shape[0]
    if n >= n_max:
        return arr
    pad_width = n_max - n
    return jnp.concatenate(
        [arr, jnp.full(pad_width, pad_value, dtype=jnp.bool_)]
    )


def pad_toa_data(toa_data: TOAData, n_max: int) -> TOAData:
    """Pad TOAData arrays from n_toas to n_max.

    Padding strategy:
    - Float arrays: pad with 0.0
    - Boolean masks: pad with False (padded TOAs belong to no flag group)
    - Integer indices: pad with 0
    - error: pad with 1.0 (so Ndiag padding = 1.0 after squaring)
    - freq: pad with 1e9 (avoid division by zero in dispersion)
    - Positions/velocities: pad with 0.0
    """
    if toa_data.n_toas >= n_max:
        return toa_data

    # Core 1D float arrays
    mjd_int = _pad_1d(toa_data.mjd_int, n_max)
    mjd_frac = _pad_1d(toa_data.mjd_frac, n_max)
    tdb_int = _pad_1d(toa_data.tdb_int, n_max)
    tdb_frac = _pad_1d(toa_data.tdb_frac, n_max)
    error = _pad_1d(toa_data.error, n_max, pad_value=1.0)
    freq = _pad_1d(toa_data.freq, n_max, pad_value=1e9)
    delta_pulse_number = _pad_1d(toa_data.delta_pulse_number, n_max)

    # Position/velocity arrays (n_toas, 3)
    ssb_obs_pos = _pad_2d(toa_data.ssb_obs_pos, n_max)
    ssb_obs_vel = _pad_2d(toa_data.ssb_obs_vel, n_max)
    obs_sun_pos = _pad_2d(toa_data.obs_sun_pos, n_max)

    # Integer indices
    obs_indices = _pad_int_1d(toa_data.obs_indices, n_max)

    # Flag masks: pad each with False
    flag_masks = {
        key: _pad_bool_1d(mask, n_max) for key, mask in toa_data.flag_masks.items()
    }

    # Optional planet positions
    planet_positions = None
    if toa_data.planet_positions is not None:
        planet_positions = {
            key: _pad_2d(pos, n_max) for key, pos in toa_data.planet_positions.items()
        }

    # Optional wideband DM
    dm_values = None
    dm_errors = None
    if toa_data.dm_values is not None:
        dm_values = _pad_1d(toa_data.dm_values, n_max)
    if toa_data.dm_errors is not None:
        dm_errors = _pad_1d(toa_data.dm_errors, n_max, pad_value=1.0)

    # Optional troposphere data
    tropo_alt = None
    tropo_alt_valid = None
    obs_geodetic_lat = None
    obs_height_km = None
    if toa_data.tropo_alt is not None:
        tropo_alt = _pad_1d(toa_data.tropo_alt, n_max, pad_value=jnp.pi / 2)
    if toa_data.tropo_alt_valid is not None:
        tropo_alt_valid = _pad_bool_1d(toa_data.tropo_alt_valid, n_max)
    if toa_data.obs_geodetic_lat is not None:
        obs_geodetic_lat = _pad_1d(toa_data.obs_geodetic_lat, n_max)
    if toa_data.obs_height_km is not None:
        obs_height_km = _pad_1d(toa_data.obs_height_km, n_max)

    return TOAData(
        mjd_int=mjd_int,
        mjd_frac=mjd_frac,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=error,
        freq=freq,
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
        n_toas=n_max,
        obs_names=toa_data.obs_names,
        tzr_tdb_int=toa_data.tzr_tdb_int,
        tzr_tdb_frac=toa_data.tzr_tdb_frac,
        tzr_freq=toa_data.tzr_freq,
        tzr_ssb_obs_pos=toa_data.tzr_ssb_obs_pos,
        tzr_obs_sun_pos=toa_data.tzr_obs_sun_pos,
    )


# ---------------------------------------------------------------------------
# Noise model padding (row dimension only)
# ---------------------------------------------------------------------------


def pad_noise_model(noise_model: NoiseModel, n_max: int) -> NoiseModel:
    """Pad noise model stored arrays from n_toas to n_max (row dimension).

    Required because noise model arrays (fourier_basis, quantization_matrix)
    are captured in lax.switch branch closures and must match the padded
    TOAData dimensions.

    Padding:
    - fourier_basis: pad rows with 0.0 (zero basis at padded TOAs)
    - quantization_matrix: pad rows with 0.0 (padded TOAs in no epoch)
    """
    padded_correlated = []
    for comp in noise_model.correlated:
        if isinstance(comp, EcorrNoise):
            qm = comp.quantization_matrix
            if qm.shape[0] < n_max:
                padded_qm = _pad_2d(qm, n_max)
                comp = EcorrNoise(
                    ecorr_names=comp.ecorr_names,
                    quantization_matrix=padded_qm,
                    ecorr_epoch_slices=comp.ecorr_epoch_slices,
                )
            padded_correlated.append(comp)

        elif isinstance(comp, PLRedNoise):
            fb = comp.fourier_basis
            if fb.shape[0] < n_max:
                padded_fb = _pad_2d(fb, n_max)
                comp = PLRedNoise(
                    fourier_basis=padded_fb,
                    freqs=comp.freqs,
                    freq_bin_widths=comp.freq_bin_widths,
                    tnredamp_name=comp.tnredamp_name,
                    tnredgam_name=comp.tnredgam_name,
                )
            padded_correlated.append(comp)

        else:
            # Generic: check for fourier_basis attribute
            if hasattr(comp, "fourier_basis"):
                fb = comp.fourier_basis
                if fb.shape[0] < n_max:
                    padded_fb = _pad_2d(fb, n_max)
                    comp = eqx.tree_at(
                        lambda c: c.fourier_basis, comp, padded_fb
                    )
            padded_correlated.append(comp)

    return NoiseModel(
        white_noise=noise_model.white_noise,
        correlated=tuple(padded_correlated),
        dm_white_noise=noise_model.dm_white_noise,
    )
