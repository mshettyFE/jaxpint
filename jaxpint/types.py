"""Core data types for JaxPINT.

Defines the three foundational types:
- PhaseResult: Pulse phase as integer + fractional parts
- TOAData: Pre-extracted TOA data as JAX arrays
- ParameterVector: Timing model parameters as a flat JAX array with metadata

All types are equinox Modules (automatic JAX pytrees) and are compatible
with jax.jit, jax.grad, jax.vmap, etc.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxpint.phase_result import PhaseResult


# ---------------------------------------------------------------------------
# TOAData
# ---------------------------------------------------------------------------

class TOAData(eqx.Module):
    """Pre-extracted TOA data as JAX arrays.

    Created by the bridge layer from PINT ``TOAs`` objects. All astropy units
    are stripped; see unit conventions below.

    Unit conventions (enforced by bridge, not by this class):
        mjd_int, mjd_frac:      days (integer MJD + fractional day in [0, 1))
        tdb_int, tdb_frac:      days (TDB timescale, same split)
        error:                  seconds
        freq:                   MHz (barycentric, Doppler-corrected)
        ssb_obs_pos:            km,   shape (n_toas, 3)
        ssb_obs_vel:            km/s, shape (n_toas, 3)
        obs_sun_pos:            km,   shape (n_toas, 3)
        delta_pulse_number:     dimensionless (cycles)
        dm_values, dm_errors:   pc/cm^3
    """

    # Core TOA data -- shape (n_toas,)
    mjd_int: Float[Array, " n_toas"]
    mjd_frac: Float[Array, " n_toas"]
    tdb_int: Float[Array, " n_toas"]
    tdb_frac: Float[Array, " n_toas"]
    error: Float[Array, " n_toas"]
    freq: Float[Array, " n_toas"]
    delta_pulse_number: Float[Array, " n_toas"]

    # Position/velocity vectors -- shape (n_toas, 3)
    ssb_obs_pos: Float[Array, "n_toas 3"]
    ssb_obs_vel: Float[Array, "n_toas 3"]
    obs_sun_pos: Float[Array, "n_toas 3"]

    # Observatory index per TOA -- shape (n_toas,)
    obs_indices: Int[Array, " n_toas"]

    # Pre-computed flag masks -- key: param name, value: bool array (n_toas,)
    flag_masks: dict[str, Bool[Array, " n_toas"]]

    # Optional planet positions -- key: planet name, value: (n_toas, 3) in km
    planet_positions: Optional[dict[str, Float[Array, "n_toas 3"]]]

    # Optional wideband DM data -- shape (n_toas,)
    dm_values: Optional[Float[Array, " n_toas"]]
    dm_errors: Optional[Float[Array, " n_toas"]]

    # Optional troposphere pre-computed data (from bridge)
    #   tropo_alt:        radians, target elevation angle (invalid replaced with pi/2)
    #   tropo_alt_valid:  True if altitude is physically valid
    #   obs_geodetic_lat: radians, observatory geodetic latitude
    #   obs_height_km:    km, observatory height above geoid
    tropo_alt: Optional[Float[Array, " n_toas"]]
    tropo_alt_valid: Optional[Bool[Array, " n_toas"]]
    obs_geodetic_lat: Optional[Float[Array, " n_toas"]]
    obs_height_km: Optional[Float[Array, " n_toas"]]

    # Static metadata (not JAX-traced)
    n_toas: int = eqx.field(static=True)
    obs_names: tuple[str, ...] = eqx.field(static=True)

    # TZR (time-zero reference) TOA for absolute phase.
    # Extracted once by the bridge from PINT's AbsPhase component (auto-generated
    # if not in par file, matching PINT's guarantee). The phase subtraction using
    # these values is handled by the orchestration layer (compute_phase / model.py),
    # not by individual phase components.
    #   tdb: days (int/frac split, same as tdb_int/tdb_frac)
    #   freq: MHz (barycentric Doppler-corrected TZRFRQ; inf means no dispersion delay)
    #   ssb_obs_pos: km, shape (3,) — SSB observer position at TZR epoch
    tzr_tdb_int: Optional[float] = eqx.field(static=True, default=None)
    tzr_tdb_frac: Optional[float] = eqx.field(static=True, default=None)
    tzr_freq: Optional[float] = eqx.field(static=True, default=None)
    tzr_ssb_obs_pos: Optional[Float[Array, " 3"]] = eqx.field(default=None)
    tzr_obs_sun_pos: Optional[Float[Array, " 3"]] = eqx.field(default=None)


# ---------------------------------------------------------------------------
# ParameterVector
# ---------------------------------------------------------------------------

class ParameterVector(eqx.Module):
    """Timing model parameters as a flat JAX array with metadata.

    Stores ALL parameters (free and frozen) in a single array. Epoch-type
    parameters (PEPOCH, T0, TASC, POSEPOCH, GLEP_*) are split: the integer
    MJD day is in ``epoch_int_values`` (static, not differentiated) and
    only the fractional day is in ``values``. The bridge layer handles
    splitting on input and recombining on output.

    Pytree: only ``values`` is a dynamic leaf (participates in jax.grad).
    All other fields are static metadata frozen into JIT traces.
    """

    values: Float[Array, " n_params"]

    # Static metadata
    frozen_mask: tuple[bool, ...] = eqx.field(static=True)
    names: tuple[str, ...] = eqx.field(static=True)
    units: tuple[str, ...] = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    _name_to_index: dict[str, int] = eqx.field(static=True)
    bounds: tuple[tuple[Optional[float], Optional[float]], ...] = eqx.field(static=True)
    epoch_int_values: dict[str, float] = eqx.field(static=True)

    def __check_init__(self):
        n = len(self.names)

        for field_name in ("frozen_mask", "units", "components", "bounds"):
            if len(getattr(self, field_name)) != n:
                raise ValueError(
                    f"len({field_name}) = {len(getattr(self, field_name))}, "
                    f"expected {n}"
                )

        if self.values.shape[0] != n:
            raise ValueError(
                f"values.shape[0] = {self.values.shape[0]}, expected {n}"
            )

        if set(self._name_to_index) != set(self.names):
            raise ValueError(
                f"_name_to_index keys {set(self._name_to_index)} "
                f"do not match names {set(self.names)}"
            )

        for name, idx in self._name_to_index.items():
            if not (0 <= idx < n):
                raise ValueError(
                    f"_name_to_index[{name!r}] = {idx} is out of range [0, {n})"
                )

        extra = set(self.epoch_int_values) - set(self.names)
        if extra:
            raise ValueError(
                f"epoch_int_values keys {extra} are not in names"
            )

    # -- Lookup helpers --

    def param_index(self, name: str) -> int:
        """Index of parameter ``name`` in the values array."""
        return self._name_to_index[name]

    def param_value(self, name: str) -> Float[Array, ""]:
        """Value of a single parameter. JIT-compatible if ``name`` is a static string."""
        return self.values[self._name_to_index[name]]

    def param_value_or(self, name: str | None, default: float = 0.0):
        """Value of a parameter if *name* is not None, otherwise *default*.

        Convenient for optional parameters stored as ``Optional[str]``
        field names on components::

            pbdot = params.param_value_or(self.pbdot_name, 0.0)
        """
        if name is None:
            return default
        return self.values[self._name_to_index[name]]

    def epoch_value(self, name: str) -> tuple[float, Float[Array, ""]]:
        """For epoch parameters: returns (integer_mjd_day, fractional_day).

        The full MJD is ``int_day + frac_day``. Only ``frac_day`` is
        differentiable.
        """
        return self.epoch_int_values[name], self.values[self._name_to_index[name]]

    # -- Free / frozen helpers --

    def free_mask_array(self) -> Bool[Array, " n_params"]:
        """Boolean array: True where parameter is free (not frozen)."""
        return jnp.array([not f for f in self.frozen_mask], dtype=jnp.bool_)

    def free_values(self) -> Float[Array, " n_free"]:
        """Extract values of free (unfrozen) parameters."""
        mask = self.free_mask_array()
        return self.values[mask]

    def free_names(self) -> tuple[str, ...]:
        """Names of free parameters (Python-level, not JIT-compatible)."""
        return tuple(n for n, f in zip(self.names, self.frozen_mask) if not f)

    def with_free_values(self, new_free: Float[Array, " n_free"]) -> ParameterVector:
        """Return a new ParameterVector with free parameter values replaced."""
        mask = self.free_mask_array()
        new_values = self.values.at[mask].set(new_free)
        return eqx.tree_at(lambda pv: pv.values, self, new_values)

    def with_value(self, name: str, val: float) -> ParameterVector:
        """Return a new ParameterVector with one parameter updated."""
        idx = self._name_to_index[name]
        new_values = self.values.at[idx].set(val)
        return eqx.tree_at(lambda pv: pv.values, self, new_values)

    def component_mask(self, component: str) -> Bool[Array, " n_params"]:
        """Boolean mask for parameters belonging to ``component``."""
        return jnp.array([c == component for c in self.components], dtype=jnp.bool_)

    @property
    def n_params(self) -> int:
        return len(self.names)

    @property
    def n_free(self) -> int:
        return sum(1 for f in self.frozen_mask if not f)
