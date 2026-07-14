"""TOAData: pre-extracted TOA data as JAX arrays."""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from jaxpint.types.dual_float import DualFloat


class TOAData(eqx.Module):
    """Pre-extracted TOA data as JAX arrays.

    Created by the bridge layer from PINT ``TOAs`` objects. All astropy units
    are stripped; see unit conventions below.

    Unit conventions (enforced by bridge, not by this class):
        mjd_int, mjd_frac:      days (integer MJD + fractional day in [0, 1))
        tdb_int, tdb_frac:      days (TDB timescale, same split)
        error:                  seconds
        freq:                   MHz (barycentric, Doppler-corrected)
        bary_seconds:           seconds (barycentered TOAs, see field docs)
        ssb_obs_pos:            km,   shape (n_toas, 3)
        ssb_obs_vel:            km/s, shape (n_toas, 3)
        obs_sun_pos:            km,   shape (n_toas, 3)
        delta_pulse_number:     dimensionless (cycles)
        dm_values, dm_errors:   pc/cm^3
    """

    # Core TOA data -- shape (n_toas,)
    # MJD in UTC. Raw observation time as recorded by telescope
    mjd_int: Float[Array, " n_toas"]
    mjd_frac: Float[Array, " n_toas"]
    # Same timestamp as MJD, but converted to Barycentric Dynamic Time (TDB)
    # Timing model operates on these values.
    # MJD is kept around for matching to original data
    tdb_int: Float[Array, " n_toas"]
    tdb_frac: Float[Array, " n_toas"]
    error: Float[Array, " n_toas"]
    freq: Float[Array, " n_toas"]
    # Offset to add to cycle number in phase computation.
    # Needed to break degeneracy of which cycle number you are on
    # For well-timed pulsars, these should all be zero
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
    # Populated when the model enables PLANET_SHAPIRO; None otherwise.
    tzr_planet_positions: Optional[dict[str, Float[Array, " 3"]]] = eqx.field(
        default=None
    )

    # Barycentered TOAs in seconds: TDB minus every delay ahead of the binary
    # component (solar-system Roemer/Shapiro, dispersion, ...), Used by enterprise;
    #   bridge  -- PINT ``model.get_barycentric_toas(toas)``;
    #   native  -- ``TimingModel.compute_barycentric_toas`` after model build
    #              (see ``with_bary_seconds``).
    # None when no model was available at conversion time.
    bary_seconds: Optional[Float[Array, " n_toas"]] = eqx.field(default=None)

    # -- Derived timestamps --

    @property
    def tdb(self) -> DualFloat:
        """TDB timestamp as a DualFloat (int day + fractional day)."""
        return DualFloat(int=self.tdb_int, frac=self.tdb_frac)

    @property
    def mjd(self) -> DualFloat:
        """MJD timestamp as a DualFloat (int day + fractional day)."""
        return DualFloat(int=self.mjd_int, frac=self.mjd_frac)

    @property
    def tdb_seconds(self) -> Float[Array, " n_toas"]:
        """TDB time in seconds.

        Multiply-then-add (rather than ``(tdb_int + tdb_frac) * 86400``)
        Loses long double precision in the process.
        For us computations, this is fine.
        """
        return self.tdb_int * 86400.0 + self.tdb_frac * 86400.0

    def with_bary_seconds(self, bary_seconds: Float[Array, " n_toas"]) -> "TOAData":
        """Return a copy with ``bary_seconds`` set (see the field docs)."""
        return eqx.tree_at(
            lambda td: td.bary_seconds,
            self,
            jnp.asarray(bary_seconds, dtype=jnp.float64),
            is_leaf=lambda x: x is None,
        )

    # -- Flag masks --

    def flag_mask(
        self, name: str, default: bool | None = None
    ) -> Bool[Array, " n_toas"]:
        """Per-TOA boolean mask for parameter ``name``.

        Present -> the stored mask. Absent -> a constant ``(n_toas,)`` array
        filled with ``default``; if ``default`` is None (the implicit "required"
        case) a ``KeyError`` is raised. JIT-safe for a static ``name`` (the
        dict-key lookup is static structure; the mask arrays are traced leaves).
        """
        if name in self.flag_masks:
            return self.flag_masks[name]
        if default is None:
            raise KeyError(name)
        return jnp.full(self.n_toas, default, dtype=jnp.bool_)

    # -- Single-TOA factory --

    @classmethod
    def single(
        cls,
        *,
        tdb_int,
        tdb_frac,
        freq,
        ssb_obs_pos,
        obs_sun_pos,
        ssb_obs_vel=None,
        planet_positions=None,
        obs_name: str = "",
    ) -> "TOAData":
        """Build a one-TOA TOAData from scalar values (e.g. a TZR reference TOA).

        Only the physically-meaningful fields are supplied; per-TOA scalars take
        neutral defaults (``error=1``, ``delta_pulse_number=0``), masks /
        wideband DM / troposphere are empty, the ``tzr_*`` back-reference fields
        are cleared, and ``mjd`` mirrors ``tdb`` (a TZR carries only TDB).
        ``ssb_obs_vel`` defaults to zero (irrelevant when ``freq`` is already
        barycentric). Inputs may be NumPy or JAX arrays / Python scalars; vector
        inputs are ``(3,)`` and are broadcast to ``(1, 3)``.
        """
        sca = lambda x: jnp.reshape(jnp.asarray(x, dtype=jnp.float64), (1,))  # noqa: E731
        vec = lambda v: jnp.reshape(jnp.asarray(v, dtype=jnp.float64), (1, 3))  # noqa: E731
        tdb_i, tdb_f = sca(tdb_int), sca(tdb_frac)
        return cls(
            mjd_int=tdb_i,
            mjd_frac=tdb_f,
            tdb_int=tdb_i,
            tdb_frac=tdb_f,
            error=jnp.ones(1),
            freq=sca(freq),
            delta_pulse_number=jnp.zeros(1),
            ssb_obs_pos=vec(ssb_obs_pos),
            ssb_obs_vel=jnp.zeros((1, 3)) if ssb_obs_vel is None else vec(ssb_obs_vel),
            obs_sun_pos=vec(obs_sun_pos),
            obs_indices=jnp.zeros(1, dtype=jnp.int32),
            flag_masks={},
            planet_positions=(
                None
                if planet_positions is None
                else {k: vec(v) for k, v in planet_positions.items()}
            ),
            dm_values=None,
            dm_errors=None,
            tropo_alt=None,
            tropo_alt_valid=None,
            obs_geodetic_lat=None,
            obs_height_km=None,
            n_toas=1,
            obs_names=(obs_name,),
            tzr_tdb_int=None,
            tzr_tdb_frac=None,
            tzr_freq=None,
            tzr_ssb_obs_pos=None,
            tzr_obs_sun_pos=None,
            tzr_planet_positions=None,
        )
