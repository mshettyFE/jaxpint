"""TOAData: pre-extracted TOA data as JAX arrays."""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, get_args

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from jaxpint.types.dual_float import DualFloat

# Time coordinates a TOAData's GP basis times may be expressed in.  The label
# travels with ``basis_seconds`` (see ``TOAData.basis_coord``) so the choice
# is inspectable data, not just producer-side documentation.
BasisCoord = Literal["barycentric", "tdb"]


class TOAData(eqx.Module):
    """Pre-extracted TOA data as JAX arrays.

    Created by the bridge layer from PINT ``TOAs`` objects. All astropy units
    are stripped; see unit conventions below.

    Unit conventions (the dtype half is enforced by :meth:`from_arrays`, the
    single builder both loaders route through; units are the caller's contract)::

        mjd_int, mjd_frac:      days (integer MJD + fractional day in [0, 1))
        tdb_int, tdb_frac:      days (TDB timescale, same split)
        error:                  seconds
        freq:                   MHz (barycentric, Doppler-corrected)
        basis_seconds:          seconds (GP basis times; see basis_coord)
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

    # Time coordinate (seconds) that GP Fourier bases and ECORR quantization
    # are evaluated at.
    # The conventions in use:
    #   bridge / native loader -- barycentered TOAs (TDB minus every delay
    #     ahead of the binary component: solar-system Roemer/Shapiro,
    #     dispersion, ...), the enterprise/discovery convention for real data,
    #     evaluated once at the par-file parameter values.  Bridge: PINT
    #     ``model.get_barycentric_toas``; native: ``native_toas_to_jax``
    #     stamps them at conversion time when a par is supplied.
    #   synthetic / test data with zero solar-system geometry -- TDB, which
    #     equals the barycentric time exactly when all such delays are zero.
    #   PINT-parity tests -- TDB deliberately, to compare pure math against
    #     PINT at PINT's own time coordinate (tests/test_pl_noise_vs_pint.py).
    basis_seconds: Optional[Float[Array, " n_toas"]] = eqx.field(default=None)
    # Which coordinate ``basis_seconds`` holds ("barycentric" | "tdb").
    basis_coord: Optional[BasisCoord] = eqx.field(static=True, default=None)

    # Provenance of the clock corrections already baked into ``mjd_*``/``tdb_*``.
    clock_realization: Optional[str] = eqx.field(static=True, default=None)

    # Absolute pulse numbers: the integer rotation count each TOA is assigned
    # to, or None (the common case -- residuals then use nearest-pulse
    # tracking). Stored as float64: rotation counts reach ~1e11-1e12, far
    # inside float64's exact-integer range (2**53). Sources: ``-pn`` flags in
    # the .tim, PINT's ``pulse_number`` column via the bridge, or
    # ``compute_pulse_numbers`` from a trusted model. NaN entries are not
    # allowed -- the loaders enforce all-or-nothing (PINT likewise refuses
    # partial coverage), because a NaN would silently poison the residuals.
    pulse_number: Optional[Float[Array, " n_toas"]] = eqx.field(default=None)

    def __check_init__(self):
        if (self.basis_seconds is None) != (self.basis_coord is None):
            raise ValueError(
                "basis_seconds and basis_coord must be set together: the GP "
                "basis times are only meaningful with their coordinate label "
                f"(got basis_seconds={'set' if self.basis_seconds is not None else None}, "
                f"basis_coord={self.basis_coord!r})."
            )
        if self.basis_coord is not None and self.basis_coord not in get_args(
            BasisCoord
        ):
            raise ValueError(
                f"Unknown basis_coord {self.basis_coord!r}; "
                f"expected one of {get_args(BasisCoord)}."
            )

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

    def with_basis_seconds(
        self, basis_seconds: Float[Array, " n_toas"], coord: BasisCoord
    ) -> "TOAData":
        """Return a copy with ``basis_seconds`` + its coordinate label set.

        ``coord`` is required: the producer declares which time coordinate
        the values are in (see the field docs).  Goes through ``__init__``
        (``dataclasses.replace``) so ``__check_init__`` re-validates —
        ``basis_coord`` is a static field, which ``eqx.tree_at`` cannot
        replace.
        """
        return dataclasses.replace(
            self,
            basis_seconds=jnp.asarray(basis_seconds, dtype=jnp.float64),
            basis_coord=coord,
        )

    def with_pulse_numbers(self, pulse_number) -> "TOAData":
        """Return a copy carrying absolute pulse numbers.

        Validated eagerly (NumPy, not traced): the array must be finite and
        integer-valued, because a stray NaN or fractional entry would corrupt
        tracked residuals silently rather than fail here.
        """
        arr = np.asarray(pulse_number, dtype=np.float64)
        if arr.shape != (self.n_toas,):
            raise ValueError(
                f"pulse_number must have shape ({self.n_toas},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("pulse_number must be finite for every TOA")
        if not np.all(arr == np.round(arr)):
            raise ValueError("pulse_number must be integer-valued")
        return dataclasses.replace(
            self, pulse_number=jnp.asarray(arr, dtype=jnp.float64)
        )

    def with_computed_pulse_numbers(self, model, params) -> "TOAData":
        """Freeze pulse numbers from a trusted model, in one call.

        The one-line form of ``compute_pulse_numbers(model, self, params)`` +
        :meth:`with_pulse_numbers`: evaluates the model phase at *params*,
        rounds each TOA to its nearest rotation, and returns a copy carrying
        those assignments -- after which every fit against it tracks
        absolutely (presence-based ``track_mode``).

        *params* should be a solution you trust (typically the par values a
        phase-connected model was published with): the assignments are only
        as good as the model that made them. Note the fixed point at a good
        solution: recomputing at converged fitted params reproduces the same
        numbers whenever every tracked residual is under half a turn, so
        "refreezing" after a successful fit is a no-op by construction.
        """
        phase = model.compute_phase(self, params)
        return self.with_pulse_numbers(phase.int + jnp.round(phase.frac))

    # -- Phase-connection editing verbs --------------------------------------
    #
    # Interactive affordances for the manual phase-connection workflow (the
    # pintk-style "insert a turn after this gap" moves). Eagerly validated
    # NumPy operations returning new TOAData -- editing verbs, not traced
    # code; do not call them inside jit.
    #
    # Sign conventions (see compute_phase_residuals):
    #     residual = phase + delta_pulse_number - pulse_number
    # so +1 phase turn RAISES the residual by one cycle, while +1 to the
    # pulse-number assignment LOWERS it by one; applying both to the same
    # TOAs cancels exactly.

    def _selection(self, after_mjd, mask) -> np.ndarray:
        """Boolean selection from exactly one of *after_mjd* / *mask*."""
        if (after_mjd is None) == (mask is None):
            raise ValueError(
                "select TOAs with exactly one of after_mjd=<MJD> or mask=<bool array>"
            )
        if after_mjd is not None:
            mjd = np.asarray(self.mjd_int) + np.asarray(self.mjd_frac)
            return mjd > float(after_mjd)
        m = np.asarray(mask)
        if m.shape != (self.n_toas,) or m.dtype != np.bool_:
            raise ValueError(
                f"mask must be a bool array of shape ({self.n_toas},), got "
                f"{m.dtype} {m.shape}"
            )
        return m

    def with_delta_pulse_number(self, delta) -> "TOAData":
        """Replace ``delta_pulse_number`` wholesale (finite values required)."""
        arr = np.asarray(delta, dtype=np.float64)
        if arr.shape != (self.n_toas,):
            raise ValueError(
                f"delta_pulse_number must have shape ({self.n_toas},), got {arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("delta_pulse_number must be finite for every TOA")
        return dataclasses.replace(
            self, delta_pulse_number=jnp.asarray(arr, dtype=jnp.float64)
        )

    def add_phase_turns(self, turns, *, after_mjd=None, mask=None) -> "TOAData":
        """Add *turns* to ``delta_pulse_number`` for the selected TOAs.

        The programmatic form of a mid-file ``PHASE`` command (integer turns)
        or a ``-padd`` flag (fractional allowed): asserts that the selected
        TOAs' phases are *turns* rotations beyond what the bare model
        predicts. Positive turns raise those residuals by ``+turns`` cycles
        (in nearest mode only the fractional part is observable, exactly as
        for PHASE commands). Serializes through ``write_tim`` as ``-padd``.
        """
        turns = float(turns)
        if not np.isfinite(turns):
            raise ValueError("turns must be finite")
        sel = self._selection(after_mjd, mask)
        dpn = np.asarray(self.delta_pulse_number, dtype=np.float64).copy()
        dpn[sel] += turns
        return dataclasses.replace(
            self, delta_pulse_number=jnp.asarray(dpn, dtype=jnp.float64)
        )

    def shift_pulse_numbers(self, turns, *, after_mjd=None, mask=None) -> "TOAData":
        """Shift the absolute pulse-number assignments of the selected TOAs.

        The manual connection move: "everything after the gap is actually
        *turns* rotations later than assigned". Requires pulse numbers to be
        present and *turns* to be a whole number (assignments are rotation
        counts). Positive turns LOWER the selected tracked residuals by
        ``turns`` cycles -- the opposite sign to :meth:`add_phase_turns`, per
        the residual convention above.
        """
        turns = float(turns)
        if turns != round(turns):
            raise ValueError(
                f"pulse-number shifts must be whole rotations, got {turns!r}"
            )
        if self.pulse_number is None:
            raise ValueError(
                "no pulse numbers to shift; assign them first "
                "(with_computed_pulse_numbers / with_pulse_numbers)"
            )
        sel = self._selection(after_mjd, mask)
        pn = np.asarray(self.pulse_number, dtype=np.float64).copy()
        pn[sel] += turns
        return self.with_pulse_numbers(pn)

    def without_pulse_numbers(self) -> "TOAData":
        """Drop the pulse-number assignments (fits revert to nearest mode)."""
        return dataclasses.replace(self, pulse_number=None)

    def require_basis_seconds(self) -> Float[Array, " n_toas"]:
        """``basis_seconds``, raising a diagnosable error when unset.

        GP components must not silently pick a time coordinate: an unset
        field means the producer of this TOAData never made the choice.  The
        check is on static pytree structure (None-ness), so it fires at trace
        time and is jit-safe.
        """
        if self.basis_seconds is None:
            raise ValueError(
                "TOAData.basis_seconds is not set, but a GP basis / ECORR "
                "quantization needs an explicit time coordinate. Real data: "
                "the converters set barycentered TOAs when given a model "
                "(bridge: pint_toas_to_jax(toas, model=...); native: "
                "native_toas_to_jax(tim, par_result)). Synthetic data with "
                "zero solar-system geometry: with_basis_seconds(tdb_seconds, "
                "'tdb') (exactly equal to barycentric time there)."
            )
        return self.basis_seconds

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
            known = ", ".join(sorted(self.flag_masks)) or "(none)"
            raise KeyError(
                f"no flag mask for masked parameter {name!r}; this TOAData "
                f"carries masks for: {known}. Masks are resolved at conversion "
                f"time from the par file's mask parameters -- build the TOAData "
                f"with the model in hand, via "
                f"pint_toas_to_jax(toas, model=<PINT model>) or "
                f"native_toas_to_jax(tim, par_result)."
            )
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

    @classmethod
    def from_arrays(
        cls,
        *,
        mjd_int,
        mjd_frac,
        tdb_int,
        tdb_frac,
        error,
        freq,
        delta_pulse_number,
        ssb_obs_pos,
        ssb_obs_vel,
        obs_sun_pos,
        obs_indices,
        n_toas: int,
        obs_names: tuple[str, ...],
        flag_masks: Optional[dict] = None,
        planet_positions: Optional[dict] = None,
        dm_values=None,
        dm_errors=None,
        tropo_alt=None,
        tropo_alt_valid=None,
        obs_geodetic_lat=None,
        obs_height_km=None,
        tzr_tdb_int=None,
        tzr_tdb_frac=None,
        tzr_freq=None,
        tzr_ssb_obs_pos=None,
        tzr_obs_sun_pos=None,
        tzr_planet_positions=None,
        basis_seconds=None,
        basis_coord: Optional[BasisCoord] = None,
        clock_realization: Optional[str] = None,
        pulse_number=None,
    ) -> "TOAData":
        """Build a TOAData from raw NumPy/JAX arrays, owning all dtype coercion.

        The single home for the dtype contract (see the class docstring): every
        continuous field is cast to ``float64``, ``obs_indices`` to ``int32``,
        and the boolean fields (``flag_masks`` values, ``tropo_alt_valid``) to
        ``bool_``; the optional blocks pass ``None`` through untouched.  Both
        loaders — the native ``.par``/``.tim`` pipeline and the PINT bridge —
        assemble their ``TOAData`` through here, so the coercion lives once
        rather than field-for-field in each.  Inputs may be NumPy (including
        longdouble, downcast via ``np.asarray`` first) or JAX arrays.
        """
        f = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)  # noqa: E731
        fopt = lambda a: None if a is None else f(a)  # noqa: E731
        planets = lambda d: (  # noqa: E731
            None if d is None else {k: f(v) for k, v in d.items()}
        )
        return cls(
            mjd_int=f(mjd_int),
            mjd_frac=f(mjd_frac),
            tdb_int=f(tdb_int),
            tdb_frac=f(tdb_frac),
            error=f(error),
            freq=f(freq),
            delta_pulse_number=f(delta_pulse_number),
            ssb_obs_pos=f(ssb_obs_pos),
            ssb_obs_vel=f(ssb_obs_vel),
            obs_sun_pos=f(obs_sun_pos),
            obs_indices=jnp.asarray(np.asarray(obs_indices), dtype=jnp.int32),
            flag_masks=(
                {}
                if not flag_masks
                else {k: jnp.asarray(v, dtype=jnp.bool_) for k, v in flag_masks.items()}
            ),
            planet_positions=planets(planet_positions),
            dm_values=fopt(dm_values),
            dm_errors=fopt(dm_errors),
            tropo_alt=fopt(tropo_alt),
            tropo_alt_valid=(
                None
                if tropo_alt_valid is None
                else jnp.asarray(tropo_alt_valid, dtype=jnp.bool_)
            ),
            obs_geodetic_lat=fopt(obs_geodetic_lat),
            obs_height_km=fopt(obs_height_km),
            n_toas=n_toas,
            obs_names=obs_names,
            tzr_tdb_int=tzr_tdb_int,
            tzr_tdb_frac=tzr_tdb_frac,
            tzr_freq=tzr_freq,
            tzr_ssb_obs_pos=fopt(tzr_ssb_obs_pos),
            tzr_obs_sun_pos=fopt(tzr_obs_sun_pos),
            tzr_planet_positions=planets(tzr_planet_positions),
            basis_seconds=fopt(basis_seconds),
            basis_coord=basis_coord,
            clock_realization=clock_realization,
            pulse_number=fopt(pulse_number),
        )
