"""Core data types for JaxPINT.

Defines the three foundational types:
- DualFloat: Extended-precision value as integer + fractional parts
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

from jaxpint.dual_float import DualFloat


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
    # MJD in UTC. Raw observation time as recorded by telescope
    mjd_int: Float[Array, " n_toas"]
    mjd_frac: Float[Array, " n_toas"]
    # Same timestamp as MJD, but converted to Barycentric Dynamic Time (TDB)
    # Timing model oeprates on these values. 
    # MJD is kept around for matching to original data
    tdb_int: Float[Array, " n_toas"]
    tdb_frac: Float[Array, " n_toas"]
    # Error in timing from MJD
    error: Float[Array, " n_toas"]
    # Observational frequency of data
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
    @property
    def tdb(self) -> DualFloat:
        """TDB timestamp as a DualFloat (int day + fractional day)."""
        return DualFloat(int=self.tdb_int, frac=self.tdb_frac)

    @property
    def mjd(self) -> DualFloat:
        """MJD timestamp as a DualFloat (int day + fractional day)."""
        return DualFloat(int=self.mjd_int, frac=self.mjd_frac)

    tzr_tdb_int: Optional[float] = eqx.field(static=True, default=None)
    tzr_tdb_frac: Optional[float] = eqx.field(static=True, default=None)
    tzr_freq: Optional[float] = eqx.field(static=True, default=None)
    tzr_ssb_obs_pos: Optional[Float[Array, " 3"]] = eqx.field(default=None)
    tzr_obs_sun_pos: Optional[Float[Array, " 3"]] = eqx.field(default=None)
    # Populated when the model enables PLANET_SHAPIRO; None otherwise.
    tzr_planet_positions: Optional[dict[str, Float[Array, " 3"]]] = eqx.field(default=None)


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

    Unit conventions
    ----------------
    All values are stored as raw float64 in a fixed internal unit system.
    Components assume these units unconditionally -- no runtime conversion.

    .. list-table::
       :header-rows: 1
       :widths: 60 20

       * - Parameter(s)
         - Unit
       * - Angles (RAJ, DECJ, OM, KIN, KOM, ELONG, ELAT)
         - radians
       * - Angular rates (OMDOT, XOMDOT)
         - rad/s
       * - Proper motion (PMRA, PMDEC, PMELONG, PMELAT)
         - mas/yr
       * - Parallax (PX)
         - mas
       * - Epochs (PEPOCH, T0, TASC, POSEPOCH, ...)
         - frac day
       * - Spin frequency (F0, F1, F2, ...)
         - Hz/s^N
       * - Dispersion (DM, DM1, ``DMX_*``, CM, ``CMX_*``)
         - pc/cm^3
       * - Orbital period (PB)
         - day
       * - Projected semi-major axis (A1)
         - ls
       * - Companion mass (M2, MTOT)
         - Msun
       * - TOA error scaling (EQUAD, ECORR)
         - seconds
       * - Frequencies (``WXFREQ_*``, ``DMWXFREQ_*``, ...)
         - 1/day
       * - Delay amplitudes (``WXSIN_*``, ``WXCOS_*``, ``FD*``, ...)
         - seconds
       * - Dimensionless (EFAC, SINI, ECC, STIGMA, ...)
         - --
       * - Everything else
         - .par native

    Epoch integer MJD days are stored separately in ``epoch_int_values``
    to preserve precision; only the fractional day enters ``values``.

    Raises
    ------
    ValueError
        If lengths of ``frozen_mask``, ``units``, or ``values`` don't match
        ``names``, or if ``epoch_int_values`` contains keys not in ``names``.
    """

    values: Float[Array, " n_params"]

    # Static metadata
    # Which fitting parameters to ignore while fitting
    frozen_mask: tuple[bool, ...] = eqx.field(static=True)
    # Names of parameters which map to values 
    # Not a dictionary with values to avoi equinox warning, as well as allow 
    # ifferentiability of parameters in jax
    names: tuple[str, ...] = eqx.field(static=True)
    # Units assigned to each name
    # Purely for documentation. JAX has a preset unit system that it will assume as mentioned above
    units: tuple[str, ...] = eqx.field(static=True)
    # Storing the integer portions of the epoch values tacitly assumes that these don't change drastically while you are fitting
    # Most of the fitting tests don't break, so I guess this is reasonable?
    epoch_int_values: dict[str, float] = eqx.field(static=True)
    # Which parameters have been analytically marginalized out (their values become
    # the fiducial linearization point y_fid used by jaxpint.bayes.marginal.marginalize).
    # Mathematically distinct from frozen_mask: frozen contributes its value to
    # residuals as-is; marginalized has its prior covariance folded into the noise
    # covariance via Woodbury. Defaults to all-False (no marginalization).
    marginalized_mask: tuple[bool, ...] = eqx.field(static=True, default=())
    # 1-sigma fit uncertainty per parameter (static metadata, NOT differentiated),
    # in the same internal unit as ``values``; NaN where the source reported none.
    # Defaults to all-NaN when omitted. Accessed via :meth:`param_uncertainty`.
    uncertainties: tuple[float, ...] = eqx.field(static=True, default=())
    # Maps parameter names to values location. Autopopulate in check_init
    _name_to_index: dict[str, int] = eqx.field(static=True, default_factory=dict)
    # Integer indices of free (= not frozen AND not marginalized) parameters.
    # Autopopulated in check_init. Needed since jit doesn't like boolean jnp masks.
    _free_indices: tuple[int, ...] = eqx.field(static=True, default=())

    def __check_init__(self):
        n = len(self.names)

        for field_name in ("frozen_mask", "units"):
            if len(getattr(self, field_name)) != n:
                raise ValueError(
                    f"len({field_name}) = {len(getattr(self, field_name))}, "
                    f"expected {n}"
                )

        if self.values.shape[0] != n:
            raise ValueError(
                f"values.shape[0] = {self.values.shape[0]}, expected {n}"
            )

        # Default marginalized_mask to all-False if not provided. Setting via
        # object.__setattr__ is the standard equinox/dataclass pattern for
        # auto-populating static fields after init.
        if len(self.marginalized_mask) == 0 and n > 0:
            object.__setattr__(self, "marginalized_mask", (False,) * n)
        elif len(self.marginalized_mask) != n:
            raise ValueError(
                f"len(marginalized_mask) = {len(self.marginalized_mask)}, expected {n}"
            )

        # Default uncertainties to all-NaN (no reported sigma) if not provided.
        if len(self.uncertainties) == 0 and n > 0:
            object.__setattr__(self, "uncertainties", (float("nan"),) * n)
        elif len(self.uncertainties) != n:
            raise ValueError(
                f"len(uncertainties) = {len(self.uncertainties)}, expected {n}"
            )

        # Build _name_to_index from names
        object.__setattr__(
            self, "_name_to_index",
            {name: i for i, name in enumerate(self.names)},
        )

        # Integer indices of free parameters (= not frozen AND not marginalized),
        # used by JIT-compiled fitter code. Boolean indexing is not supported
        # inside jax.jit; integer indices are.
        object.__setattr__(
            self, "_free_indices",
            tuple(
                i for i in range(n)
                if not self.frozen_mask[i] and not self.marginalized_mask[i]
            ),
        )

        extra = set(self.epoch_int_values) - set(self.names)
        if extra:
            raise ValueError(
                f"epoch_int_values keys {extra} are not in names"
            )

    # -- Lookup helpers --

    def param_index(self, name: str) -> int:
        """Index of parameter ``name`` in the values array.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        int
            Zero-based index into ``values``.
        """
        return self._name_to_index[name]

    def param_value(self, name: str) -> Float[Array, ""]:
        """Value of a single parameter. JIT-compatible if ``name`` is a static string.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        scalar
            The parameter's current value.
        """
        return self.values[self._name_to_index[name]]

    def param_uncertainty(self, name: str) -> float:
        """1-sigma fit uncertainty of a parameter (static metadata, in the same
        internal unit as :meth:`param_value`).

        Returns ``nan`` when the source par file reported no uncertainty for this
        parameter (frozen/value-only line, or a kind that does not track one).
        This is plain Python metadata, not a traced JAX value, so it is meant for
        host-side use (e.g. building priors), not inside ``jax.jit``.

        Parameters
        ----------
        name : str
            Parameter name.

        Returns
        -------
        float
            The 1-sigma uncertainty, or ``nan`` if not reported.
        """
        return self.uncertainties[self._name_to_index[name]]

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

        Parameters
        ----------
        name : str
            Epoch parameter name (e.g. ``"PEPOCH"``, ``"T0"``).

        Returns
        -------
        tuple of (float, scalar)
            ``(integer_mjd_day, fractional_day)``.
        """
        return self.epoch_int_values[name], self.values[self._name_to_index[name]]

    def epoch_dual(self, name: str) -> DualFloat:
        """For epoch parameters: returns a DualFloat(integer_mjd, fractional_day).

        The full MJD is ``result.int + result.frac``. Only the fractional
        part is differentiable.

        Parameters
        ----------
        name : str
            Epoch parameter name (e.g. ``"PEPOCH"``, ``"T0"``).

        Returns
        -------
        DualFloat
            Extended-precision epoch value.
        """
        int_val = jnp.asarray(self.epoch_int_values[name], dtype=jnp.float64)
        frac_val = self.values[self._name_to_index[name]]
        return DualFloat(int=int_val, frac=frac_val)

    # -- Free / frozen helpers --

    def free_mask_array(self) -> Bool[Array, " n_params"]:
        """Boolean array: True where parameter is free (not frozen)."""
        return jnp.array([not f for f in self.frozen_mask], dtype=jnp.bool_)

    def free_indices_array(self) -> Int[Array, " n_free"]:
        """Integer indices of free parameters as a JAX array (JIT-safe)."""
        return jnp.array(self._free_indices, dtype=jnp.int32)

    def free_values(self) -> Float[Array, " n_free"]:
        """Extract values of free (unfrozen) parameters."""
        indices = self.free_indices_array()
        return self.values[indices]

    def free_names(self) -> tuple[str, ...]:
        """Names of free parameters (= not frozen AND not marginalized).
        Python-level, not JIT-compatible."""
        return tuple(
            self.names[i] for i in range(len(self.names))
            if not self.frozen_mask[i] and not self.marginalized_mask[i]
        )

    def marginalized_names(self) -> tuple[str, ...]:
        """Names of parameters that have been analytically marginalized out.
        Python-level, not JIT-compatible."""
        return tuple(
            self.names[i] for i in range(len(self.names))
            if self.marginalized_mask[i]
        )

    def with_free_values(self, new_free: Float[Array, " n_free"]) -> ParameterVector:
        """Return a new ParameterVector with free parameter values replaced.

        Parameters
        ----------
        new_free : array, shape (n_free,)
            Replacement values for the free (unfrozen) parameters.

        Returns
        -------
        ParameterVector
            Copy with updated free-parameter values.
        """
        indices = self.free_indices_array()
        new_values = self.values.at[indices].set(new_free)
        return eqx.tree_at(lambda pv: pv.values, self, new_values)

    def with_value(self, name: str, val: float) -> ParameterVector:
        """Return a new ParameterVector with one parameter updated.

        Parameters
        ----------
        name : str
            Parameter name.
        val : float
            New value.

        Returns
        -------
        ParameterVector
            Copy with the specified parameter updated.
        """
        idx = self._name_to_index[name]
        new_values = self.values.at[idx].set(val)
        return eqx.tree_at(lambda pv: pv.values, self, new_values)

    def with_marginalized(self, names) -> ParameterVector:
        """Return a copy with marginalized_mask=True for the given names.

        Values for marg'd entries are left as-is — they become the y_fid that
        :func:`~jaxpint.bayes.marginalize` anchors its linearization on.
        Marginalized parameters are dropped from ``free_values()`` /
        ``with_free_values()`` (their values are held fixed at y_fid during
        likelihood evaluation, but their prior covariance is folded into the
        noise covariance via Woodbury inside ``marginalize``).

        Parameters
        ----------
        names
            Iterable of parameter names (``str``) to mark as marginalized.
            Each name must be present in ``self.names``.

        Returns
        -------
        ParameterVector
            Copy with updated ``marginalized_mask``. Construction re-runs
            ``__check_init__``, which recomputes ``_free_indices`` to exclude
            both frozen and newly-marginalized entries.
        """
        marked = set(names)
        unknown = marked - set(self.names)
        if unknown:
            raise KeyError(
                f"with_marginalized: unknown parameter name(s) {sorted(unknown)}; "
                f"not in self.names."
            )
        new_mask = tuple(
            self.marginalized_mask[i] or (self.names[i] in marked)
            for i in range(len(self.names))
        )
        # Construct fresh so __check_init__ rebuilds _free_indices.
        return ParameterVector(
            values=self.values,
            frozen_mask=self.frozen_mask,
            names=self.names,
            units=self.units,
            epoch_int_values=self.epoch_int_values,
            marginalized_mask=new_mask,
            uncertainties=self.uncertainties,
        )

    @property
    def n_params(self) -> int:
        return len(self.names)

    @property
    def n_free(self) -> int:
        return sum(1 for f in self.frozen_mask if not f)
