"""Timing model orchestration: chains delays and sums phases.

Ports the orchestration logic from PINT's ``TimingModel.delay()`` and
``TimingModel.phase()``.  All fields are static (saves HLO comp times)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import (
    DelayComponent,
    DispersionDelayComponent,
    ParamDecl,
    PhaseComponent,
    _make_component_names,
)

from jaxpint.types.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class TimingModel(eqx.Module):
    """Orchestrates delay and phase components.

    Parameters
    ----------
    delay_components : tuple[DelayComponent, ...]
        Delay components applied sequentially (order matters: each sees
        the accumulated delay from prior components).
    phase_components : tuple[PhaseComponent, ...]
        Phase components whose contributions are summed (order irrelevant).
    phoff_name : str or None
        If set, a constant phase offset (``-PHOFF``) is applied *after*
        TZR subtraction.  This matches PINT's ``PhaseOffset`` semantics
        where the offset is applied only to observation TOAs, not the TZR.
    """

    # Top-level / administrative parameters PINT puts on the timing model itself
    # (and the AbsPhase TZR anchor).  Mostly metadata; BINARY drives binary
    # detection, PHOFF activates the phase-offset handling (see jaxpint.par.spec).
    PARAMS = (
        ParamDecl("PSR", kind="str", aliases=("PSRJ", "PSRB")),
        ParamDecl("EPHEM", kind="str"),
        ParamDecl("CLOCK", kind="str", aliases=("CLK",)),
        ParamDecl("UNITS", kind="str"),
        ParamDecl("TIMEEPH", kind="str"),
        ParamDecl("T2CMETHOD", kind="str"),
        ParamDecl("INFO", kind="str"),
        ParamDecl("TRACK", kind="str"),
        ParamDecl("BINARY", kind="str"),
        ParamDecl("DILATEFREQ", kind="bool"),
        ParamDecl("DMDATA", kind="bool"),
        ParamDecl("NTOA", kind="int"),
        ParamDecl("START", kind="mjd"),
        ParamDecl("FINISH", kind="mjd"),
        ParamDecl("TZRMJD", kind="mjd"),
        ParamDecl(
            "TZRFRQ", kind="str"
        ),  # metadata (may be inf); never in the JAX vector
        ParamDecl("TZRSITE", kind="str"),
        ParamDecl("PHOFF"),  # triggers PHASE_OFFSET (see par.spec)
        ParamDecl("RM"),
        ParamDecl("CHI2"),
        ParamDecl("CHI2R"),
        ParamDecl("TRES"),
        ParamDecl("DMRES"),
    )

    delay_components: tuple[DelayComponent, ...] = eqx.field(static=True)
    phase_components: tuple[PhaseComponent, ...] = eqx.field(static=True)
    dispersion_components: tuple[DispersionDelayComponent, ...] = eqx.field(
        static=True, default=()
    )
    phoff_name: Optional[str] = eqx.field(static=True, default=None)

    def compute_delay(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Chain all delay components sequentially.

        Each component sees the accumulated delay from prior components.

        Parameters
        ----------
        toa_data : TOAData
            Pulse time-of-arrival data.
        params : ParameterVector
            Timing model parameters.

        Returns
        -------
        Float[Array, " n_toas"]
            Total signal delay in **seconds**.
        """
        # ``delay_components`` is a static eqx field, so we iterate at
        # trace time. This produces a flat HLO graph proportional to the
        # number of components — much smaller than ``lax.fori_loop`` +
        # ``lax.switch``, which forces every branch into the graph as a
        # separate subgraph.
        delay = jnp.zeros(toa_data.n_toas)
        for comp in self.delay_components:
            delay = delay + comp(toa_data, params, delay)
        return delay

    def compute_delay_to_binary(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Sum delay components up to (but excluding) the binary component.

        Mirrors PINT's ``delay(cutoff=<binary>)`` used by
        ``get_barycentric_toas``: every delay that maps the topocentric arrival
        time to the binary barycentre, but not the binary orbital delay itself.
        With no binary component this equals :meth:`compute_delay`.
        """
        from jaxpint.binary import (
            BinaryBT,
            BinaryBTPiecewise,
            BinaryDD,
            BinaryDDGR,
            BinaryDDK,
            BinaryELL1,
        )

        binary_types = (
            BinaryBT,
            BinaryBTPiecewise,
            BinaryDD,
            BinaryDDGR,
            BinaryDDK,
            BinaryELL1,
        )
        delay = jnp.zeros(toa_data.n_toas)
        for comp in self.delay_components:
            if isinstance(comp, binary_types):
                break
            delay = delay + comp(toa_data, params, delay)
        return delay

    def compute_barycentric_toas(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> DualFloat:
        """Barycentric TOAs (TDB days), mirroring PINT ``get_barycentric_toas``.

        ``bary = tdb - delay_before_binary`` (the delay is in seconds, converted
        to days and subtracted in extended precision).
        """
        corr_days = self.compute_delay_to_binary(toa_data, params) / 86400.0
        tdb = toa_data.tdb
        return DualFloat.from_days(tdb.int, tdb.frac - corr_days)

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute total model DM at each TOA (pc/cm³).

        Sums DM contributions from all :class:`~jaxpint.components.DispersionDelayComponent`
        instances in :attr:`dispersion_components`.  Used for wideband
        DM residual computation.

        Parameters
        ----------
        toa_data : TOAData
            Pulse time-of-arrival data.
        params : ParameterVector
            Timing model parameters.

        Returns
        -------
        Float[Array, " n_toas"]
            Total dispersion measure at each TOA, in pc/cm³.
        """
        delay = self.compute_delay(toa_data, params)
        dm = jnp.zeros(toa_data.n_toas)
        for comp in self.dispersion_components:
            dm = dm + comp.compute_dm(toa_data, params, delay)
        return dm

    def compute_phase(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> DualFloat:
        """Compute absolute model phase for each TOA.

        1. Computes total delay via :meth:`compute_delay`.
        2. Sums phase contributions from all phase components.
        3. Subtracts the TZR reference phase for absolute phase
           (using ``toa_data.tzr_tdb_int/frac``).

        Parameters
        ----------
        toa_data : TOAData
            Pulse time-of-arrival data.
        params : ParameterVector
            Timing model parameters.

        Returns
        -------
        DualFloat
            Absolute pulse phase in cycles (int/frac split).
        """
        delay = self.compute_delay(toa_data, params)
        phase = self._sum_phase_components(toa_data, params, delay)

        # Subtract TZR reference phase for absolute phase
        if toa_data.tzr_tdb_int is not None:
            tzr_phase = self._tzr_phase(toa_data, params)
            phase = phase - tzr_phase

        # Phase offset: applied after TZR subtraction so it does not
        # cancel out (PINT applies PHOFF only to observation TOAs).
        if self.phoff_name is not None:
            phoff = params.param_value(self.phoff_name)
            phase = phase + DualFloat.from_cycles(
                jnp.zeros(toa_data.n_toas), -phoff * jnp.ones(toa_data.n_toas)
            )

        return phase

    def _sum_phase_components(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Sum phase contributions from all phase components."""
        zeros = jnp.zeros(toa_data.n_toas)
        phase = DualFloat.from_cycles(zeros, zeros)
        for comp in self.phase_components:
            phase = phase + comp(toa_data, params, delay)
        return phase

    def _tzr_phase(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> DualFloat:
        """Compute model phase at the TZR reference TOA.

        Builds a minimal single-element TOAData from the TZR fields
        stored on *toa_data*, evaluates the full delay + phase pipeline
        at that point, and returns a DualFloat that broadcasts when
        subtracted from the full TOA phases.
        """
        tzr_toa = _reconstruct_tzr_toa(toa_data)
        tzr_delay = self.compute_delay(tzr_toa, params)
        tzr_phase = self._sum_phase_components(tzr_toa, params, tzr_delay)
        # Broadcast: return shape (1,) so subtraction works with (n_toas,)
        return tzr_phase

    # ------------------------------------------------------------------
    # Component indexing
    # ------------------------------------------------------------------

    @property
    def components(self) -> tuple[DelayComponent | PhaseComponent, ...]:
        """Return all components, delay components first then phase components."""
        return self.delay_components + self.phase_components

    @property
    def component_names(self) -> tuple[str, ...]:
        """Unique names for all components, auto-disambiguated for duplicates."""
        return _make_component_names(self.components)

    def __getitem__(
        self, key: str | int | slice
    ) -> DelayComponent | PhaseComponent | tuple[DelayComponent | PhaseComponent, ...]:
        """Retrieve component(s) by name, integer index, or slice.

        Parameters
        ----------
        key : str, int, or slice
            If ``str``, looks up a component by its unique name (see
            :attr:`component_names`).  If ``int`` or ``slice``, indexes
            into the combined ``(delay_components + phase_components)``
            tuple.

        Returns
        -------
        DelayComponent, PhaseComponent, or tuple thereof
            The matched component(s).

        Raises
        ------
        KeyError
            If *key* is a string that does not match any component name.
        TypeError
            If *key* is not ``str``, ``int``, or ``slice``.
        """
        if isinstance(key, str):
            names = self.component_names
            comps = self.components
            for i, name in enumerate(names):
                if name == key:
                    return comps[i]
            raise KeyError(f"{key!r} not found. Available components: {names}")
        elif isinstance(key, (int, slice)):
            return self.components[key]
        raise TypeError(f"indices must be str, int, or slice, not {type(key).__name__}")

    # ------------------------------------------------------------------
    # Per-component decomposition (diagnostic / inspection only)
    # ------------------------------------------------------------------

    def decompose_delay(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> dict[str, Float[Array, " n_toas"]]:
        """Return each delay component's individual contribution.

        Runs the full sequential delay chain, recording each component's
        delta (what it adds on top of the accumulated delay from prior
        components).

        .. warning::

           This method uses Python loops and returns a plain ``dict``.
           It must **not** be called inside ``jax.jit``-compiled
           functions.  Use :meth:`compute_delay` for JIT-compatible
           total-delay computation.

        Returns
        -------
        dict[str, Array]
            Mapping from component name to its delay contribution in
            seconds, shape ``(n_toas,)``.
        """
        names = _make_component_names(self.delay_components)
        result: dict[str, Float[Array, " n_toas"]] = {}
        accumulated = jnp.zeros(toa_data.n_toas)
        for name, comp in zip(names, self.delay_components):
            contribution = comp(toa_data, params, accumulated)
            result[name] = contribution
            accumulated = accumulated + contribution
        return result

    def decompose_phase(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> dict[str, DualFloat]:
        """Return each phase component's individual contribution.

        Computes total delay first, then evaluates each phase component
        independently (phase components are summed, not sequential).

        .. warning::

           This method uses Python loops and returns a plain ``dict``.
           It must **not** be called inside ``jax.jit``-compiled
           functions.  Use :meth:`compute_phase` for JIT-compatible
           total-phase computation.

        Returns
        -------
        dict[str, DualFloat]
            Mapping from component name to its phase contribution in
            cycles.
        """
        delay = self.compute_delay(toa_data, params)
        names = _make_component_names(self.phase_components)
        result: dict[str, DualFloat] = {}
        for name, comp in zip(names, self.phase_components):
            result[name] = comp(toa_data, params, delay)
        return result

    def decompose_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> dict[str, Float[Array, " n_toas"]]:
        """Return each dispersion component's DM contribution.

        Computes total delay first, then calls ``compute_dm`` on each
        :class:`~jaxpint.components.DispersionDelayComponent`.  Component names match those
        used by :meth:`decompose_delay` for the same components.

        .. warning::

           This method uses Python loops and returns a plain ``dict``.
           It must **not** be called inside ``jax.jit``-compiled
           functions.  Use :meth:`compute_dm` for JIT-compatible
           total-DM computation.

        Returns
        -------
        dict[str, Array]
            Mapping from component name to its DM contribution in
            pc/cm³, shape ``(n_toas,)``.
        """
        delay = self.compute_delay(toa_data, params)
        names = _make_component_names(self.dispersion_components)
        result: dict[str, Float[Array, " n_toas"]] = {}
        for name, comp in zip(names, self.dispersion_components):
            result[name] = comp.compute_dm(toa_data, params, delay)
        return result


def _reconstruct_tzr_toa(toa_data: TOAData) -> TOAData:
    """Construct a single-element TOAData for the TZR reference TOA.

    Uses the TZR fields stored on *toa_data*.  Fields not relevant to
    the TZR evaluation (e.g. observatory indices) are set to zeros.
    Note: ``flag_masks`` is empty (TZR should have no jumps applied).

    """
    one = jnp.ones(1)
    zero = jnp.zeros(1)
    zero3 = jnp.zeros((1, 3))

    tdb_int = jnp.array([toa_data.tzr_tdb_int])
    tdb_frac = jnp.array([toa_data.tzr_tdb_frac])

    freq = (
        jnp.array([toa_data.tzr_freq])
        if toa_data.tzr_freq is not None
        else jnp.inf * one
    )

    ssb_obs_pos = (
        toa_data.tzr_ssb_obs_pos[None, :]
        if toa_data.tzr_ssb_obs_pos is not None
        else zero3
    )

    obs_sun_pos = (
        toa_data.tzr_obs_sun_pos[None, :]
        if toa_data.tzr_obs_sun_pos is not None
        else zero3
    )
    # ``planet_positions`` is forwarded from ``toa_data.tzr_planet_positions``
    # when populated by the bridge — required for SolarSystemShapiroDelay
    # with PLANET_SHAPIRO Y to evaluate against the TZR.
    planet_positions = (
        {k: v[None, :] for k, v in toa_data.tzr_planet_positions.items()}
        if toa_data.tzr_planet_positions is not None
        else None
    )

    return TOAData(
        mjd_int=tdb_int,
        mjd_frac=tdb_frac,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=one,
        freq=freq,
        delta_pulse_number=zero,
        ssb_obs_pos=ssb_obs_pos,
        ssb_obs_vel=zero3,
        obs_sun_pos=obs_sun_pos,
        obs_indices=jnp.zeros(1, dtype=jnp.int32),
        flag_masks={},
        planet_positions=planet_positions,
        dm_values=None,
        dm_errors=None,
        tropo_alt=None,
        tropo_alt_valid=None,
        obs_geodetic_lat=None,
        obs_height_km=None,
        n_toas=1,
        obs_names=toa_data.obs_names[:1] if toa_data.obs_names else ("",),
        tzr_tdb_int=None,
        tzr_tdb_frac=None,
        tzr_freq=None,
        tzr_ssb_obs_pos=None,
        tzr_obs_sun_pos=None,
        tzr_planet_positions=None,
    )
