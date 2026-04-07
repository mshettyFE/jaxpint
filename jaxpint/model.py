"""Timing model orchestration: chains delays and sums phases.

Ports the orchestration logic from PINT's ``TimingModel.delay()`` and
``TimingModel.phase()`` as a pure Equinox module.  All fields are static
(component lists don't change during fitting), so ``TimingModel`` has no
dynamic leaves -- all differentiation flows through ``ParameterVector.values``.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, DispersionDelayComponent, PhaseComponent
from jaxpint.dual_float import DualFloat
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

        Returns
        -------
        Float[Array, " n_toas"]
            Total signal delay in **seconds**.
        """
        n = len(self.delay_components)
        if n == 0:
            return jnp.zeros(toa_data.n_toas)

        branches = [
            lambda td, p, d, comp=comp: d + comp(td, p, d)
            for comp in self.delay_components
        ]

        def body(i, delay):
            return jax.lax.switch(i, branches, toa_data, params, delay)

        return jax.lax.fori_loop(
            0, n, body, jnp.zeros(toa_data.n_toas)
        )

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute total model DM at each TOA (pc/cm³).

        Sums DM contributions from all :class:`DispersionDelayComponent`
        instances in :attr:`dispersion_components`.  Used for wideband
        DM residual computation.
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
            phase = phase + DualFloat.cycles(
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
        n = len(self.phase_components)
        if n == 0:
            zeros = jnp.zeros(toa_data.n_toas)
            return DualFloat.cycles(zeros, zeros)

        branches = [
            lambda td, p, d, comp=comp: comp(td, p, d)
            for comp in self.phase_components
        ]

        def body(i, acc):
            phase_int, phase_frac = acc
            contribution = jax.lax.switch(i, branches, toa_data, params, delay)
            summed = DualFloat.cycles(phase_int, phase_frac) + contribution
            return (summed.int, summed.frac)

        zeros = jnp.zeros(toa_data.n_toas)
        result_int, result_frac = jax.lax.fori_loop(
            0, n, body, (zeros, zeros)
        )
        return DualFloat.cycles(result_int, result_frac)

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
        tzr_toa = _build_tzr_toa_data(toa_data)
        tzr_delay = self.compute_delay(tzr_toa, params)
        tzr_phase = self._sum_phase_components(tzr_toa, params, tzr_delay)
        # Broadcast: return shape (1,) so subtraction works with (n_toas,)
        return tzr_phase


def _build_tzr_toa_data(toa_data: TOAData) -> TOAData:
    """Construct a single-element TOAData for the TZR reference TOA.

    Uses the TZR fields stored on *toa_data*.  Fields not relevant to
    the TZR evaluation (e.g. observatory indices) are set to zeros.

    Note: ``flag_masks`` is empty and ``planet_positions`` is None.
    This is correct for PhaseJump (TZR should have no jumps applied),
    but means SolarSystemShapiroDelay with per-planet positions will
    not contribute to the TZR phase.
    """
    one = jnp.ones(1)
    zero = jnp.zeros(1)
    zero3 = jnp.zeros((1, 3))

    tdb_int = jnp.array([toa_data.tzr_tdb_int])
    tdb_frac = jnp.array([toa_data.tzr_tdb_frac])

    freq = jnp.array([toa_data.tzr_freq]) if toa_data.tzr_freq is not None else jnp.inf * one

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
        planet_positions=None,
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
    )
