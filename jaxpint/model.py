"""Timing model orchestration: chains delays and sums phases.

Ports the orchestration logic from PINT's ``TimingModel.delay()`` and
``TimingModel.phase()`` as a pure Equinox module.  All fields are static
(component lists don't change during fitting), so ``TimingModel`` has no
dynamic leaves -- all differentiation flows through ``ParameterVector.values``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, PhaseComponent
from jaxpint.phase_result import PhaseResult
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
    """

    delay_components: tuple[DelayComponent, ...] = eqx.field(static=True)
    phase_components: tuple[PhaseComponent, ...] = eqx.field(static=True)

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

    def compute_phase(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> PhaseResult:
        """Compute absolute model phase for each TOA.

        1. Computes total delay via :meth:`compute_delay`.
        2. Sums phase contributions from all phase components.
        3. Subtracts the TZR reference phase for absolute phase
           (using ``toa_data.tzr_tdb_int/frac``).

        Returns
        -------
        PhaseResult
            Absolute pulse phase in cycles (int/frac split).
        """
        delay = self.compute_delay(toa_data, params)
        phase = self._sum_phase_components(toa_data, params, delay)

        # Subtract TZR reference phase for absolute phase
        if toa_data.tzr_tdb_int is not None:
            tzr_phase = self._tzr_phase(toa_data, params)
            phase = phase - tzr_phase

        return phase

    def _sum_phase_components(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> PhaseResult:
        """Sum phase contributions from all phase components."""
        n = len(self.phase_components)
        if n == 0:
            zeros = jnp.zeros(toa_data.n_toas)
            return PhaseResult.create(zeros, zeros)

        branches = [
            lambda td, p, d, comp=comp: comp(td, p, d)
            for comp in self.phase_components
        ]

        def body(i, acc):
            phase_int, phase_frac = acc
            contribution = jax.lax.switch(i, branches, toa_data, params, delay)
            summed = PhaseResult.create(phase_int, phase_frac) + contribution
            return (summed.int, summed.frac)

        zeros = jnp.zeros(toa_data.n_toas)
        result_int, result_frac = jax.lax.fori_loop(
            0, n, body, (zeros, zeros)
        )
        return PhaseResult.create(result_int, result_frac)

    def _tzr_phase(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> PhaseResult:
        """Compute model phase at the TZR reference TOA.

        Builds a minimal single-element TOAData from the TZR fields
        stored on *toa_data*, evaluates the full delay + phase pipeline
        at that point, and returns a PhaseResult that broadcasts when
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
        obs_sun_pos=zero3,
        obs_indices=jnp.zeros(1, dtype=jnp.int32),
        flag_masks={},
        planet_positions=None,
        dm_values=None,
        dm_errors=None,
        n_toas=1,
        obs_names=toa_data.obs_names[:1] if toa_data.obs_names else ("",),
        tzr_tdb_int=None,
        tzr_tdb_frac=None,
        tzr_freq=None,
        tzr_ssb_obs_pos=None,
    )
