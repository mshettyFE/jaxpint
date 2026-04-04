"""Legacy WAVE timing noise phase component.

Ports PINT's ``Wave`` class as a pure Equinox module.  The phase is
modelled as a sum of harmonics of a base angular frequency:

    phase(t) = F0 * Σ_n (a_n * sin((n+1)*ω*(t - WAVEEPOCH - delay))
                        + b_n * cos((n+1)*ω*(t - WAVEEPOCH - delay)))

where ω = WAVE_OM (rad/d), a_n/b_n are the WAVE amplitude pairs (seconds),
and F0 is the spin frequency.

Note: this is superseded by WaveX but retained for backward compatibility.

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.phase_result import PhaseResult
from jaxpint.types import TOAData, ParameterVector


class Wave(PhaseComponent):
    """Legacy WAVE timing noise model.

    Parameters
    ----------
    n_terms : int
        Number of WAVE terms (harmonic pairs).
    waveepoch_name : str
        Name of the reference epoch parameter.
    wave_om_name : str
        Name of the base angular frequency parameter (rad/day).
    wave_sin_names : tuple[str, ...]
        Names of sine amplitude parameters (seconds).
        These are the ``WAVEn_A`` entries produced by the bridge
        from PINT's pairParameter splitting.
    wave_cos_names : tuple[str, ...]
        Names of cosine amplitude parameters (seconds).
        These are the ``WAVEn_B`` entries.
    f0_name : str
        Name of the spin frequency parameter (default ``"F0"``).
    """

    n_terms: int = eqx.field(static=True)
    wave_sin_names: tuple[str, ...] = eqx.field(static=True)
    wave_cos_names: tuple[str, ...] = eqx.field(static=True)
    waveepoch_name: str = eqx.field(static=True, default="WAVEEPOCH")
    wave_om_name: str = eqx.field(static=True, default="WAVE_OM")
    f0_name: str = eqx.field(static=True, default="F0")

    def __check_init__(self):
        if self.n_terms < 1:
            raise ValueError("Wave requires at least one term")
        for attr in ("wave_sin_names", "wave_cos_names"):
            if len(getattr(self, attr)) != self.n_terms:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_terms ({self.n_terms})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> PhaseResult:
        """Compute Wave phase contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        PhaseResult
            Phase contribution in cycles (int + frac split).
        """
        ep_int, ep_frac = params.epoch_value(self.waveepoch_name)
        wave_om = params.param_value(self.wave_om_name)  # rad/day
        f0 = params.param_value(self.f0_name)  # Hz

        dt_int = toa_data.tdb_int - ep_int
        dt_frac = toa_data.tdb_frac - ep_frac
        dt_days = dt_int + dt_frac - delay / SECS_PER_DAY

        # Base phase in radians
        base_phase = wave_om * dt_days

        # Sum harmonics: WAVE terms use (k+1)*base_phase
        time_delay = jnp.zeros(toa_data.n_toas)  # total in seconds
        for k in range(self.n_terms):
            wave_a = params.param_value(self.wave_sin_names[k])
            wave_b = params.param_value(self.wave_cos_names[k])
            arg = (k + 1) * base_phase
            time_delay = time_delay + wave_a * jnp.sin(arg) + wave_b * jnp.cos(arg)

        # Convert to phase: seconds * Hz = cycles
        phase = time_delay * f0

        return PhaseResult.create(jnp.zeros(toa_data.n_toas), phase)
