"""Fourier-basis timing noise delay component (WaveX).

Ports PINT's ``WaveX`` class as a pure Equinox module.  The delay is
modelled as a sum of sinusoids at specified frequencies:

    delay(t) = Σ_i (WXSIN_i * sin(2π * WXFREQ_i * (t - WXEPOCH - delay))
                  + WXCOS_i * cos(2π * WXFREQ_i * (t - WXEPOCH - delay)))

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import fourier_sum


class WaveX(DelayComponent):
    """Fourier-basis timing noise (WaveX).

    Parameters
    ----------
    n_components : int
        Number of Fourier components.
    wxepoch_name : str
        Name of the reference epoch parameter.
    wxfreq_names : tuple[str, ...]
        Names of the frequency parameters (1/day).
    wxsin_names : tuple[str, ...]
        Names of the sine amplitude parameters (seconds).
    wxcos_names : tuple[str, ...]
        Names of the cosine amplitude parameters (seconds).
    """

    n_components: int = eqx.field(static=True)
    wxfreq_names: tuple[str, ...] = eqx.field(static=True)
    wxsin_names: tuple[str, ...] = eqx.field(static=True)
    wxcos_names: tuple[str, ...] = eqx.field(static=True)
    wxepoch_name: str = eqx.field(static=True, default="WXEPOCH")

    def __check_init__(self):
        if self.n_components < 1:
            raise ValueError("WaveX requires at least one component")
        for attr in ("wxfreq_names", "wxsin_names", "wxcos_names"):
            if len(getattr(self, attr)) != self.n_components:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_components ({self.n_components})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute WaveX delay contribution.

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
        array, shape (n_toas,)
            WaveX delay in seconds.
        """
        epoch = params.epoch_dual(self.wxepoch_name)
        dt_days = (toa_data.tdb - epoch).total - delay / SECS_PER_DAY

        freqs = jnp.array([params.param_value(n) for n in self.wxfreq_names])
        sins = jnp.array([params.param_value(n) for n in self.wxsin_names])
        coses = jnp.array([params.param_value(n) for n in self.wxcos_names])

        return fourier_sum(dt_days, freqs, sins, coses)
