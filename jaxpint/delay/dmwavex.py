"""Fourier-basis DM noise delay component (DMWaveX).

Ports PINT's ``DMWaveX`` class as a pure Equinox module.  The DM is
modelled as a Fourier sum and converted to delay via DM dispersion:

    DM(t) = Σ_i (DMWXSIN_i * sin(2π * DMWXFREQ_i * (t - DMWXEPOCH))
               + DMWXCOS_i * cos(2π * DMWXFREQ_i * (t - DMWXEPOCH)))

    delay = DM(t) * K_DM / freq^2

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent
from jaxpint.constants import DMCONST
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import fourier_sum


class DMWaveX(DispersionDelayComponent):
    """Fourier-basis DM noise (DMWaveX).

    Parameters
    ----------
    n_components : int
        Number of Fourier components.
    dmwxepoch_name : str
        Name of the reference epoch parameter.
    dmwxfreq_names : tuple[str, ...]
        Names of the frequency parameters (1/day).
    dmwxsin_names : tuple[str, ...]
        Names of the sine amplitude parameters (pc cm^-3).
    dmwxcos_names : tuple[str, ...]
        Names of the cosine amplitude parameters (pc cm^-3).
    """

    n_components: int = eqx.field(static=True)
    dmwxfreq_names: tuple[str, ...] = eqx.field(static=True)
    dmwxsin_names: tuple[str, ...] = eqx.field(static=True)
    dmwxcos_names: tuple[str, ...] = eqx.field(static=True)
    dmwxepoch_name: str = eqx.field(static=True, default="DMWXEPOCH")

    def __check_init__(self):
        if self.n_components < 1:
            raise ValueError("DMWaveX requires at least one component")
        for attr in ("dmwxfreq_names", "dmwxsin_names", "dmwxcos_names"):
            if len(getattr(self, attr)) != self.n_components:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_components ({self.n_components})"
                )

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        epoch = params.epoch_dual(self.dmwxepoch_name)
        dt_days = (toa_data.tdb - epoch).total

        freqs = jnp.array([params.param_value(n) for n in self.dmwxfreq_names])
        sins = jnp.array([params.param_value(n) for n in self.dmwxsin_names])
        coses = jnp.array([params.param_value(n) for n in self.dmwxcos_names])

        return fourier_sum(dt_days, freqs, sins, coses)

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DMWaveX delay contribution.

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
            DMWaveX delay in seconds.
        """
        dm = self.compute_dm(toa_data, params, delay)
        return dm * DMCONST / toa_data.freq ** 2
