"""Piecewise dispersion delay component (DMX).

Ports PINT's ``DispersionDMX`` class as a pure Equinox module.  The dispersion
measure is modelled as piecewise-constant within user-defined MJD bins:

    DM(t) = Σ DMX_i   for each bin i where DMXR1_i <= t <= DMXR2_i

and the delay for each TOA is:

    delay = DM(t) * K_DM / freq^2

where freq is in MHz and K_DM = 1 / 2.41e-4 (MHz^2 s cm^3 / pc).

All derivatives are handled automatically by ``jax.jacobian`` through
``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.constants import DMCONST
from jaxpint.types import TOAData, ParameterVector


class DispersionDMX(DelayComponent):
    """Piecewise-constant DM dispersion delay (DMX model).

    Parameters
    ----------
    n_bins : int
        Number of DMX bins.
    dmx_names : tuple[str, ...]
        Names of DMX value parameters, e.g. ``("DMX_0001", "DMX_0002")``.
    dmxr1_names : tuple[str, ...]
        Names of bin-start MJD epoch parameters, e.g. ``("DMXR1_0001", ...)``.
    dmxr2_names : tuple[str, ...]
        Names of bin-end MJD epoch parameters, e.g. ``("DMXR2_0001", ...)``.
    """

    n_bins: int = eqx.field(static=True)
    dmx_names: tuple[str, ...] = eqx.field(static=True)
    dmxr1_names: tuple[str, ...] = eqx.field(static=True)
    dmxr2_names: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        if self.n_bins < 1:
            raise ValueError("DispersionDMX requires at least one bin")
        for attr in ("dmx_names", "dmxr1_names", "dmxr2_names"):
            if len(getattr(self, attr)) != self.n_bins:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_bins ({self.n_bins})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute piecewise dispersion delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (MJD times, frequencies, etc.).
        params : ParameterVector
            Timing-model parameters containing DMX, DMXR1, DMXR2 values.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in **seconds**.
            Not used by this component.

        Returns
        -------
        array, shape (n_toas,)
            Dispersion delay in **seconds**.
        """
        # TOA MJD in UTC — matches PINT's bin assignment on tbl["mjd_float"]
        toa_mjd = toa_data.mjd_int + toa_data.mjd_frac

        dm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            # Bin boundaries (MJD epoch parameters with int/frac split)
            r1_int, r1_frac = params.epoch_value(self.dmxr1_names[i])
            r2_int, r2_frac = params.epoch_value(self.dmxr2_names[i])
            r1 = r1_int + r1_frac
            r2 = r2_int + r2_frac

            # Boolean mask: TOA falls within this bin (inclusive)
            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)

            # DM value for this bin
            dmx_val = params.param_value(self.dmx_names[i])

            # Accumulate DM contribution
            dm = dm + jnp.where(in_bin, dmx_val, 0.0)

        return dm * DMCONST / toa_data.freq**2
