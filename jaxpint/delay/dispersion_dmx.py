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

from jaxpint.components import DispersionDelayComponent
from jaxpint.constants import DMCONST
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class DispersionDMX(DispersionDelayComponent):
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

    Raises
    ------
    ValueError
        If ``n_bins`` is less than 1.
    ValueError
        If the length of ``dmx_names``, ``dmxr1_names``, or ``dmxr2_names``
        does not match ``n_bins``.
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

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute piecewise-constant DM from DMX bins.

        Each TOA receives the DMX value of the bin it falls within.
        TOAs outside all bins receive zero DM contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (MJD times for bin assignment).
        params : ParameterVector
            Timing-model parameters containing DMX, DMXR1, DMXR2 values.
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this method).

        Returns
        -------
        array, shape (n_toas,)
            Piecewise DM in pc cm^-3 at each TOA.
        """
        toa_mjd = toa_data.mjd.total

        dm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            r1 = params.epoch_dual(self.dmxr1_names[i]).total
            r2 = params.epoch_dual(self.dmxr2_names[i]).total

            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)
            dmx_val = params.param_value(self.dmx_names[i])
            dm = dm + jnp.where(in_bin, dmx_val, 0.0)

        return dm

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
        dm = self.compute_dm(toa_data, params, delay)
        return dm * DMCONST / toa_data.freq**2
