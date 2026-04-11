"""Dispersion jump component (DMJUMP).

Ports PINT's ``DispersionJump`` as a pure Equinox module.  DMJUMPs apply
constant DM offsets per flag group.  They affect the model DM (used in
wideband fitting) but do **not** contribute to timing delay.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent
from jaxpint.types import TOAData, ParameterVector


class DispersionJump(DispersionDelayComponent):
    """Constant DM offsets per flag group (DMJUMP).

    Parameters
    ----------
    dmjump_names : tuple[str, ...]
        Names of the DMJUMP parameters in the ``ParameterVector``.
        Each name must have a corresponding boolean mask in
        ``toa_data.flag_masks``.
    """

    dmjump_names: tuple[str, ...] = eqx.field(static=True)

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DM offsets from DMJUMP flag masks.

        Each DMJUMP value is subtracted from the model DM for all TOAs
        matching its flag mask.  Used in wideband DM fitting.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data with ``flag_masks`` containing boolean
            masks for each DMJUMP parameter.
        params : ParameterVector
            Timing-model parameters containing DMJUMP values (pc cm^-3).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this method).

        Returns
        -------
        array, shape (n_toas,)
            DM offset in pc cm^-3 at each TOA.
        """
        dm = jnp.zeros(toa_data.n_toas)
        for name in self.dmjump_names:
            mask = toa_data.flag_masks[name]
            val = params.param_value(name)
            dm = jnp.where(mask, dm - val, dm)
        return dm

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Return zero delay; DMJUMPs affect model DM only, not timing delay.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters (unused).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused).

        Returns
        -------
        array, shape (n_toas,)
            Zeros; DMJUMP does not contribute to timing delay.
        """
        return jnp.zeros(toa_data.n_toas)
