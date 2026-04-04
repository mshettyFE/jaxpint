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
        """DMJUMP does not contribute to timing delay."""
        return jnp.zeros(toa_data.n_toas)
