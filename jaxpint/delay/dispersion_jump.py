"""Dispersion jump component (DMJUMP).

DMJUMPs apply constant DM offsets per flag group.  They affect the model DM (used in
wideband fitting) but do **not** contribute to timing delay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent, ParamDecl
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.DISPERSION_JUMP, pint_names=("DispersionJump",))
class DispersionJump(DispersionDelayComponent):
    """Constant DM offsets per flag group (DMJUMP).

    Parameters
    ----------
    dmjump_names : tuple[str, ...]
        Names of the DMJUMP parameters in the ``ParameterVector``.
        Each name must have a corresponding boolean mask in
        ``toa_data.flag_masks``.
    """

    PARAMS = (ParamDecl("DMJUMP1", kind="mask", aliases=("DMJUMP",), prefix="DMJUMP"),)

    dmjump_names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[DispersionJump]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        dmjump_names = ctx.par.params.names_with_prefix("DMJUMP")
        if not dmjump_names:
            return None
        return cls(dmjump_names=dmjump_names)

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
            mask = toa_data.flag_mask(name)
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
