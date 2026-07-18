"""Phase jump component: arbitrary phase offsets for TOA subsets.

Each JUMP parameter stores a time offset in seconds and is converted to phase by
multiplying by F0 (spin frequency).  Boolean masks in ``TOAData.flag_masks``
select which TOAs each jump applies to.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl, PhaseComponent
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.PHASE_JUMP, pint_names=("PhaseJump",))
class PhaseJump(PhaseComponent):
    """Arbitrary phase jumps applied to TOA subsets.

    Each jump parameter value is in seconds and is converted to phase
    (cycles) by multiplying by F0.  The boolean mask for each jump is
    looked up via ``toa_data.flag_mask(jump_name, default=False)``.

    Parameters
    ----------
    jump_param_names : tuple[str, ...]
        Names of the JUMP parameters in the ``ParameterVector``,
        e.g. ``("JUMP1", "JUMP2", "JUMP3")``.
    f0_name : str
        Name of the spin frequency parameter (default ``"F0"``).

    Raises
    ------
    ValueError
        If no JUMP parameters are provided (``jump_param_names`` is empty).
    """

    PARAMS = (
        ParamDecl("JUMP1", kind="mask", unit="s", aliases=("JUMP",), prefix="JUMP"),
    )

    jump_param_names: tuple[str, ...] = eqx.field(static=True)
    f0_name: str = eqx.field(static=True, default="F0")

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PhaseJump]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        jump_names = ctx.par.params.names_with_prefix("JUMP")
        if not jump_names:
            return None
        return cls(jump_param_names=jump_names)

    def __check_init__(self):
        if len(self.jump_param_names) == 0:
            raise ValueError("PhaseJump requires at least one JUMP parameter")

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute phase jump contributions.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data with ``flag_masks`` containing boolean
            masks for each JUMP parameter.
        params : ParameterVector
            Timing-model parameters containing JUMP values (seconds) and F0 (Hz).
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this component).

        Returns
        -------
        DualFloat
            Phase contribution in cycles (int + frac split).
        """
        f0 = params.param_value(self.f0_name)
        phase = jnp.zeros(toa_data.n_toas)

        for jump_name in self.jump_param_names:
            mask = toa_data.flag_mask(jump_name, default=False)
            jump_val = params.param_value(jump_name)
            phase = jnp.where(mask, phase + jump_val * f0, phase)

        return DualFloat.from_cycles(jnp.zeros(toa_data.n_toas), phase)
