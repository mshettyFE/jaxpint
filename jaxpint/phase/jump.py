"""Phase jump component: arbitrary phase offsets for TOA subsets.

Ports PINT's ``PhaseJump`` class as a pure Equinox module.  Each JUMP
parameter stores a time offset in seconds and is converted to phase by
multiplying by F0 (spin frequency).  Boolean masks in ``TOAData.flag_masks``
select which TOAs each jump applies to.

All derivatives are computed via JAX autodiff (d_phase/d_JUMP = F0).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl, PhaseComponent
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class PhaseJump(PhaseComponent):
    """Arbitrary phase jumps applied to TOA subsets.

    Each jump parameter value is in seconds and is converted to phase
    (cycles) by multiplying by F0.  The boolean mask for each jump is
    looked up from ``toa_data.flag_masks[jump_name]``.

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
            mask = toa_data.flag_masks.get(
                jump_name, jnp.zeros(toa_data.n_toas, dtype=jnp.bool_)
            )
            jump_val = params.param_value(jump_name)
            phase = jnp.where(mask, phase + jump_val * f0, phase)

        return DualFloat.cycles(jnp.zeros(toa_data.n_toas), phase)
