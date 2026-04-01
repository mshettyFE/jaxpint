"""Base component types for JaxPINT timing model modules."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector
from jaxpint.phase_result import PhaseResult


class PhaseComponent(eqx.Module):
    """Base class for components that contribute to pulse phase.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> PhaseResult``.

    In the timing model, all PhaseComponents see the same total delay
    and their phase contributions are summed.
    """

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> PhaseResult:
        raise NotImplementedError


class DelayComponent(eqx.Module):
    """Base class for components that contribute to signal delay.

    Subclasses implement ``__call__(self, toa_data, params, delay) -> Array``.

    In the timing model, DelayComponents are applied sequentially:
    each component sees the accumulated delay from prior components.
    """

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        raise NotImplementedError
