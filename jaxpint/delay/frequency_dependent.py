"""Frequency-dependent delay component (FD parameters).

The delay is a polynomial in the log of the observing frequency:

    delay(f) = Σ_i FD_i * (log(f / 1 GHz))^i     (i = 1, 2, ...)

where f is the barycentric radio frequency in MHz.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.FREQUENCY_DEPENDENT, pint_names=("FD",))
class FrequencyDependent(DelayComponent):
    """Polynomial delay in log-frequency (FD model).

    Parameters
    ----------
    fd_param_names : tuple[str, ...]
        Names of the FD parameters, ordered starting from FD1.
        E.g. ``("FD1", "FD2", "FD3")``.

    Raises
    ------
    ValueError
        If no FD terms are provided (``fd_param_names`` is empty).
    """

    PARAMS = (ParamDecl("FD1", unit="s", prefix="FD"),)

    fd_param_names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[FrequencyDependent]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        fd_indices = ctx.par.params.prefix_indices("FD")
        if not fd_indices:
            return None
        return cls(fd_param_names=tuple(f"FD{i}" for i in fd_indices))

    def __check_init__(self):
        if len(self.fd_param_names) == 0:
            raise ValueError("FrequencyDependent requires at least one FD term")

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute FD delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (freq in MHz).
        params : ParameterVector
            Timing-model parameters containing FD1, FD2, etc.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        array, shape (n_toas,)
            FD delay in seconds.
        """
        # freq is in MHz; 1 GHz = 1000 MHz
        log_freq = jnp.log(toa_data.freq / 1000.0)
        log_freq = jnp.where(jnp.isfinite(log_freq), log_freq, 0.0)

        result = jnp.zeros(toa_data.n_toas)
        for i, name in enumerate(self.fd_param_names):
            coeff = params.param_value(name)
            result = result + coeff * log_freq ** (i + 1)
        return result
