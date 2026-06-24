"""Frequency-dependent delay component (FD parameters).

Ports PINT's ``FD`` class as a pure Equinox module.  The delay is a
polynomial in the log of the observing frequency:

    delay(f) = Σ_i FD_i * (log(f / 1 GHz))^i     (i = 1, 2, ...)

where f is the barycentric radio frequency in MHz.

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.types import TOAData, ParameterVector


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
