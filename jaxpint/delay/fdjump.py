"""System-dependent frequency-dependent delay component (FDJump).

Ports PINT's ``FDJump`` class as a pure Equinox module.  Each FDJump
parameter applies a polynomial delay in log-frequency (or linear frequency)
to a subset of TOAs identified by boolean flag masks:

    delay_q = Σ_p FDpJUMPq * y^p

where y = log(f / 1 GHz) if FDJUMPLOG, else y = f / 1 GHz,
and the sum is over the polynomial orders p present for system q.

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector


class FDJump(DelayComponent):
    """System-dependent FD polynomial delay (FDJump).

    Parameters
    ----------
    fdjump_param_names : tuple[str, ...]
        Names of the FDJump parameters in the ParameterVector.
    fdjump_fd_indices : tuple[int, ...]
        Polynomial power for each parameter (1-indexed: FD1 → 1, FD2 → 2, ...).
    use_log : bool
        If True, use log(freq/GHz); if False, use freq/GHz.
    """

    fdjump_param_names: tuple[str, ...] = eqx.field(static=True)
    fdjump_fd_indices: tuple[int, ...] = eqx.field(static=True)
    use_log: bool = eqx.field(static=True, default=True)

    def __check_init__(self):
        if len(self.fdjump_param_names) != len(self.fdjump_fd_indices):
            raise ValueError(
                "fdjump_param_names and fdjump_fd_indices must have the same length"
            )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute FDJump delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (freq in MHz, flag_masks for system selection).
        params : ParameterVector
            Timing-model parameters containing FDpJUMPq values.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        array, shape (n_toas,)
            FDJump delay in seconds.
        """
        freq_ghz = toa_data.freq / 1000.0

        if self.use_log:
            y = jnp.log(freq_ghz)
            y = jnp.where(jnp.isfinite(y), y, 0.0)
        else:
            y = freq_ghz

        result = jnp.zeros(toa_data.n_toas)

        for name, fd_idx in zip(self.fdjump_param_names, self.fdjump_fd_indices):
            mask = toa_data.flag_masks.get(
                name, jnp.zeros(toa_data.n_toas, dtype=jnp.bool_)
            )
            coeff = params.param_value(name)
            result = jnp.where(mask, result + coeff * y ** fd_idx, result)

        return result
