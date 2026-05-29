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

from jaxpint.components import DelayComponent, ParamDecl
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

    Raises
    ------
    ValueError
        If ``fdjump_param_names`` and ``fdjump_fd_indices`` have different
        lengths.
    """

    PARAMS = (
        ParamDecl("FD1JUMP1", kind="mask", unit="s", aliases=("FD1JUMP",), prefix="FD1JUMP"),
        ParamDecl("FD2JUMP1", kind="mask", unit="s", aliases=("FD2JUMP",), prefix="FD2JUMP"),
        ParamDecl("FD3JUMP1", kind="mask", unit="s", aliases=("FD3JUMP",), prefix="FD3JUMP"),
        ParamDecl("FD4JUMP1", kind="mask", unit="s", aliases=("FD4JUMP",), prefix="FD4JUMP"),
        ParamDecl("FD5JUMP1", kind="mask", unit="s", aliases=("FD5JUMP",), prefix="FD5JUMP"),
        ParamDecl("FD6JUMP1", kind="mask", unit="s", aliases=("FD6JUMP",), prefix="FD6JUMP"),
        ParamDecl("FD7JUMP1", kind="mask", unit="s", aliases=("FD7JUMP",), prefix="FD7JUMP"),
        ParamDecl("FD8JUMP1", kind="mask", unit="s", aliases=("FD8JUMP",), prefix="FD8JUMP"),
        ParamDecl("FD9JUMP1", kind="mask", unit="s", aliases=("FD9JUMP",), prefix="FD9JUMP"),
        ParamDecl("FD10JUMP1", kind="mask", unit="s", aliases=("FD10JUMP",), prefix="FD10JUMP"),
        ParamDecl("FD11JUMP1", kind="mask", unit="s", aliases=("FD11JUMP",), prefix="FD11JUMP"),
        ParamDecl("FD12JUMP1", kind="mask", unit="s", aliases=("FD12JUMP",), prefix="FD12JUMP"),
        ParamDecl("FD13JUMP1", kind="mask", unit="s", aliases=("FD13JUMP",), prefix="FD13JUMP"),
        ParamDecl("FD14JUMP1", kind="mask", unit="s", aliases=("FD14JUMP",), prefix="FD14JUMP"),
        ParamDecl("FD15JUMP1", kind="mask", unit="s", aliases=("FD15JUMP",), prefix="FD15JUMP"),
        ParamDecl("FD16JUMP1", kind="mask", unit="s", aliases=("FD16JUMP",), prefix="FD16JUMP"),
        ParamDecl("FD17JUMP1", kind="mask", unit="s", aliases=("FD17JUMP",), prefix="FD17JUMP"),
        ParamDecl("FD18JUMP1", kind="mask", unit="s", aliases=("FD18JUMP",), prefix="FD18JUMP"),
        ParamDecl("FD19JUMP1", kind="mask", unit="s", aliases=("FD19JUMP",), prefix="FD19JUMP"),
        ParamDecl("FD20JUMP1", kind="mask", unit="s", aliases=("FD20JUMP",), prefix="FD20JUMP"),
        ParamDecl("FDJUMPLOG", kind="bool"),
    )

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
