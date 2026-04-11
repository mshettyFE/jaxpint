"""Chromatic measure delay component: CM Taylor expansion.

Ports PINT's ``ChromaticCM`` class as a pure Equinox module.  The chromatic
measure is modelled as a Taylor expansion about CMEPOCH:

    CM(t) = CM + CM1*(t - CMEPOCH) + CM2*(t - CMEPOCH)^2/2! + ...

and the delay for each TOA is:

    delay = CM(t) * K_DM * freq^(-alpha)

where freq is in MHz and alpha = TNCHROMIDX.

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.constants import DAYS_PER_JULIAN_YEAR, DMCONST
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner


class ChromaticCM(DelayComponent):
    """Chromatic measure delay using a Taylor expansion about CMEPOCH.

    Parameters
    ----------
    cm_param_names : tuple[str, ...]
        Names of the CM Taylor coefficients, ordered by derivative index.
        E.g. ``("CM",)`` for constant CM, or ``("CM", "CM1", "CM2")``.
    cmepoch_name : str
        Name of the reference-epoch parameter (default ``"CMEPOCH"``).
    tnchromidx_name : str
        Name of the chromatic index parameter (default ``"TNCHROMIDX"``).

    Raises
    ------
    ValueError
        If no CM terms are provided (``cm_param_names`` is empty).
    """

    cm_param_names: tuple[str, ...] = eqx.field(static=True)
    cmepoch_name: str = eqx.field(static=True, default="CMEPOCH")
    tnchromidx_name: str = eqx.field(static=True, default="TNCHROMIDX")

    def __check_init__(self):
        if len(self.cm_param_names) == 0:
            raise ValueError("ChromaticCM requires at least one CM term")

    def _compute_dt_yr(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Time from CMEPOCH to each TOA, in Julian years."""
        epoch = params.epoch_dual(self.cmepoch_name)
        dt_days = (toa_data.tdb - epoch).total
        return dt_days / DAYS_PER_JULIAN_YEAR

    def _get_cm_coeffs(
        self, params: ParameterVector
    ) -> Float[Array, " n_terms"]:
        """Assemble ``[CM, CM1, CM2, ...]`` for :func:`taylor_horner`."""
        return jnp.array(
            [params.param_value(name) for name in self.cm_param_names]
        )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute chromatic CM delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters containing CM, CM1, ..., CMEPOCH, TNCHROMIDX.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        array, shape (n_toas,)
            Chromatic delay in seconds.
        """
        dt_yr = self._compute_dt_yr(toa_data, params)
        cm_coeffs = self._get_cm_coeffs(params)
        cm = taylor_horner(dt_yr, cm_coeffs)
        alpha = params.param_value(self.tnchromidx_name)
        return cm * DMCONST * toa_data.freq ** (-alpha)
