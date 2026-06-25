"""Chromatic measure delay component: CM Taylor expansion.

The chromatic measure is modelled as a Taylor expansion about CMEPOCH:

    CM(t) = CM + CM1*(t - CMEPOCH) + CM2*(t - CMEPOCH)^2/2! + ...

and the delay for each TOA is:

    delay = CM(t) * K_DM * freq^(-alpha)

where freq is in MHz and alpha = TNCHROMIDX.

"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import DMCONST
from jaxpint.delay._epoch import dt_years_from_epoch
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

    PARAMS = (
        ParamDecl("CM"),
        ParamDecl("CM1", prefix="CM"),
        ParamDecl("CMEPOCH", kind="mjd"),
        ParamDecl("TNCHROMIDX"),
    )

    cm_param_names: tuple[str, ...] = eqx.field(static=True)
    cmepoch_name: str = eqx.field(static=True, default="CMEPOCH")
    tnchromidx_name: str = eqx.field(static=True, default="TNCHROMIDX")

    def __check_init__(self):
        if len(self.cm_param_names) == 0:
            raise ValueError("ChromaticCM requires at least one CM term")

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
        dt_yr = dt_years_from_epoch(toa_data, params, self.cmepoch_name)
        cm_coeffs = params.param_values(self.cm_param_names)
        cm = taylor_horner(dt_yr, cm_coeffs)
        alpha = params.param_value(self.tnchromidx_name)
        return cm * DMCONST * toa_data.freq ** (-alpha)
