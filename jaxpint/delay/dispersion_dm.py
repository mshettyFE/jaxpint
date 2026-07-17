"""Dispersion delay component: DM Taylor expansion.

The dispersion measure is modelled as a Taylor expansion about DMEPOCH:

    DM(t) = DM + DM1*(t - DMEPOCH) + DM2*(t - DMEPOCH)^2/2! + ...

and the delay for each TOA is:

    delay = DM(t) * K_DM / freq^2

where freq is in MHz and K_DM = 1 / 2.41e-4 (MHz^2 s cm^3 / pc).

"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent, ParamDecl
from jaxpint.delay._epoch import dt_years_from_epoch
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner


class DispersionDM(DispersionDelayComponent):
    """DM dispersion delay using a Taylor expansion about DMEPOCH.

    Parameters
    ----------
    dm_param_names : tuple[str, ...]
        Names of the DM Taylor coefficients in the ``ParameterVector``,
        ordered by derivative index.  E.g. ``("DM",)`` for constant DM,
        or ``("DM", "DM1", "DM2")`` for a second-order expansion.
    dmepoch_name : str
        Name of the reference-epoch parameter (default ``"DMEPOCH"``).

    Raises
    ------
    ValueError
        If no DM terms are provided (``dm_param_names`` is empty).
    ValueError
        If the first DM term is not ``'DM'``.
    """

    PARAMS = (
        ParamDecl("DM"),
        ParamDecl("DM1", prefix="DM"),
        ParamDecl("DMEPOCH", kind="mjd"),
    )

    dm_param_names: tuple[str, ...] = eqx.field(static=True)
    dmepoch_name: str = eqx.field(static=True, default="DMEPOCH")

    def __check_init__(self):
        if len(self.dm_param_names) == 0:
            raise ValueError("DispersionDM requires at least one DM term")
        if self.dm_param_names[0] != "DM":
            raise ValueError(
                f"First DM term must be 'DM', got '{self.dm_param_names[0]}'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Evaluate the DM Taylor expansion at each TOA.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times used for dt from DMEPOCH).
        params : ParameterVector
            Timing-model parameters containing DM, DM1, ..., and DMEPOCH.
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this method).

        Returns
        -------
        array, shape (n_toas,)
            Dispersion measure in pc cm^-3 at each TOA.
        """
        dt_yr = dt_years_from_epoch(toa_data, params, self.dmepoch_name)
        dm_coeffs = params.param_values(self.dm_param_names)
        return taylor_horner(dt_yr, dm_coeffs)
