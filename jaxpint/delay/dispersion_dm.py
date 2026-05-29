"""Dispersion delay component: DM Taylor expansion.

Ports PINT's ``DispersionDM`` class as a pure Equinox module.  The dispersion
measure is modelled as a Taylor expansion about DMEPOCH:

    DM(t) = DM + DM1*(t - DMEPOCH) + DM2*(t - DMEPOCH)^2/2! + ...

and the delay for each TOA is:

    delay = DM(t) * K_DM / freq^2

where freq is in MHz and K_DM = 1 / 2.41e-4 (MHz^2 s cm^3 / pc).

All hand-coded derivatives are omitted; ``jax.jacobian`` through
``__call__`` replaces PINT's ``d_delay_d_dmparam``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent, ParamDecl
from jaxpint.constants import DAYS_PER_JULIAN_YEAR, DMCONST
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner


# ---------------------------------------------------------------------------
# DispersionDM
# ---------------------------------------------------------------------------

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
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_dt_yr(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Time from DMEPOCH to each TOA, in Julian years.

        Uses the integer/fractional MJD split to avoid catastrophic
        cancellation when TDB and DMEPOCH are close in value.
        """
        epoch = params.epoch_dual(self.dmepoch_name)
        dt_days = (toa_data.tdb - epoch).total
        return dt_days / DAYS_PER_JULIAN_YEAR

    def _get_dm_coeffs(
        self, params: ParameterVector
    ) -> Float[Array, " n_terms"]:
        """Assemble ``[DM, DM1, DM2, ...]`` for :func:`taylor_horner`."""
        return jnp.array(
            [params.param_value(name) for name in self.dm_param_names]
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
        dt_yr = self._compute_dt_yr(toa_data, params)
        dm_coeffs = self._get_dm_coeffs(params)
        return taylor_horner(dt_yr, dm_coeffs)

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute dispersion delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, etc.).
        params : ParameterVector
            Timing-model parameters containing DM, DM1, ..., and DMEPOCH.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in **seconds**.
            Not used by this component (dispersion is frequency-dependent,
            not time-dependent), but accepted for API consistency.

        Returns
        -------
        array, shape (n_toas,)
            Dispersion delay in **seconds**.
        """
        dm = self.compute_dm(toa_data, params, delay)
        return dm * DMCONST / toa_data.freq ** 2
