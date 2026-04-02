"""Astrometry delay component: equatorial coordinates (ICRS).

Ports PINT's ``AstrometryEquatorial`` as a pure Equinox module.  Computes the
solar-system geometric delay (Roemer delay) and, optionally, the parallax
delay.

The Roemer delay is the projection of the observatory's SSB offset onto the
pulsar direction:

    delay = -dot(ssb_obs_pos, L_hat) / c

where L_hat is the unit vector from the SSB to the pulsar, computed from
RAJ and DECJ (with optional proper-motion correction).

All hand-coded derivatives are omitted; ``jax.jacobian`` through
``__call__`` replaces PINT's ``d_delay_astrometry_d_*`` functions.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Speed of light in km/s (ssb_obs_pos is in km, result in seconds).
_C_KM_PER_S: float = 299792.458

# 1 kpc in km (for parallax distance conversion).
_KPC_TO_KM: float = 3.0856775814913673e16


# ---------------------------------------------------------------------------
# AstrometryEquatorial
# ---------------------------------------------------------------------------

class AstrometryEquatorial(DelayComponent):
    """Geometric delay for equatorial (ICRS) sky coordinates.

    Parameters
    ----------
    raj_name, decj_name : str
        Names of the RA/DEC parameters in the ``ParameterVector`` (radians).
    pmra_name, pmdec_name : str or None
        Names of proper-motion parameters (mas/yr).  ``None`` disables PM.
    px_name : str or None
        Name of the parallax parameter (mas).  ``None`` disables parallax.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.  Required when PM is
        active; ignored otherwise.
    """

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    px_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_L_hat(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas 3"]:
        """Unit vector from SSB to pulsar in ICRS Cartesian coordinates.

        Without proper motion the direction is constant; with proper motion
        a linear correction is applied per TOA.
        """
        return compute_pulsar_direction(
            toa_data, params,
            raj_name=self.raj_name,
            decj_name=self.decj_name,
            pmra_name=self.pmra_name,
            pmdec_name=self.pmdec_name,
            posepoch_name=self.posepoch_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute geometric (Roemer + parallax) delay.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters.
        delay : array, shape (n_toas,)
            Accumulated delay from prior components (not used).

        Returns
        -------
        array, shape (n_toas,)
            Geometric delay in seconds.
        """
        L_hat = self._compute_L_hat(toa_data, params)

        # Roemer delay: projection of observer offset onto pulsar direction.
        re_dot_L = jnp.sum(toa_data.ssb_obs_pos * L_hat, axis=1)  # km
        result = -re_dot_L / _C_KM_PER_S  # seconds

        #  Lorimer & Kramer (2004), "Handbook of Pulsar Astronomy", Section 8.2.4
        # Also follows from Smart, 1977, chapter 9.
        if self.px_name is not None:
            px_mas = params.param_value(self.px_name)
            # Distance in km: L = (1/PX_arcsec) kpc = (1000/PX_mas) kpc
            L_km = (1000.0 / px_mas) * _KPC_TO_KM
            re_sqr = jnp.sum(toa_data.ssb_obs_pos ** 2, axis=1)  # km^2
            result += 0.5 * (re_sqr / L_km) * (1.0 - re_dot_L ** 2 / re_sqr) / _C_KM_PER_S

        return result
