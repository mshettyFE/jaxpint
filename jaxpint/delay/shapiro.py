"""Solar system Shapiro delay component.

Ports PINT's ``SolarSystemShapiro`` as a pure Equinox module.  Computes the
general-relativistic time delay caused by light passing through the
gravitational field of the Sun and (optionally) planets.

Formula (Backer & Hellings 1986, Eq 4.6 with gamma=1):

    delay = -2 * (GM/c^3) * log((r - r*cos(theta)) / AU)

where r is the observer-to-body distance, theta is the angle between the
body direction and the pulsar direction, and AU is the astronomical unit.
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import TSUN, AU_KM, PLANET_MASSES, PLANET_NAMES
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, ecl_to_icrs_rotation


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _ss_obj_shapiro_delay(
    obj_pos: Float[Array, "n_toas 3"],
    psr_dir: Float[Array, "n_toas 3"],
    T_obj: float,
) -> Float[Array, " n_toas"]:
    """Shapiro delay from a single solar system body.

    Parameters
    ----------
    obj_pos : array, shape (n_toas, 3)
        Observatory-to-object position vector in km.
    psr_dir : array, shape (n_toas, 3)
        Unit vector from SSB to pulsar (ICRS).
    T_obj : float
        Gravitational mass parameter GM/c^3 in seconds.

    Returns
    -------
    array, shape (n_toas,)
        Shapiro delay contribution in seconds.
    """
    r = jnp.sqrt(jnp.sum(obj_pos**2, axis=1))
    rcostheta = jnp.sum(obj_pos * psr_dir, axis=1)
    # Guard against log(0) when pulsar is directly behind the body.
    arg = jnp.maximum((r - rcostheta) / AU_KM, 1e-100)
    # For barycentered TOAs (r ≈ 0), the Shapiro delay is zero —
    # matches PINT's explicit skip for observatory == "barycenter".
    return jnp.where(r > 0.0, -2.0 * T_obj * jnp.log(arg), 0.0)


# ---------------------------------------------------------------------------
# SolarSystemShapiroDelay
# ---------------------------------------------------------------------------


class SolarSystemShapiroDelay(DelayComponent):
    """Solar system Shapiro delay (Sun + optional planets).

    Parameters
    ----------
    raj_name, decj_name : str
        Names of the RA/DEC (or ELONG/ELAT) parameters in the
        ``ParameterVector`` (radians).
    pmra_name, pmdec_name : str or None
        Names of proper-motion parameters (mas/yr).  None disables PM.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.
    planet_shapiro : bool
        If True, include Jupiter, Saturn, Venus, Uranus, and Neptune
        contributions in addition to the Sun.
    obliquity_arcsec : float or None
        When not None, the direction parameters are interpreted as ecliptic
        coordinates and rotated to ICRS using this obliquity (arcseconds).
    """

    PARAMS = (ParamDecl("PLANET_SHAPIRO", kind="bool"),)

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    planet_shapiro: bool = eqx.field(static=True, default=False)
    obliquity_arcsec: Optional[float] = eqx.field(static=True, default=None)

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute solar system Shapiro delay.

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
            Shapiro delay in seconds.
        """
        psr_dir = compute_pulsar_direction(
            toa_data,
            params,
            raj_name=self.raj_name,
            decj_name=self.decj_name,
            pmra_name=self.pmra_name,
            pmdec_name=self.pmdec_name,
            posepoch_name=self.posepoch_name,
        )

        # When using ecliptic coordinates, rotate direction to ICRS.
        if self.obliquity_arcsec is not None:
            psr_dir = psr_dir @ ecl_to_icrs_rotation(self.obliquity_arcsec)

        # Sun contribution (always).
        result = _ss_obj_shapiro_delay(toa_data.obs_sun_pos, psr_dir, TSUN)

        # Planet contributions (static field -> plain if is JIT-safe).
        if self.planet_shapiro:
            if toa_data.planet_positions is None:
                raise ValueError(
                    "planet_shapiro=True but toa_data.planet_positions is None"
                )
            for pl in PLANET_NAMES:
                col = f"obs_{pl}_pos"
                if col not in toa_data.planet_positions:
                    raise KeyError(f"Missing planet position '{col}' in toa_data")
                result = result + _ss_obj_shapiro_delay(
                    toa_data.planet_positions[col], psr_dir, PLANET_MASSES[pl]
                )

        return result
