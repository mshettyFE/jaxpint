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

from jaxpint.components import DelayComponent
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction

# ---------------------------------------------------------------------------
# Constants (JPL Solar System Constants)
# ---------------------------------------------------------------------------

# Solar mass parameter in seconds: GM_sun / c^3.
_TSUN: float = 4.92549094830932e-6

# Astronomical unit in km (IAU 2012 exact definition).
_AU_KM: float = 149597870.7

# Planet mass parameters in seconds: T_planet = T_sun / mass_ratio.
_PLANET_MASSES: dict[str, float] = {
    "jupiter": _TSUN / 1047.3486,
    "saturn": _TSUN / 3497.898,
    "venus": _TSUN / 408523.71,
    "uranus": _TSUN / 22902.98,
    "neptune": _TSUN / 19412.24,
}

_PLANET_NAMES: tuple[str, ...] = ("jupiter", "saturn", "venus", "uranus", "neptune")


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
    r = jnp.sqrt(jnp.sum(obj_pos ** 2, axis=1))
    rcostheta = jnp.sum(obj_pos * psr_dir, axis=1)
    # Guard against log(0) for TZR TOA where obs_sun_pos is zeros.
    arg = jnp.maximum((r - rcostheta) / _AU_KM, 1e-100)
    return -2.0 * T_obj * jnp.log(arg)


# ---------------------------------------------------------------------------
# SolarSystemShapiroDelay
# ---------------------------------------------------------------------------

class SolarSystemShapiroDelay(DelayComponent):
    """Solar system Shapiro delay (Sun + optional planets).

    Parameters
    ----------
    raj_name, decj_name : str
        Names of the RA/DEC parameters in the ``ParameterVector`` (radians).
    pmra_name, pmdec_name : str or None
        Names of proper-motion parameters (mas/yr).  None disables PM.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.
    planet_shapiro : bool
        If True, include Jupiter, Saturn, Venus, Uranus, and Neptune
        contributions in addition to the Sun.
    """

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    planet_shapiro: bool = eqx.field(static=True, default=False)

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
            toa_data, params,
            raj_name=self.raj_name,
            decj_name=self.decj_name,
            pmra_name=self.pmra_name,
            pmdec_name=self.pmdec_name,
            posepoch_name=self.posepoch_name,
        )

        # Sun contribution (always).
        result = _ss_obj_shapiro_delay(toa_data.obs_sun_pos, psr_dir, _TSUN)

        # Planet contributions (static field -> plain if is JIT-safe).
        if self.planet_shapiro:
            for pl in _PLANET_NAMES:
                col = f"obs_{pl}_pos"
                result = result + _ss_obj_shapiro_delay(
                    toa_data.planet_positions[col], psr_dir, _PLANET_MASSES[pl]
                )

        return result
