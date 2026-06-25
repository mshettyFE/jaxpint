"""Solar wind dispersion delay component.

The solar wind contributes a frequency-dependent delay through the excess dispersion
measure caused by free electrons between the observer and the pulsar.

Two geometry models are supported:

    SWM=0 (Edwards et al. 2006, MNRAS 372 1549, Section 2.5.4):
        Fixed power-law index p=2.  Geometry:
            DM_SW = NE_SW * AU^2 * rho / (r * sin(rho))
        where rho = pi - theta, theta is the pulsar-observer-Sun angle,
        and r is the observer-Sun distance.

    SWM=1 (You et al. 2012, MNRAS 422 1160; Hazboun et al. 2022, ApJ 929 39):
        Variable power-law index.  The line-of-sight integral of
        r^{-p} is computed via Gauss-Legendre quadrature on the
        substituted angular variable, giving accurate results for all
        elongation angles including near conjunction.

The electron density at 1 AU is modelled as a Taylor expansion:

    NE_SW(t) = NE_SW + NE_SW1*(t - SWEPOCH) + NE_SW2*(t - SWEPOCH)^2/2! + ...

and the delay is:

    delay = NE_SW(t) * geometry(theta, r) * K_DM / freq^2

References
----------
- Edwards et al. 2006, MNRAS, 372, 1549
- You et al. 2012, MNRAS, 422, 1160
- Hazboun et al. 2022, ApJ, 929, 39
- Madison et al. 2019, ApJ, 872, 150
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import (
    AU_KM,
    DMCONST,
    PC_TO_KM,
)
from jaxpint.delay._epoch import dt_years_from_epoch
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, ecl_to_icrs_rotation, taylor_horner


# ---------------------------------------------------------------------------
# Gauss-Legendre quadrature nodes/weights (32 points on [-1, 1])
# ---------------------------------------------------------------------------

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(32)
_GL_NODES = jnp.array(_GL_NODES)
_GL_WEIGHTS = jnp.array(_GL_WEIGHTS)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _sun_angle_and_distance(
    toa_data: TOAData,
    psr_dir: Float[Array, "n_toas 3"],
) -> tuple[Float[Array, " n_toas"], Float[Array, " n_toas"]]:
    """Compute the pulsar-observer-Sun angle and observer-Sun distance.

    Parameters
    ----------
    toa_data : TOAData
        Pre-extracted TOA data containing ``obs_sun_pos`` (km).
    psr_dir : array, shape (n_toas, 3)
        Unit vector from SSB to pulsar (ICRS).

    Returns
    -------
    theta : array, shape (n_toas,)
        Pulsar-observer-Sun angle in radians.
    r_km : array, shape (n_toas,)
        Observer-Sun distance in km.
    """
    obs_sun = toa_data.obs_sun_pos  # (n_toas, 3) km
    r_km = jnp.sqrt(jnp.sum(obs_sun**2, axis=1))
    obs_sun_hat = obs_sun / r_km[:, None]
    cos_theta = jnp.sum(obs_sun_hat * psr_dir, axis=1)
    theta = jnp.arccos(jnp.clip(cos_theta, -1.0, 1.0))
    return theta, r_km


def _solar_wind_geometry_swm0(
    theta: Float[Array, " n_toas"],
    r_km: Float[Array, " n_toas"],
) -> Float[Array, " n_toas"]:
    """SWM=0 geometry factor (Edwards et al. 2006, Eq. 29-30).

    Returns the geometry factor in parsecs:
        geometry = AU^2 * rho / (r * sin(rho))
    where rho = pi - theta.

    Since sin(rho) = sin(theta), this simplifies to:
        geometry = AU^2 * (pi - theta) / (r * sin(theta))
    """
    rho = jnp.pi - theta
    sin_theta = jnp.sin(theta)
    # Guard against sin(theta) = 0 at theta = 0 or pi.
    # Near these limits rho/sin(rho) -> 1, so ratio -> 1.
    safe_sin = jnp.where(sin_theta == 0.0, 1.0, sin_theta)
    ratio = jnp.where(sin_theta == 0.0, 1.0, rho / safe_sin)
    geometry_km = AU_KM**2 * ratio / r_km
    return geometry_km / PC_TO_KM


def _solar_wind_geometry_swm1(
    theta: Float[Array, " n_toas"],
    r_km: Float[Array, " n_toas"],
    p: Float[Array, ""],
) -> Float[Array, " n_toas"]:
    """SWM=1 geometry factor (Hazboun et al. 2022, Eq. 11).

    Computes the line-of-sight integral of r^{-p} via Gauss-Legendre
    quadrature.  With the substitution z = b*tan(u), the integral becomes:

        integral = b^{1-p} * int_{theta-pi/2}^{pi/2} cos^{p-2}(u) du

    and the geometry factor is:

        geometry = (AU / b)^p * b * integral
                 = AU^p * b^{2-2p} * int cos^{p-2}(u) du

    The Gauss-Legendre nodes are mapped from [-1, 1] to
    [theta - pi/2, pi/2].  This approach is numerically stable for all
    elongation angles (including near conjunction) and is fully
    differentiable via JAX autodiff.
    """
    sin_theta = jnp.sin(theta)

    b_km = r_km * sin_theta  # impact parameter (km)
    b_au = b_km / AU_KM  # impact parameter (AU, dimensionless value)

    # Integration limits: u in [theta - pi/2, pi/2]
    u_lo = theta - jnp.pi / 2.0  # shape (n_toas,)
    u_hi = jnp.pi / 2.0
    half_width = (u_hi - u_lo) / 2.0  # = (pi - theta) / 2
    midpoint = (u_lo + u_hi) / 2.0  # = theta / 2

    # Map GL nodes from [-1, 1] to [u_lo, u_hi]:
    # u = midpoint + half_width * t, shape (n_toas, 32)
    u = midpoint[:, None] + half_width[:, None] * _GL_NODES[None, :]

    # Integrand: cos^{p-2}(u)
    cos_u = jnp.cos(u)
    # Guard cos_u to avoid 0^negative when p < 2 at u = +/-pi/2.
    safe_cos = jnp.maximum(cos_u, 1e-30)
    integrand = safe_cos ** (p - 2.0)

    # Weighted sum: integral = half_width * sum(weights * integrand)
    integral = half_width * jnp.sum(_GL_WEIGHTS[None, :] * integrand, axis=1)

    # geometry = (1/b_au)^p * b_km * integral
    # (matches PINT's (1/b_au)^p * b * [hypergeom_term + gamma_term])
    # Guard b_au to avoid 0^(-p) at conjunction/opposition.
    safe_b_au = jnp.where(b_au == 0.0, 1.0, b_au)
    inv_b_au_p = jnp.where(b_au == 0.0, 0.0, (1.0 / safe_b_au) ** p)

    geometry_km = inv_b_au_p * b_km * integral
    return geometry_km / PC_TO_KM


class SolarWindDispersion(DelayComponent):
    """Dispersion delay from the solar wind.

    Parameters
    ----------
    ne_sw_param_names : tuple[str, ...]
        Names of the NE_SW Taylor coefficients in the ``ParameterVector``,
        ordered by derivative index.  E.g. ``("NE_SW",)`` for constant,
        or ``("NE_SW", "NE_SW1")`` for a first-order expansion.
    swepoch_name : str
        Name of the reference-epoch parameter for the NE_SW expansion.
    swm : int
        Solar wind model: 0 (Edwards et al.) or 1 (Hazboun et al.).
    swp_name : str or None
        Name of the power-law index parameter.  Required for SWM=1.
    raj_name, decj_name : str
        Astrometry coordinate parameter names (RA/DEC or ELONG/ELAT).
    pmra_name, pmdec_name : str or None
        Proper-motion parameter names.
    posepoch_name : str or None
        Position epoch parameter name.
    obliquity_arcsec : float or None
        When set, coordinates are ecliptic; rotate to ICRS.

    Raises
    ------
    ValueError
        If no NE_SW terms are provided (``ne_sw_param_names`` is empty).
    ValueError
        If the first NE_SW term is not ``'NE_SW'``.
    ValueError
        If ``swm`` is not 0 or 1.
    ValueError
        If ``swm`` is 1 and ``swp_name`` is None.
    """

    PARAMS = (
        ParamDecl("NE_SW", aliases=("NE1AU", "SOLARN0")),
        ParamDecl("NE_SW1", prefix="NE_SW"),
        ParamDecl("SWEPOCH", kind="mjd"),
        ParamDecl("SWM", kind="int"),
        ParamDecl("SWP"),
    )

    ne_sw_param_names: tuple[str, ...] = eqx.field(static=True)
    swepoch_name: str = eqx.field(static=True, default="SWEPOCH")
    swm: int = eqx.field(static=True, default=0)
    swp_name: Optional[str] = eqx.field(static=True, default=None)

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    obliquity_arcsec: Optional[float] = eqx.field(static=True, default=None)

    def __check_init__(self):
        if len(self.ne_sw_param_names) == 0:
            raise ValueError("SolarWindDispersion requires at least one NE_SW term")
        if self.ne_sw_param_names[0] != "NE_SW":
            raise ValueError(
                f"First NE_SW term must be 'NE_SW', got '{self.ne_sw_param_names[0]}'"
            )
        if self.swm not in (0, 1):
            raise ValueError(f"SWM must be 0 or 1, got {self.swm}")
        if self.swm == 1 and self.swp_name is None:
            raise ValueError("SWM=1 requires swp_name to be set")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute solar wind dispersion delay.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times, frequencies, positions).
        params : ParameterVector
            Timing-model parameters containing NE_SW, SWEPOCH, etc.
        delay : array, shape (n_toas,)
            Accumulated delay from prior components (not used).

        Returns
        -------
        array, shape (n_toas,)
            Solar wind dispersion delay in seconds.
        """
        # 1. Pulsar direction (unit vector, ICRS).
        psr_dir = compute_pulsar_direction(
            toa_data,
            params,
            raj_name=self.raj_name,
            decj_name=self.decj_name,
            pmra_name=self.pmra_name,
            pmdec_name=self.pmdec_name,
            posepoch_name=self.posepoch_name,
        )
        if self.obliquity_arcsec is not None:
            psr_dir = psr_dir @ ecl_to_icrs_rotation(self.obliquity_arcsec)

        # 2. Sun angle and distance.
        theta, r_km = _sun_angle_and_distance(toa_data, psr_dir)

        # 3. Geometry factor (parsecs).
        if self.swm == 0:
            geometry_pc = _solar_wind_geometry_swm0(theta, r_km)
        else:  # swm == 1
            assert self.swp_name is not None
            p = params.param_value(self.swp_name)
            geometry_pc = _solar_wind_geometry_swm1(theta, r_km, p)

        # 4. NE_SW Taylor expansion (cm^-3).
        dt_yr = dt_years_from_epoch(toa_data, params, self.swepoch_name)
        ne_sw_coeffs = params.param_values(self.ne_sw_param_names)
        ne_sw = taylor_horner(dt_yr, ne_sw_coeffs)

        # 5. Solar wind DM (pc / cm^3) and delay (seconds).
        dm_sw = ne_sw * geometry_pc
        return dm_sw * DMCONST / toa_data.freq**2
