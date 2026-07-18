"""Astrometry delay components: equatorial (ICRS) and ecliptic coordinates.

Both components compute the solar-system geometric delay (Roemer delay)
and, optionally, the parallax delay.

The Roemer delay is the projection of the observatory's SSB offset onto the
pulsar direction:

    delay = -dot(ssb_obs_pos, L_hat) / c

where L_hat is the unit vector from the SSB to the pulsar in ICRS.
For equatorial coordinates L_hat is computed directly from RAJ/DECJ;
for ecliptic coordinates it is computed from ELONG/ELAT and rotated to ICRS.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import C_KM_PER_S, KPC_TO_KM
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, compute_pulsar_direction_ecl

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


# ---------------------------------------------------------------------------
# Shared geometric delay computation
# ---------------------------------------------------------------------------


def _geometric_delay(
    L_hat: Float[Array, "n_toas 3"],
    toa_data: TOAData,
    params: ParameterVector,
    px_name: Optional[str],
) -> Float[Array, " n_toas"]:
    """Compute Roemer + parallax delay given pulsar direction in ICRS.

    Parameters
    ----------
    L_hat : array, shape (n_toas, 3)
        Unit vector from SSB to pulsar in ICRS Cartesian coordinates.
    toa_data : TOAData
        Pre-extracted TOA data (``ssb_obs_pos`` in km, ICRS).
    params : ParameterVector
        Timing-model parameters.
    px_name : str or None
        Parallax parameter name (mas).  ``None`` disables parallax.

    Returns
    -------
    array, shape (n_toas,)
        Geometric delay in seconds.
    """
    # Roemer delay: projection of observer offset onto pulsar direction.
    re_dot_L = jnp.sum(toa_data.ssb_obs_pos * L_hat, axis=1)  # km
    result = -re_dot_L / C_KM_PER_S  # seconds

    #  Lorimer & Kramer (2004), "Handbook of Pulsar Astronomy", Section 8.2.4
    # Also follows from Smart, 1977, chapter 9.
    if px_name is not None:
        px_mas = params.param_value(px_name)
        # Distance in km: 1/PX_mas gives kpc (1 mas parallax = 1 kpc distance)
        L_km = (1.0 / px_mas) * KPC_TO_KM
        re_sqr = jnp.sum(toa_data.ssb_obs_pos**2, axis=1)  # km^2
        # Guard against 0/0 for barycentric TOAs (ssb_obs_pos == 0) like TZR.
        # Using 1.0 as safe denominator is fine: re_sqr==0 implies re_dot_L==0,
        # so the numerator (re_sqr / L_km) is also 0 and the term vanishes.
        re_sqr_safe = jnp.where(re_sqr == 0.0, 1.0, re_sqr)
        result += (
            0.5 * (re_sqr_safe / L_km) * (1.0 - re_dot_L**2 / re_sqr_safe) / C_KM_PER_S
        )

    return result


@register_component(
    component=Component.ASTROMETRY_EQUATORIAL, pint_names=("AstrometryEquatorial",)
)
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

    PARAMS = (
        ParamDecl("RAJ", kind="angle", unit="hourangle", aliases=("RA",)),
        ParamDecl("DECJ", kind="angle", unit="deg", aliases=("DEC",)),
        ParamDecl("PMRA"),
        ParamDecl("PMDEC"),
        ParamDecl("PX"),
        ParamDecl("POSEPOCH", kind="mjd"),
    )

    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    px_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "AstrometryEquatorial":
        """Construct from a parsed model (astrometry names resolved on ``ctx``)."""
        from jaxpint._build_context import opt_name

        return cls(
            raj_name=ctx.raj,
            decj_name=ctx.decj,
            pmra_name=ctx.pmra,
            pmdec_name=ctx.pmdec,
            px_name=opt_name(ctx.par, "PX"),
            posepoch_name=ctx.posepoch,
        )

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
            toa_data,
            params,
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
        return _geometric_delay(L_hat, toa_data, params, self.px_name)


@register_component(
    component=Component.ASTROMETRY_ECLIPTIC, pint_names=("AstrometryEcliptic",)
)
class AstrometryEcliptic(DelayComponent):
    """Geometric delay for ecliptic sky coordinates.

    Computes the same Roemer + parallax delay as ``AstrometryEquatorial``,
    but the pulsar position is specified in ecliptic coordinates (ELONG/ELAT)
    which are rotated to ICRS before the delay computation.

    Parameters
    ----------
    elong_name, elat_name : str
        Names of ecliptic longitude/latitude parameters in the
        ``ParameterVector`` (radians).
    pmelong_name, pmelat_name : str or None
        Names of proper-motion parameters (mas/yr).  ``None`` disables PM.
    px_name : str or None
        Name of the parallax parameter (mas).  ``None`` disables parallax.
    posepoch_name : str or None
        Epoch parameter for proper-motion reference.  Required when PM is
        active; ignored otherwise.
    obliquity_arcsec : float
        Obliquity of the ecliptic in arcseconds (e.g. 84381.406 for
        IERS2010).  Resolved from the ECL parameter at bridge time.
    """

    PARAMS = (
        ParamDecl("ELONG", kind="angle", unit="deg", aliases=("LAMBDA",)),
        ParamDecl("ELAT", kind="angle", unit="deg", aliases=("BETA",)),
        ParamDecl("PMELONG", aliases=("PMLAMBDA",)),
        ParamDecl("PMELAT", aliases=("PMBETA",)),
        ParamDecl("PX"),
        ParamDecl("POSEPOCH", kind="mjd"),
        ParamDecl("ECL", kind="str"),
    )

    elong_name: str = eqx.field(static=True, default="ELONG")
    elat_name: str = eqx.field(static=True, default="ELAT")
    pmelong_name: Optional[str] = eqx.field(static=True, default=None)
    pmelat_name: Optional[str] = eqx.field(static=True, default=None)
    px_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    obliquity_arcsec: float = eqx.field(static=True, default=84381.406)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "AstrometryEcliptic":
        """Construct from a parsed model (astrometry names resolved on ``ctx``)."""
        from jaxpint._build_context import opt_name

        # An ecliptic-frame model always resolves obliquity in _resolve_astrometry.
        assert ctx.obliquity_arcsec is not None
        return cls(
            elong_name=ctx.raj,
            elat_name=ctx.decj,
            pmelong_name=ctx.pmra,
            pmelat_name=ctx.pmdec,
            px_name=opt_name(ctx.par, "PX"),
            posepoch_name=ctx.posepoch,
            obliquity_arcsec=ctx.obliquity_arcsec,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_L_hat(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas 3"]:
        """Unit vector from SSB to pulsar in ICRS Cartesian coordinates.

        Computes the direction in ecliptic frame from ELONG/ELAT (with
        optional proper-motion correction), then rotates to ICRS.
        """
        return compute_pulsar_direction_ecl(
            toa_data,
            params,
            elong_name=self.elong_name,
            elat_name=self.elat_name,
            pmelong_name=self.pmelong_name,
            pmelat_name=self.pmelat_name,
            posepoch_name=self.posepoch_name,
            obliquity_arcsec=self.obliquity_arcsec,
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
        return _geometric_delay(L_hat, toa_data, params, self.px_name)
