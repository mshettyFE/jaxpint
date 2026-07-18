"""Power-law solar wind DM noise model for JaxPINT.

Implements solar wind DM perturbations with a power-law power spectral
density, matching PINT's ``PLSWNoise`` component.  Commonly used as
stochastic perturbations on top of a deterministic solar wind model.

The noise covariance is decomposed as::

    C_sw = F_sw · diag(w) · F_swᵀ

where *F_sw* is the Fourier design matrix scaled at runtime by the solar wind
geometry factor and ``DMCONST / f_obs²``, and *w* are the power-law PSD weights.
The geometry factor depends on the pulsar direction and observer-Sun position
(and fitted astrometry), so the basis is computed at runtime (dynamic-basis
component) reusing the geometry functions from :mod:`jaxpint.delay.solar_wind`.
Shared machinery lives in
:class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.

References
----------
- Hazboun et al. 2022, APJ, Volume 929, Issue 1, id.39
- Susurla et al. 2024, A&A, Volume 692, id.A18
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.constants import DMCONST
from jaxpint.delay.solar_wind import (
    _solar_wind_geometry_swm0,
    _solar_wind_geometry_swm1,
    _sun_angle_and_distance,
)
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, ecl_to_icrs_rotation

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.PL_SW_NOISE, pint_names=("PLSWNoise",))
class PLSWNoise(_PowerLawFourierNoise):
    """Power-law solar wind DM noise.

    The raw Fourier design matrix is scaled per TOA by
    ``geometry_pc · DMCONST / f_obs²`` (geometry from the SWM=0 or SWM=1 model),
    so this is a dynamic-basis component.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Raw (unscaled) Fourier design matrix.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin.
    tnswamp_name : str
        Parameter name for the log10 amplitude.
    tnswgam_name : str
        Parameter name for the spectral index.
    swm : int
        Solar wind model (0 or 1).
    swp_name : str or None
        Parameter name for the radial power-law index (SWM=1 only).
    raj_name, decj_name : str
        Astrometry parameter names for pulsar direction.
    pmra_name, pmdec_name : str or None
        Proper motion parameter names.
    posepoch_name : str or None
        Position epoch parameter name.
    obliquity_arcsec : float or None
        Obliquity in arcseconds (set when using ecliptic coordinates).
    """

    PARAMS = (
        ParamDecl("TNSWAMP"),
        ParamDecl("TNSWGAM"),
        ParamDecl("TNSWC", kind="int"),
    )

    tnswamp_name: str = eqx.field(static=True)
    tnswgam_name: str = eqx.field(static=True)
    swm: int = eqx.field(static=True)
    swp_name: Optional[str] = eqx.field(static=True, default=None)
    raj_name: str = eqx.field(static=True, default="RAJ")
    decj_name: str = eqx.field(static=True, default="DECJ")
    pmra_name: Optional[str] = eqx.field(static=True, default=None)
    pmdec_name: Optional[str] = eqx.field(static=True, default=None)
    posepoch_name: Optional[str] = eqx.field(static=True, default=None)
    obliquity_arcsec: Optional[float] = eqx.field(static=True, default=None)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PLSWNoise]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        import jax.numpy as jnp
        from jaxpint._build_context import basis_seconds, span_seconds
        from jaxpint.utils import build_fourier_basis

        par = ctx.par
        toa_data = ctx.toa_data
        if toa_data is None:
            return None
        basis_s = basis_seconds(toa_data)
        n_freqs = par.int_params.get("TNSWC", 100)
        T = span_seconds(par, basis_s)

        F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

        swm = par.int_params.get("SWM", 0)
        swp_name = "SWP" if swm == 1 else None

        return cls(
            fourier_basis=jnp.asarray(F),
            freqs=jnp.asarray(freqs),
            freq_bin_widths=jnp.asarray(freq_bin_widths),
            tnswamp_name="TNSWAMP",
            tnswgam_name="TNSWGAM",
            swm=swm,
            swp_name=swp_name,
            raj_name=ctx.raj,
            decj_name=ctx.decj,
            pmra_name=ctx.pmra,
            pmdec_name=ctx.pmdec,
            posepoch_name=ctx.posepoch,
            obliquity_arcsec=ctx.obliquity_arcsec,
        )

    @property
    def _amp_name(self) -> str:
        return self.tnswamp_name

    @property
    def _gam_name(self) -> str:
        return self.tnswgam_name

    def _sw_scaling(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute per-TOA solar wind scaling factor.

        Returns ``geometry_pc · DMCONST / f_obs²`` for each TOA, where
        ``geometry_pc`` is the solar wind geometry factor in parsecs.
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

        # 4. Solar wind DM scaling (same as PINT: geometry * DMconst / freq^2).
        return geometry_pc * DMCONST / toa_data.freq**2

    def _basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """Fourier basis scaled per TOA by the solar wind geometry factor."""
        D = self._sw_scaling(toa_data, params)  # (n_toas,)
        return self._fourier_basis_jax * D[:, None]
