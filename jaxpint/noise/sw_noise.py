"""Power-law solar wind DM noise model for JaxPINT.

Implements solar wind DM perturbations with a power-law power spectral
density, matching PINT's ``PLSWNoise`` component.  Commonly used as
stochastic perturbations on top of a deterministic solar wind model.

The noise covariance is decomposed as::

    C_sw = F_sw · diag(w) · F_swᵀ

where *F_sw* is the Fourier design matrix scaled at runtime by the
solar wind geometry factor and ``DMCONST / f_obs²``, and *w* are
the power-law PSD weights.

The geometry factor depends on the pulsar direction and observer-Sun
position, so it must be computed at runtime.  This component reuses
the geometry functions from :mod:`jaxpint.delay.solar_wind`.

References
----------
- Hazboun et al. 2022, APJ, Volume 929, Issue 1, id.39
- Susurla et al. 2024, A&A, Volume 692, id.A18
"""

from __future__ import annotations

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent, ParamDecl
from jaxpint.constants import DMCONST, FYR
from jaxpint.delay.solar_wind import (
    _solar_wind_geometry_swm0,
    _solar_wind_geometry_swm1,
    _sun_angle_and_distance,
)
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import compute_pulsar_direction, ecl_to_icrs_rotation


class PLSWNoise(NoiseComponent):
    """Power-law solar wind DM noise.

    The raw Fourier design matrix is stored unscaled.  At each call the
    basis is multiplied by ``geometry_pc · DMCONST / f_obs²`` where the
    geometry factor is computed from the solar wind model (SWM=0 or 1).

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

    fourier_basis: Float[Array, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]
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

    def psd_weights(
        self,
        params: ParameterVector,
    ) -> Float[Array, " n_basis"]:
        """Compute power-law PSD weights for the solar wind noise Fourier basis.

        The power spectral density follows the convention::

            P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)

        Each weight is ``P(f) · Δf``, repeated twice for the sin/cos
        pair at that frequency.

        Parameters
        ----------
        params : ParameterVector
            Must contain values for ``TNSWAMP`` (log10 amplitude)
            and ``TNSWGAM`` (spectral index).

        Returns
        -------
        weights : (2 * n_freqs,)
            PSD weights for each basis column.
        """
        log10_A = params.param_value(self.tnswamp_name)
        gamma = params.param_value(self.tnswgam_name)
        A = 10.0 ** log10_A

        psd = (
            A ** 2
            / (12.0 * jnp.pi ** 2)
            * FYR ** (gamma - 3.0)
            * self.freqs ** (-gamma)
        )
        return jnp.repeat(psd * self.freq_bin_widths, 2)

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
            toa_data, params,
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
            p = params.param_value(self.swp_name)
            geometry_pc = _solar_wind_geometry_swm1(theta, r_km, p)

        # 4. Solar wind DM scaling (same as PINT: geometry * DMconst / freq^2).
        return geometry_pc * DMCONST / toa_data.freq ** 2

    def _scaled_basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """Return Fourier basis scaled by solar wind geometry."""
        D = self._sw_scaling(toa_data, params)  # (n_toas,)
        return self.fourier_basis * D[:, None]

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return the Woodbury ``(Ndiag, U, Phidiag)`` triple for solar wind noise.

        Solar wind noise is purely low-rank: ``Ndiag = 0``. The basis is
        scaled at runtime by the solar wind geometry factor and
        ``DMCONST / f_obs^2``.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including Sun positions and radio frequencies.
        params : ParameterVector
            Current parameter values for amplitude, spectral index, and
            astrometry parameters.

        Returns
        -------
        Ndiag : (n_toas,)
            Zero diagonal (solar wind noise has no white component).
        U : (n_toas, 2 * n_freqs)
            Solar-wind-geometry-scaled Fourier design matrix.
        Phidiag : (2 * n_freqs,)
            Power-law PSD weights.
        """
        Ndiag = jnp.zeros(toa_data.n_toas)
        return Ndiag, self._scaled_basis(toa_data, params), self.psd_weights(params)

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a random solar wind noise realization.

        Draws standard-normal Fourier amplitudes and projects them
        through the SW-geometry-scaled basis matrix.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including Sun positions and radio frequencies.
        params : ParameterVector
            Current parameter values for amplitude, spectral index, and
            astrometry parameters.
        key : jax.Array
            PRNG key for random sampling.

        Returns
        -------
        noise : (n_toas,)
            Solar wind noise realization in seconds.
        """
        weights = self.psd_weights(params)
        basis = self._scaled_basis(toa_data, params)
        n_basis = basis.shape[1]
        a = jax.random.normal(key, shape=(n_basis,))
        return basis @ (jnp.sqrt(weights) * a)
