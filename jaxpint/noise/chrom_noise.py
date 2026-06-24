"""Power-law chromatic noise model for JaxPINT.

Implements frequency-dependent chromatic noise (e.g. from ISM scattering)
with a power-law power spectral density, matching PINT's ``PLChromNoise``
component.

The noise covariance is decomposed as::

    C_chrom = F_chrom · diag(w) · F_chromᵀ

where *F_chrom* is the raw Fourier design matrix scaled at runtime by
``(f_ref / f_obs)^α`` (with ``f_ref = 1400 MHz`` and ``α = TNCHROMIDX``)
and *w* are the power-law PSD weights.

The chromatic index ``α`` may be a fittable parameter, so the scaling
is computed at runtime to maintain JAX differentiability.
"""

from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent, ParamDecl
from jaxpint.constants import FYR
from jaxpint.types import TOAData, ParameterVector


class PLChromNoise(NoiseComponent):
    """Power-law chromatic noise with arbitrary chromatic index.

    The raw Fourier design matrix is stored unscaled.  At each call to
    :meth:`covariance` or :meth:`generate`, the basis is multiplied by
    ``(f_ref / f_obs)^α`` where ``α`` comes from the ``TNCHROMIDX``
    parameter, making the scaling differentiable through ``α``.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Raw (unscaled) Fourier design matrix with alternating sin/cos
        columns.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin.
    tnchromamp_name : str
        Parameter name for the log10 amplitude.
    tnchromgam_name : str
        Parameter name for the spectral index.
    tnchromidx_name : str
        Parameter name for the chromatic index (α).
    fref : float
        Reference radio frequency in MHz (default 1400.0).
    """

    fourier_basis: Float[Array, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]
    PARAMS = (
        ParamDecl("TNCHROMAMP"),
        ParamDecl("TNCHROMGAM"),
        ParamDecl("TNCHROMIDX"),
        ParamDecl("TNCHROMC", kind="int"),
        ParamDecl("TNCHROMTSPAN"),
    )

    tnchromamp_name: str = eqx.field(static=True)
    tnchromgam_name: str = eqx.field(static=True)
    tnchromidx_name: str = eqx.field(static=True)
    fref: float = eqx.field(static=True, default=1400.0)

    def psd_weights(
        self,
        params: ParameterVector,
    ) -> Float[Array, " n_basis"]:
        """Compute power-law PSD weights for the chromatic noise Fourier basis.

        The power spectral density follows the convention::

            P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)

        Each weight is ``P(f) · Δf``, repeated twice for the sin/cos
        pair at that frequency.

        Parameters
        ----------
        params : ParameterVector
            Must contain values for ``TNCHROMAMP`` (log10 amplitude)
            and ``TNCHROMGAM`` (spectral index).

        Returns
        -------
        weights : (2 * n_freqs,)
            PSD weights for each basis column.
        """
        log10_A = params.param_value(self.tnchromamp_name)
        gamma = params.param_value(self.tnchromgam_name)
        A = 10.0 ** log10_A

        psd = (
            A ** 2
            / (12.0 * jnp.pi ** 2)
            * FYR ** (gamma - 3.0)
            * self.freqs ** (-gamma)
        )
        return jnp.repeat(psd * self.freq_bin_widths, 2)

    def _scaled_basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """Return Fourier basis scaled by ``(f_ref / f_obs)^α``.

        Parameters
        ----------
        toa_data : TOAData
            Must contain ``freq`` in MHz.
        params : ParameterVector
            Must contain the chromatic index parameter.

        Returns
        -------
        F_chrom : (n_toas, 2 * n_freqs)
        """
        alpha = params.param_value(self.tnchromidx_name)
        D = (self.fref / toa_data.freq) ** alpha  # (n_toas,)
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
        """Return the Woodbury ``(Ndiag, U, Phidiag)`` triple for chromatic noise.

        Chromatic noise is purely low-rank: ``Ndiag = 0``. The basis is
        scaled at runtime by ``(f_ref / f_obs)^alpha`` to account for the
        chromatic index.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including radio frequencies for chromatic scaling.
        params : ParameterVector
            Current parameter values for amplitude, spectral index, and
            chromatic index.

        Returns
        -------
        Ndiag : (n_toas,)
            Zero diagonal (chromatic noise has no white component).
        U : (n_toas, 2 * n_freqs)
            Chromatically-scaled Fourier design matrix.
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
        """Draw a random chromatic noise realization.

        Draws standard-normal Fourier amplitudes and projects them
        through the chromatically-scaled basis matrix.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including radio frequencies for chromatic scaling.
        params : ParameterVector
            Current parameter values for amplitude, spectral index, and
            chromatic index.
        key : jax.Array
            PRNG key for random sampling.

        Returns
        -------
        noise : (n_toas,)
            Chromatic noise realization in seconds.
        """
        weights = self.psd_weights(params)
        basis = self._scaled_basis(toa_data, params)
        n_basis = basis.shape[1]
        a = jax.random.normal(key, shape=(n_basis,))
        return basis @ (jnp.sqrt(weights) * a)
