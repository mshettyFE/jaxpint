"""Power-law DM noise model for JaxPINT.

Implements dispersion-measure (DM) variations with a power-law power
spectral density using a frequency-scaled Fourier basis, matching
PINT's ``PLDMNoise`` component.

The noise covariance is decomposed as::

    C_dm = F_dm · diag(w) · F_dmᵀ

where *F_dm* is a Fourier design matrix pre-scaled by ``(f_ref / f_obs)²``
(with ``f_ref = 1400 MHz``) and *w* are the power-law PSD weights
computed from the amplitude and spectral index parameters.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent, ParamDecl
from jaxpint.constants import FYR
from jaxpint.types import TOAData, ParameterVector


class PLDMNoise(NoiseComponent):
    """Power-law DM noise via frequency-scaled Fourier basis.

    The Fourier design matrix is pre-scaled by ``(1400 / f_obs)²`` by
    the bridge, so that DM's inverse-frequency-squared dependence is
    baked into the basis.  The PSD weights depend on the amplitude
    (``TNDMAMP``) and spectral index (``TNDMGAM``) parameters and are
    computed dynamically so that they are differentiable.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Pre-computed Fourier design matrix already scaled by
        ``(1400 / f_obs)²`` per TOA.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin (used to weight the PSD).
    tndmamp_name : str
        Parameter name for the log10 amplitude.
    tndmgam_name : str
        Parameter name for the spectral index.
    """

    fourier_basis: Float[Array, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]
    PARAMS = (
        ParamDecl("TNDMAMP"),
        ParamDecl("TNDMGAM"),
        ParamDecl("TNDMC", kind="int"),
        ParamDecl("TNDMTSPAN"),
    )

    tndmamp_name: str = eqx.field(static=True)
    tndmgam_name: str = eqx.field(static=True)

    def __post_init__(self):
        # Store the Fourier basis as numpy on host RAM (source of
        # truth). See PLRedNoise for the rationale.
        if not isinstance(self.fourier_basis, np.ndarray):
            object.__setattr__(
                self, "fourier_basis", np.asarray(self.fourier_basis),
            )

    @functools.cached_property
    def _fourier_basis_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Lazy device-converted view of ``fourier_basis``; see PLRedNoise."""
        return jnp.asarray(self.fourier_basis)

    def psd_weights(
        self,
        params: ParameterVector,
    ) -> Float[Array, " n_basis"]:
        """Compute power-law PSD weights for the DM noise Fourier basis.

        The power spectral density follows the convention::

            P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)

        Each weight is ``P(f) · Δf``, repeated twice for the sin/cos
        pair at that frequency.

        Parameters
        ----------
        params : ParameterVector
            Must contain values for ``TNDMAMP`` (log10 amplitude)
            and ``TNDMGAM`` (spectral index).

        Returns
        -------
        weights : (2 * n_freqs,)
            PSD weights for each basis column.
        """
        log10_A = params.param_value(self.tndmamp_name)
        gamma = params.param_value(self.tndmgam_name)
        A = 10.0 ** log10_A

        psd = (
            A ** 2
            / (12.0 * jnp.pi ** 2)
            * FYR ** (gamma - 3.0)
            * self.freqs ** (-gamma)
        )
        return jnp.repeat(psd * self.freq_bin_widths, 2)

    def static_basis(self) -> Float[Array, "n_toas n_basis"]:
        return self.fourier_basis

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Return the Woodbury ``(Ndiag, U, Phidiag)`` triple for DM noise.

        DM noise is purely low-rank: ``Ndiag = 0``.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data (used for array sizing).
        params : ParameterVector
            Current parameter values for DM amplitude and spectral index.

        Returns
        -------
        Ndiag : (n_toas,)
            Zero diagonal (DM noise has no white component).
        U : (n_toas, 2 * n_freqs)
            DM-scaled Fourier design matrix.
        Phidiag : (2 * n_freqs,)
            Power-law PSD weights.
        """
        Ndiag = jnp.zeros(toa_data.n_toas)
        return Ndiag, self._fourier_basis_jax, self.psd_weights(params)

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a random DM noise realization.

        Draws standard-normal Fourier amplitudes and projects them
        through the DM-scaled basis matrix scaled by sqrt(weights).

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data (used for basis matrix dimensions).
        params : ParameterVector
            Current parameter values for DM amplitude and spectral index.
        key : jax.Array
            PRNG key for random sampling.

        Returns
        -------
        noise : (n_toas,)
            DM noise realization in seconds.
        """
        weights = self.psd_weights(params)
        n_basis = self.fourier_basis.shape[1]
        a = jax.random.normal(key, shape=(n_basis,))
        return self._fourier_basis_jax @ (jnp.sqrt(weights) * a)
