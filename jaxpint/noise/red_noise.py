"""Power-law red noise model for JaxPINT.

Implements achromatic red noise with a power-law power spectral density
using an alternating Fourier basis (sin/cos pairs), matching PINT's
``PLRedNoise`` component.

The noise covariance is decomposed as::

    C_rn = F · diag(w) · Fᵀ

where *F* is a Fourier design matrix (pre-computed by the bridge) and
*w* are the power-law PSD weights computed from the amplitude and
spectral index parameters.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.constants import FYR
from jaxpint.types import TOAData, ParameterVector


class PLRedNoise(NoiseComponent):
    """Power-law red noise via alternating Fourier basis.

    The Fourier design matrix *F* is pre-computed by the bridge from
    TOA times and stored as a JAX array.  The PSD weights depend on
    the amplitude (``TNREDAMP``) and spectral index (``TNREDGAM``)
    parameters and are computed dynamically so that they are
    differentiable.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Pre-computed Fourier design matrix with alternating sin/cos
        columns: ``[sin(2πf₁t), cos(2πf₁t), sin(2πf₂t), ...]``.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin (used to weight the PSD).
    tnredamp_name : str
        Parameter name for the log10 amplitude.
    tnredgam_name : str
        Parameter name for the spectral index.
    """

    fourier_basis: Float[Array, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]
    tnredamp_name: str = eqx.field(static=True)
    tnredgam_name: str = eqx.field(static=True)

    def __post_init__(self):
        # Store the Fourier basis as numpy on host RAM (source of
        # truth). The JAX-converted view used on the hot path is built
        # lazily and cached via ``_fourier_basis_jax``; see NoiseModel
        # module docstring for the rationale (discovery's
        # ``jnparray()``-in-closure pattern).
        if not isinstance(self.fourier_basis, np.ndarray):
            object.__setattr__(
                self, "fourier_basis", np.asarray(self.fourier_basis),
            )

    @functools.cached_property
    def _fourier_basis_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Lazy device-converted view of ``fourier_basis``.

        Cached on ``self.__dict__`` for the lifetime of this instance:
        the host→device transfer fires once on first access, the device
        buffer is reused on every subsequent call.
        """
        return jnp.asarray(self.fourier_basis)

    def psd_weights(
        self,
        params: ParameterVector,
    ) -> Float[Array, " n_basis"]:
        """Compute power-law PSD weights for the Fourier basis.

        Returns one weight per basis column (sin and cos of each
        frequency get the same weight).

        The power spectral density follows the convention::

            P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)

        Each weight is ``P(f) · Δf``, repeated twice for the sin/cos
        pair at that frequency.

        Parameters
        ----------
        params : ParameterVector
            Must contain values for ``TNREDAMP`` (log10 amplitude)
            and ``TNREDGAM`` (spectral index).

        Returns
        -------
        weights : (2 * n_freqs,)
            PSD weights for each basis column.
        """
        log10_A = params.param_value(self.tnredamp_name)
        gamma = params.param_value(self.tnredgam_name)
        A = 10.0 ** log10_A

        psd = (
            A ** 2
            / (12.0 * jnp.pi ** 2)
            * FYR ** (gamma - 3.0)
            * self.freqs ** (-gamma)
        )
        # weight = PSD(f) * Δf, repeated for sin and cos
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
        """Return the Woodbury ``(Ndiag, U, Phidiag)`` triple for red noise.

        Red noise is purely low-rank: ``Ndiag = 0``.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data (used for array sizing).
        params : ParameterVector
            Current parameter values for amplitude and spectral index.

        Returns
        -------
        Ndiag : (n_toas,)
            Zero diagonal (red noise has no white component).
        U : (n_toas, 2 * n_freqs)
            Fourier design matrix.
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
        """Draw a random red noise realization.

        Draws standard-normal Fourier amplitudes and projects them
        through the basis matrix scaled by sqrt(weights).

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data (used for basis matrix dimensions).
        params : ParameterVector
            Current parameter values for amplitude and spectral index.
        key : jax.Array
            PRNG key for random sampling.

        Returns
        -------
        noise : (n_toas,)
            Red noise realization in seconds.
        """
        weights = self.psd_weights(params)
        n_basis = self.fourier_basis.shape[1]
        a = jax.random.normal(key, shape=(n_basis,))
        return self._fourier_basis_jax @ (jnp.sqrt(weights) * a)
