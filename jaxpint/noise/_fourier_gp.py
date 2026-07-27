"""Shared base for low-rank Fourier-GP noise components.

``PLRedNoise``, ``PLDMNoise``, ``PLChromNoise``, ``PLSWNoise`` and
``FreeSpectrumNoise`` all model a noise process as a low-rank
``C = F · diag(w) · Fᵀ`` on a Fourier basis ``F``. They differ only in how the
per-column PSD weights ``w`` are parameterized and whether ``F`` is fixed or
scaled per TOA at evaluation time.

This module adds the Fourier-specific layer: the frequency metadata
(``freqs`` / ``freq_bin_widths``) and the delegation of ``psd_weights`` to a
:class:`~jaxpint.spectra.SpectralModel`. :class:`_PowerLawFourierNoise` is
the further specialization for the four power-law components: it implements
``psd_weights`` in terms of an amplitude and spectral index, exposed via
:attr:`~_PowerLawFourierNoise._amp_name` /
:attr:`~_PowerLawFourierNoise._gam_name`. ``FreeSpectrumNoise`` — whose
weights are per-bin, not power-law — subclasses :class:`_FourierGPNoise`
directly and supplies its own ``psd_weights``.
"""

from __future__ import annotations


import numpy as np
from jaxtyping import Array, Float

from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.spectra import PowerLawSpectrum, SpectralModel
from jaxpint.types import ParameterVector


class _FourierGPNoise(_BasisGPNoise):
    """Fourier specialization of :class:`~jaxpint.noise._basis_gp._BasisGPNoise`.

    Subclasses must implement :meth:`psd_weights` (the diagonal ``w``, one
    entry per basis column) and -- when the basis is scaled per TOA --
    override :meth:`_basis`.
    """

    # Constructors may pass a jax array; __post_init__ coerces to host
    # numpy (the source of truth) -- the union reflects both stages.
    fourier_basis: Float[Array, "n_toas n_basis"] | Float[np.ndarray, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]

    def __post_init__(self):
        # Host numpy is the source of truth; the device view is built lazily
        # by _BasisGPNoise._columns_jax. See that module's docstring.
        self._coerce_host_field("fourier_basis")

    def _host_columns(
        self,
    ) -> Float[np.ndarray, "n_toas n_basis"] | Float[Array, "n_toas n_basis"]:
        # Pass-through, never np.asarray: inside a jit trace of a
        # reconstructed instance this field is a tracer (see base docstring).
        return self.fourier_basis

    @property
    def _fourier_basis_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Device view of ``fourier_basis`` (alias of the generic lazy cache)."""
        return self._columns_jax

    # -- Fourier-specific prior plumbing -----------------------------------

    def _weights_from(
        self,
        spectrum: SpectralModel,
        name_map: dict[str, str],
        params: ParameterVector,
    ) -> Float[Array, " n_basis"]:
        """Delegate ``psd_weights`` to a :class:`~jaxpint.spectra.SpectralModel`.

        The PSD-shape arithmetic lives once in :mod:`jaxpint.spectra`; a noise
        component supplies the model plus ``name_map`` (spectrum hyperparameter
        *suffix* -> this component's :class:`ParameterVector` name) and this
        builds the ``value_of`` lookup the model expects, passing the component's
        own ``freqs`` / ``freq_bin_widths``.
        """
        return spectrum.psd_weights(
            self.freqs,
            self.freq_bin_widths,
            lambda suffix: params.param_value(name_map[suffix]),
        )


class _PowerLawFourierNoise(_FourierGPNoise):
    """Power-law specialization of :class:`_FourierGPNoise`.

    The base for the four power-law components (red / DM / chromatic /
    solar-wind). Subclasses declare their amplitude/spectral-index name fields
    and ``PARAMS``, expose the names via :attr:`_amp_name` / :attr:`_gam_name`,
    and -- when the basis is scaled per TOA -- override :meth:`_basis`.
    """

    # -- subclass hooks --------------------------------------------------

    @property
    def _amp_name(self) -> str:
        """Parameter name of the log10 amplitude."""
        raise NotImplementedError

    @property
    def _gam_name(self) -> str:
        """Parameter name of the spectral index."""
        raise NotImplementedError

    # -- power-law weights -----------------------------------------------

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_basis"]:
        """Power-law PSD weights, delegated to :class:`~jaxpint.spectra.PowerLawSpectrum`.

        ``P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)``; each weight is ``P(f) · Δf``,
        repeated for the sin/cos pair. The formula lives once in ``jaxpint.spectra``.
        """
        return self._weights_from(
            PowerLawSpectrum(),
            {"log10_A": self._amp_name, "gamma": self._gam_name},
            params,
        )
