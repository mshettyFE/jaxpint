"""Shared base for low-rank Fourier-GP noise components.

``PLRedNoise``, ``PLDMNoise``, ``PLChromNoise``, ``PLSWNoise`` and
``FreeSpectrumNoise`` all model a noise process as a low-rank
``C = F · diag(w) · Fᵀ`` on a Fourier basis ``F``. They differ only in how the
per-column PSD weights ``w`` are parameterized and whether ``F`` is fixed or
scaled per TOA at evaluation time.

:class:`_FourierGPNoise` captures the shared machinery (the Woodbury covariance,
generation, and lazy basis handling) and declares a single hook,
:meth:`~_FourierGPNoise.psd_weights`, that each subclass must implement.
:class:`_PowerLawFourierNoise` is the specialization for the four power-law
components: it implements ``psd_weights`` in terms of an amplitude and spectral
index, exposed via :attr:`~_PowerLawFourierNoise._amp_name` /
:attr:`~_PowerLawFourierNoise._gam_name`. ``FreeSpectrumNoise`` — whose weights
are per-bin, not power-law — subclasses :class:`_FourierGPNoise` directly and
supplies its own ``psd_weights``.

Static vs dynamic basis
-----------------------
A component is "static" iff its basis is fixed at build time (red/DM — the DM
``(1400/f)²`` factor is pre-baked by the bridge). Such components override
:meth:`static_basis` to return their basis, so :class:`~jaxpint.noise.NoiseModel`
can pre-stack it once. Components whose basis depends on fitted parameters
(chromatic ``(fref/f)^α``, solar-wind geometry) leave ``static_basis`` at the
:class:`~jaxpint.components.NoiseComponent` default (``None`` → dynamic) and
override :meth:`_basis` instead. Defaulting to dynamic fails safe: a dynamic
basis is always correct, just not pre-stacked.
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.spectra import PowerLawSpectrum, SpectralModel
from jaxpint.types import TOAData, ParameterVector


class _FourierGPNoise(NoiseComponent):
    """Base for low-rank Fourier-GP noise (``C = F · diag(w) · Fᵀ``).

    Subclasses must implement :meth:`psd_weights` (the diagonal ``w``, one entry
    per basis column) and -- when the basis is scaled per TOA -- override
    :meth:`_basis`.
    """

    fourier_basis: Float[Array, "n_toas n_basis"]
    freqs: Float[Array, " n_freqs"]
    freq_bin_widths: Float[Array, " n_freqs"]

    def __post_init__(self):
        # Keep the basis as host numpy (source of truth); the device view is
        # built lazily and cached by ``_fourier_basis_jax``. This keeps the
        # (potentially large) basis off-device until first use -- at full-PTA
        # scale only one pulsar's basis sits on the device at a time.
        if not isinstance(self.fourier_basis, np.ndarray):
            object.__setattr__(self, "fourier_basis", np.asarray(self.fourier_basis))

    @property
    def _fourier_basis_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Lazy device-converted view of ``fourier_basis`` (cached per instance).

        Cached manually instead of via ``functools.cached_property``: inside
        a jit trace ``jnp.asarray`` returns a tracer, and caching a tracer on
        the (persistent) host instance leaks it into later traces. Only
        concrete arrays are cached; traced conversions are recomputed per
        trace (where they become jaxpr constants anyway).
        """
        cached = self.__dict__.get("_fourier_basis_jax_cache")
        if cached is None:
            cached = jnp.asarray(self.fourier_basis)
            if not isinstance(cached, jax.core.Tracer):
                self.__dict__["_fourier_basis_jax_cache"] = cached  # pyright: ignore[reportIndexIssue]
        return cached

    # -- subclass hooks --------------------------------------------------

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_basis"]:
        """PSD weights, one per basis column (the diagonal of ``diag(w)``).

        The single mandatory hook: each subclass maps its hyperparameters to the
        per-column weights (sin/cos pairs share a value). Power-law components
        get this for free from :class:`_PowerLawFourierNoise`.
        """
        raise NotImplementedError

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

    def _basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """The Fourier basis used in :meth:`covariance` / :meth:`generate`.

        Defaults to the fixed (pre-baked) basis. Chromatic / solar-wind noise
        override this to scale it per TOA by a parameter-dependent factor.
        """
        return self._fourier_basis_jax

    # -- shared machinery ------------------------------------------------

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas n_basis"],
        Float[Array, " n_basis"],
    ]:
        """Woodbury ``(Ndiag, U, Phidiag)`` triple; purely low-rank (``Ndiag = 0``)."""
        Ndiag = jnp.zeros(toa_data.n_toas)
        return Ndiag, self._basis(toa_data, params), self.psd_weights(params)

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a realization: ``F · (sqrt(w) ⊙ z)`` with ``z ~ N(0, I)``."""
        weights = self.psd_weights(params)
        basis = self._basis(toa_data, params)
        a = jax.random.normal(key, shape=(basis.shape[1],))
        return basis @ (jnp.sqrt(weights) * a)


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
