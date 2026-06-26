"""Shared base for power-law-PSD Fourier noise components.

``PLRedNoise``, ``PLDMNoise``, ``PLChromNoise`` and ``PLSWNoise`` all model a
noise process as a low-rank ``C = F · diag(w) · Fᵀ`` with power-law PSD weights
``w``. They differ only in (a) which amplitude / spectral-index parameters they
read and (b) whether the Fourier basis ``F`` is fixed or scaled per TOA at
evaluation time. This base captures the shared machinery; subclasses supply the
two parameter-name accessors and, when the basis is scaled at runtime, override
:meth:`_basis`.

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

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.constants import FYR
from jaxpint.types import TOAData, ParameterVector


class _PowerLawFourierNoise(NoiseComponent):
    """Base for power-law-PSD Fourier noise (red / DM / chromatic / solar-wind).

    Subclasses must declare their amplitude/spectral-index name fields and
    ``PARAMS``, expose the names via :attr:`_amp_name` / :attr:`_gam_name`, and
    -- when the basis is scaled per TOA -- override :meth:`_basis`.
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

    @functools.cached_property
    def _fourier_basis_jax(self) -> Float[Array, "n_toas n_basis"]:
        """Lazy device-converted view of ``fourier_basis`` (cached per instance)."""
        return jnp.asarray(self.fourier_basis)

    # -- subclass hooks --------------------------------------------------

    @property
    def _amp_name(self) -> str:
        """Parameter name of the log10 amplitude."""
        raise NotImplementedError

    @property
    def _gam_name(self) -> str:
        """Parameter name of the spectral index."""
        raise NotImplementedError

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

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_basis"]:
        """Power-law PSD weights, one per basis column (sin/cos share a value).

        ``P(f) = (A² / 12π²) · f_yr^(γ-3) · f^(-γ)``; each weight is
        ``P(f) · Δf``, repeated twice for the sin/cos pair at that frequency.
        """
        log10_A = params.param_value(self._amp_name)
        gamma = params.param_value(self._gam_name)
        A = 10.0**log10_A
        psd = A**2 / (12.0 * jnp.pi**2) * FYR ** (gamma - 3.0) * self.freqs ** (-gamma)
        return jnp.repeat(psd * self.freq_bin_widths, 2)

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
