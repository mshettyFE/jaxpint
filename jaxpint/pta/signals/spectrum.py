"""Pluggable PSD models for Fourier-basis GP processes.

A :class:`SpectralModel` maps named hyperparameters to the diagonal PSD
weights ``Φ`` of a Fourier-basis Gaussian process — the ``Phidiag`` of the
Woodbury triple ``C = diag(N) + U diag(Φ) Uᵀ``.  Every model here keeps
``Φ`` **diagonal**: swapping the spectrum (power law → broken power law →
free spectrum) only changes how the diagonal is filled and how many
hyperparameters exist; the Woodbury solve path is untouched.  (What *would*
break the diagonal fast path is a dense inter-frequency covariance, e.g.
discovery's FFT/time-domain kernels — none of these models need that yet.)

Conventions match ``discovery.signals``:

- ``powerlaw``: ``S(f) = A²/(12π²) · f_yr^(γ-3) · f^(-γ)``, weight
  ``S(f)·Δf`` per frequency, repeated for the (sin, cos) pair.
- ``brokenpowerlaw``: the power law times a smooth bend
  ``(1 + (f/f_b)^(1/κ))^(κγ)`` with ``κ = 0.1`` (Arzoumanian et al. 2020
  convention, δ = 0 above the bend).
- ``freespectrum``: one weight ``10^(2·log10_ρ_k)`` per frequency bin;
  ``ρ_k`` is the per-bin RMS in seconds and absorbs ``Δf``.

Injectors resolve parameter names through a ``value_of`` callable so the
same model works for prefixed :class:`~jaxpint.types.GlobalParams` (common
processes) and per-pulsar :class:`~jaxpint.types.ParameterVector` use.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence, Union

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

__all__ = [
    "SpectralModel",
    "PowerLawSpectrum",
    "BrokenPowerLawSpectrum",
    "FreeSpectrum",
]


# The parameter-resolution environment a caller passes to
# :meth:`SpectralModel.psd_weights`: maps a parameter *suffix* (a key of
# ``param_defaults()``, e.g. ``"gamma"`` or ``"log10_rho_3"``) to its current
# value. See test_spectral_models.py for example
_ValueOf = Callable[[str], Array]


class SpectralModel(ABC):
    """PSD model on a Fourier basis: parameter names plus diagonal weights."""

    @abstractmethod
    def param_defaults(self) -> dict[str, float]:
        """Hyperparameter name suffixes and their default initial values.

        Injectors prepend their prefix (e.g. ``gwb_``) when registering.
        """
        ...

    @abstractmethod
    def psd_weights(
        self,
        freqs: Float[Array, " n_freq"],
        df: ArrayLike,
        value_of: _ValueOf,
    ) -> Float[Array, " n_basis"]:
        """Diagonal PSD weights, one per basis column (sin/cos share a value).

        Parameters
        ----------
        freqs : (n_freq,) array
            Fourier frequencies in Hz.
        df : scalar or (n_freq,) array
            Frequency bin widths.
        value_of : callable
            Maps a parameter suffix from :meth:`param_defaults` to its
            (possibly traced) value.
        """
        ...


class PowerLawSpectrum(SpectralModel):
    """Standard power-law PSD (NANOGrav convention); params ``log10_A, gamma``."""

    def __init__(self, log10_A: float = -15.0, gamma: float = 4.33):
        self.defaults = {"log10_A": log10_A, "gamma": gamma}

    def param_defaults(self) -> dict[str, float]:
        return dict(self.defaults)

    def psd_weights(self, freqs, df, value_of) -> Float[Array, " n_basis"]:
        from jaxpint.pta.signals.gwb import powerlaw_psd

        psd = powerlaw_psd(freqs, value_of("log10_A"), value_of("gamma"))
        return jnp.repeat(psd * df, 2)


class BrokenPowerLawSpectrum(SpectralModel):
    """Power law with a smooth spectral bend at ``f_b`` (δ = 0 above it).

    ``S(f) = S_pl(f) · (1 + (f/f_b)^(1/κ))^(κγ)`` with fixed smoothness
    ``κ`` (discovery's ``brokenpowerlaw``): below the bend the slope is
    ``-γ``, above it the spectrum flattens.  Params ``log10_A, gamma,
    log10_fb``.
    """

    def __init__(
        self,
        log10_A: float = -15.0,
        gamma: float = 4.33,
        log10_fb: float = -8.0,
        kappa: float = 0.1,
    ):
        self.kappa = kappa
        self.defaults = {"log10_A": log10_A, "gamma": gamma, "log10_fb": log10_fb}

    def param_defaults(self) -> dict[str, float]:
        return dict(self.defaults)

    def psd_weights(self, freqs, df, value_of) -> Float[Array, " n_basis"]:
        from jaxpint.pta.signals.gwb import powerlaw_psd

        gamma = value_of("gamma")
        psd = powerlaw_psd(freqs, value_of("log10_A"), gamma)
        fb = 10.0 ** value_of("log10_fb")
        bend = (1.0 + (freqs / fb) ** (1.0 / self.kappa)) ** (self.kappa * gamma)
        return jnp.repeat(psd * bend * df, 2)


class FreeSpectrum(SpectralModel):
    """Per-frequency free spectrum; params ``log10_rho_0 … log10_rho_{n-1}``.

    Each bin's weight is ``10^(2·log10_ρ_k)`` — ``ρ_k`` is that bin's RMS
    amplitude in seconds (discovery's ``freespectrum``; ``Δf`` is absorbed
    into ρ, so ``df`` is ignored).  ``Φ`` stays diagonal: a free spectrum
    is more *hyperparameters*, not more covariance structure, and runs
    through the identical Woodbury solve as a power law.

    Parameters
    ----------
    n_components : int
        Number of frequency bins (must match the consuming injector's
        ``n_components``).
    log10_rho : float or sequence of float
        Initial value(s); a scalar is broadcast to all bins.
    """

    def __init__(
        self,
        n_components: int,
        log10_rho: Union[float, Sequence[float]] = -8.0,
    ):
        self.n_components = n_components
        if isinstance(log10_rho, (int, float)):
            rho0 = [float(log10_rho)] * n_components
        else:
            rho0 = [float(r) for r in log10_rho]
            if len(rho0) != n_components:
                raise ValueError(
                    f"log10_rho has {len(rho0)} entries, expected {n_components}"
                )
        self._names = tuple(f"log10_rho_{k}" for k in range(n_components))
        self.defaults = dict(zip(self._names, rho0))

    @property
    def param_names(self) -> tuple[str, ...]:
        """The per-bin parameter suffixes, in frequency order."""
        return self._names

    def param_defaults(self) -> dict[str, float]:
        return dict(self.defaults)

    def psd_weights(self, freqs, df, value_of) -> Float[Array, " n_basis"]:
        log10_rho = jnp.stack([value_of(n) for n in self._names])
        return jnp.repeat(10.0 ** (2.0 * log10_rho), 2)


def validate_spectrum_components(spectrum: SpectralModel, n_components: int) -> None:
    """Raise if a bin-count-carrying spectrum disagrees with the injector."""
    n_spec: Optional[int] = getattr(spectrum, "n_components", None)
    if n_spec is not None and n_spec != n_components:
        raise ValueError(
            f"spectrum has n_components={n_spec} but the injector was built "
            f"with n_components={n_components}"
        )
