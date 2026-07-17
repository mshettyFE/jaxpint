"""Low-level spectral-density kernels for Fourier-basis GP processes.

Pure functions mapping ``(frequency, hyperparameters) -> power spectral
density``.
Keeping the formulae here means single source of truth for PSD convention.

Kernels return the PSD **per frequency** (length ``n_freq``).  Two things are
deliberately *not* folded in, because they are not uniform across models:

- **The frequency bin width** ``df``.  A power law multiplies the PSD by it; a
  free spectrum absorbs it into ``rho`` and ignores it.  Callers apply ``df``
  themselves.
- **The (sin, cos) column pairing.**  The Fourier basis is laid out interleaved
  ``[sin(f0), cos(f0), sin(f1), cos(f1), ...]``, so each frequency's weight
  applies to two columns.  Callers finish with :func:`expand_sin_cos`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float

from jaxpint.constants import FYR

__all__ = [
    "powerlaw_psd",
    "broken_powerlaw_psd",
    "free_spectrum_psd",
    "expand_sin_cos",
]


def powerlaw_psd(
    f: Float[Array, " n_freq"], log10_A: Float, gamma: Float
) -> Float[Array, " n_freq"]:
    r"""Power-law power spectral density (NANOGrav convention).

    Follows the parameterisation of Arzoumanian et al. (2016) [psd_a16]_ Eq. 1,
    derived from the characteristic-strain relation of Phinney (2001) [psd_p01]_:
    ``S(f) = h_c^2(f) / (12 pi^2 f^3)``.

    .. math::
        S(f) = \frac{A^2}{12\pi^2}
               \left(\frac{f}{f_{\rm yr}}\right)^{-\gamma}
               f_{\rm yr}^{-3}

    Parameters
    ----------
    f : (n_freq,) array
        Frequencies in Hz.
    log10_A : scalar
        Log-10 of the dimensionless amplitude.
    gamma : scalar
        Spectral index (positive for red noise).

    Returns
    -------
    psd : (n_freq,) array
        Power spectral density in units of s^3.

    References
    ----------
    .. [psd_a16] Arzoumanian et al. (2016), ApJ 821, 13.
    .. [psd_p01] Phinney (2001), astro-ph/0108028.
    """
    return (
        (10.0 ** (2.0 * log10_A))
        / (12.0 * jnp.pi**2)
        * FYR ** (gamma - 3.0)
        * f ** (-gamma)
    )


def broken_powerlaw_psd(
    f: Float[Array, " n_freq"],
    log10_A: ArrayLike,
    gamma: ArrayLike,
    log10_fb: ArrayLike,
    kappa: ArrayLike = 0.1,
) -> Float[Array, " n_freq"]:
    r"""Power law with a smooth spectral bend at ``f_b = 10^{log10_fb}``.

    ``S(f) = S_pl(f) * (1 + (f/f_b)^{1/\kappa})^{\kappa\gamma}`` — below the
    bend the slope is ``-gamma``; above it the spectrum flattens
    (Arzoumanian et al. 2020 convention, delta = 0 above the bend).  The fixed
    smoothness ``kappa`` defaults to discovery's 0.1.

    Parameters
    ----------
    f : (n_freq,) array
        Frequencies in Hz.
    log10_A, gamma : scalar
        Power-law amplitude and spectral index (see :func:`powerlaw_psd`).
    log10_fb : scalar
        Log-10 of the bend frequency in Hz.
    kappa : scalar
        Bend smoothness (dimensionless).

    Returns
    -------
    psd : (n_freq,) array
        Power spectral density in units of s^3.
    """
    fb = 10.0**log10_fb
    bend = (1.0 + (f / fb) ** (1.0 / kappa)) ** (kappa * gamma)
    return powerlaw_psd(f, log10_A, gamma) * bend


def free_spectrum_psd(log10_rho: Float[Array, " n_freq"]) -> Float[Array, " n_freq"]:
    r"""Per-frequency free-spectrum PSD, ``10^{2 log10_rho}``.

    ``rho_k`` is the per-bin RMS amplitude in seconds (discovery's
    ``freespectrum``); the frequency bin width ``df`` is absorbed into ``rho``,
    so — unlike :func:`powerlaw_psd` — there is no ``f`` or ``df`` dependence.

    Parameters
    ----------
    log10_rho : (n_freq,) array
        Log-10 per-bin RMS amplitudes.

    Returns
    -------
    psd : (n_freq,) array
        Per-bin variance weights.
    """
    return 10.0 ** (2.0 * log10_rho)


def expand_sin_cos(psd_per_freq: Float[Array, " n_freq"]) -> Float[Array, " n_basis"]:
    """Repeat each per-frequency weight for its (sin, cos) basis column pair.

    The Fourier basis interleaves columns as
    ``[sin(f0), cos(f0), sin(f1), cos(f1), ...]`` (see
    :func:`jaxpint.pta.signals.gwb.fourier_basis` /
    :func:`jaxpint.utils.build_fourier_basis`), so a length-``n_freq`` PSD maps
    to length-``2*n_freq`` weights by assigning each frequency's value to both
    of its columns.  Getting this ordering wrong silently misaligns every
    weight — keep it defined in exactly one place.
    """
    return jnp.repeat(psd_per_freq, 2)
