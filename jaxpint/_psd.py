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
    "turnover_psd",
    "turnover_knee_psd",
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


def turnover_psd(
    f: Float[Array, " n_freq"],
    log10_A: ArrayLike,
    gamma: ArrayLike,
    lf0: ArrayLike,
    kappa: ArrayLike = 10.0 / 3.0,
    beta: ArrayLike = 0.5,
) -> Float[Array, " n_freq"]:
    r"""Power law with a low-frequency turnover below ``f_0 = 10^{lf0}``.

    The environmentally-driven GWB spectrum of Sampson, Cornish & McWilliams
    (2015) [psd_scm15]_ in enterprise's ``gp_priors.turnover`` convention: the
    characteristic strain is a power law suppressed below the transition
    frequency,

    .. math::
        h_c(f) = A \left(\frac{f}{f_{\rm yr}}\right)^{(3-\gamma)/2}
                 \Big[1 + \big(f_0/f\big)^{\kappa}\Big]^{-\beta},

    and ``S(f) = h_c^2(f) / (12 \pi^2 f^3)``, i.e. the plain power law times
    the suppression factor ``(1 + (f_0/f)^\kappa)^{-2\beta}``.  Well above
    ``f_0`` this reduces to :func:`powerlaw_psd`; below it the spectrum bends
    down with asymptotic extra slope ``2\beta\kappa``.

    Parameters
    ----------
    f : (n_freq,) array
        Frequencies in Hz.
    log10_A, gamma : scalar
        Power-law amplitude and spectral index (see :func:`powerlaw_psd`).
    lf0 : scalar
        Log-10 of the turnover frequency in Hz.
    kappa : scalar
        Turnover sharpness (10/3 for stellar three-body scattering).
    beta : scalar
        Strain suppression exponent (production analyses fix 0.5).

    Returns
    -------
    psd : (n_freq,) array
        Power spectral density in units of s^3.

    References
    ----------
    .. [psd_scm15] Sampson, Cornish & McWilliams (2015), PRD 91, 084055.
    """
    f0 = 10.0**lf0
    suppression = (1.0 + (f0 / f) ** kappa) ** (2.0 * beta)
    return powerlaw_psd(f, log10_A, gamma) / suppression


def turnover_knee_psd(
    f: Float[Array, " n_freq"],
    log10_A: ArrayLike,
    gamma: ArrayLike,
    lfb: ArrayLike,
    lfk: ArrayLike,
    kappa: ArrayLike = 10.0 / 3.0,
    delta: ArrayLike = -1.0,
) -> Float[Array, " n_freq"]:
    r"""Turnover spectrum with an additional high-frequency knee.

    Enterprise's ``gp_priors.turnover_knee``: a low-frequency environmental
    bend at ``f_b = 10^{lfb}`` (as in :func:`turnover_psd` with
    ``beta = 1/2``) plus a knee at ``f_k = 10^{lfk}`` where the finite number
    of contributing binaries steepens the spectrum,

    .. math::
        h_c(f) = A \left(\frac{f}{f_{\rm yr}}\right)^{(3-\gamma)/2}
                 \big(1 + f/f_k\big)^{\delta}
                 \Big[1 + \big(f_b/f\big)^{\kappa}\Big]^{-1/2},

    with ``S(f) = h_c^2(f) / (12 \pi^2 f^3)``.  With the bend far below the
    band and ``delta = 0`` this reduces to :func:`powerlaw_psd`.

    Parameters
    ----------
    f : (n_freq,) array
        Frequencies in Hz.
    log10_A, gamma : scalar
        Power-law amplitude and spectral index (see :func:`powerlaw_psd`).
    lfb : scalar
        Log-10 of the low-frequency (environmental) bend frequency in Hz.
    lfk : scalar
        Log-10 of the knee frequency in Hz (population finiteness).
    kappa : scalar
        Bend sharpness (10/3 for stellar three-body scattering).
    delta : scalar
        Strain slope change above the knee (negative steepens the PSD).

    Returns
    -------
    psd : (n_freq,) array
        Power spectral density in units of s^3.
    """
    fb = 10.0**lfb
    fk = 10.0**lfk
    knee = (1.0 + f / fk) ** (2.0 * delta)
    bend = 1.0 + (fb / f) ** kappa
    return powerlaw_psd(f, log10_A, gamma) * knee / bend


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
