"""Frequentist F-statistic detection significance via empirical backgrounds.

Mirrors the NANOGrav CGW approach: the F-statistic is the detection *statistic*, but
its significance comes from an *empirical* null distribution -- built by destroying the
coherence a real CGW induces -- rather than the theoretical ``chi^2`` tail (which
ignores both the sky-maximization look-elsewhere effect and any noise mis-modeling).
Two nulls, as NANOGrav reports:

* **phase shifts** -- give each pulsar's signal a random phase, killing the
  *inter-pulsar phase coherence* while preserving every per-pulsar spectrum;
* **sky scrambles** -- randomize the pulsar sky positions, killing the *geometric*
  (antenna) coherence.

The observed sky-maximized ``2F`` is then scored against each background: the p-value
is the fraction of null realizations that exceed it.

Efficiency.  For the Earth term the sin/cos quadratures ``sin/cos(2 pi f t)`` are
**sky-independent** -- only the antenna patterns ``F+, Fx`` carry the sky (see
:func:`jaxpint.pta.signals.cw.cw_delay_from_array`).  So each pulsar's 2x2 matched
filter ``(S, C)`` and Gram ``G`` are extracted **once**; the whole sky map and every
background realization are then cheap analytic antenna-folding:

    b_a = (F+_a, Fx_a) (x) (S_a, C_a),   M_a = (F+_a, Fx_a)(F+_a, Fx_a)^T (x) G_a,
    2F(sky) = b^T M^+ b,   b = sum_a b_a,  M = sum_a M_a.

A phase shift rotates ``(S_a, C_a)`` (M unchanged); a sky scramble redraws ``F+, Fx``.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.stats import chi2

from jaxpint.pta.incoherent_ul import extract_pulsar_blocks
from jaxpint.pta.signals.cw import _TREF, _fplus_fcross_costheta

__all__ = [
    "quadrature_blocks",
    "fstat_skymap",
    "fstat_p",
    "fstat_p_pvalue",
    "phase_shift_background",
    "sky_scramble_background",
    "pvalue",
]


def quadrature_blocks(
    g: Callable,
    reduced_params,
    toa_data,
    log10_fgw: float,
) -> tuple[Float[Array, " 2"], Float[Array, "2 2"]]:
    """Per-pulsar Earth-term sin/cos matched filter ``(S, C)`` and 2x2 Gram ``G``.

    ``S = (d | sin), C = (d | cos)`` (data projection) and ``G_ab = (q_a | q_b)`` for
    the two quadratures ``q = {sin, cos}(2 pi f (t - t_ref))``, in the pulsar's
    timing-marginalized GLS metric (via
    :func:`~jaxpint.pta.incoherent_ul.extract_pulsar_blocks`).  Both are
    **sky-independent** (Earth term), so extract once per pulsar per frequency and
    reuse for the whole sky map and every background realization.

    Parameters
    ----------
    g : callable
        Single-pulsar timing-marginalized log-likelihood
        ``g(reduced_params, external_delay=...)``.  For a detection run this is the
        *injected-data* likelihood (signal baked in), so ``(S, C)`` carries the signal.
    reduced_params
        The reduced-parameter skeleton ``g`` expects.
    toa_data : TOAData
        Pulse time-of-arrival data (uses TDB times).
    log10_fgw : float
        ``log10`` GW frequency (Hz).

    Returns
    -------
    sc : (2,) array
        The matched filter ``(S, C)``.
    gram : (2, 2) array
        The quadrature Gram ``G``.
    """
    f0 = 10.0**log10_fgw
    phase = 2.0 * jnp.pi * f0 * (toa_data.tdb_seconds - _TREF)
    basis = jnp.stack([jnp.sin(phase), jnp.cos(phase)])  # (2, n_toas)
    sc, gram = extract_pulsar_blocks(g, reduced_params, basis)
    return sc, gram


def _antenna_grid(
    positions: Float[Array, "npsr 3"],
    cos_gwtheta: Float[Array, " npix"],
    gwphi: Float[Array, " npix"],
) -> Float[Array, "npix npsr 2"]:
    """Antenna patterns ``(F+, Fx)`` for every (pixel, pulsar)."""
    sin_th = jnp.sqrt(jnp.clip(1.0 - cos_gwtheta**2, 0.0, None))
    return jax.vmap(
        lambda ct, st, gp: jax.vmap(
            lambda p: jnp.stack(_fplus_fcross_costheta(p, ct, st, gp))
        )(positions)
    )(cos_gwtheta, sin_th, gwphi)


def _two_f_map(
    F_pix: Float[Array, "npix npsr 2"],
    sc_all: Float[Array, "npsr 2"],
    gram_all: Float[Array, "npsr 2 2"],
) -> Float[Array, " npix"]:
    """Network ``2F = b^T M^+ b`` per pixel, folding ``(S,C)/G`` with antenna ``F_pix``."""
    b = jax.vmap(lambda F: jnp.einsum("ai,aj->ij", F, sc_all).reshape(4))(F_pix)
    M = jax.vmap(lambda F: jnp.einsum("ai,ak,ajl->ijkl", F, F, gram_all).reshape(4, 4))(
        F_pix
    )
    # Moore-Penrose (not solve): the network Gram can be rank-deficient at a given
    # pixel (few pulsars, or a degenerate sky-scramble geometry).
    return jax.vmap(lambda bb, MM: bb @ jnp.linalg.pinv(MM) @ bb)(b, M)


def fstat_skymap(
    sc_all: Float[Array, "npsr 2"],
    gram_all: Float[Array, "npsr 2 2"],
    positions: Float[Array, "npsr 3"],
    cos_gwtheta: Float[Array, " npix"],
    gwphi: Float[Array, " npix"],
) -> Float[Array, " npix"]:
    """Earth-term ``2F`` over a sky grid (the detection statistic is its max).

    Parameters
    ----------
    sc_all : (npsr, 2) array
        Per-pulsar matched filters ``(S, C)`` (:func:`quadrature_blocks`).
    gram_all : (npsr, 2, 2) array
        Per-pulsar quadrature Grams ``G``.
    positions : (npsr, 3) array
        Pulsar unit vectors.
    cos_gwtheta, gwphi : (npix,) arrays
        Sky grid (cos-colatitude, right ascension).

    Returns
    -------
    (npix,) array
        ``2F`` at each pixel.
    """
    return _two_f_map(_antenna_grid(positions, cos_gwtheta, gwphi), sc_all, gram_all)


def fstat_p(
    sc_all: Float[Array, "npsr 2"],
    gram_all: Float[Array, "npsr 2 2"],
) -> Float[Array, ""]:
    """Incoherent ``F_p`` detection statistic ``2F_p = sum_a (S,C)_a^T G_a^-1 (S,C)_a``.

    Each pulsar's single-frequency power, its amplitude and phase profiled out (2 dof
    per pulsar), summed over pulsars (Ellis, Siemens & Creighton 2012, arXiv:1204.4218).
    Distance-**robust** (assumes no inter-pulsar coherence) and **sky-independent** (the
    antenna is absorbed into each pulsar's free amplitude) -- so, unlike the coherent
    :func:`fstat_skymap`, it is (near-)invariant under the phase-shift / sky-scramble
    nulls and those backgrounds do not calibrate it.  Its significance comes from the
    analytic null instead: ``2F_p ~ chi^2(2 * n_psr)`` (see :func:`fstat_p_pvalue` and
    :func:`jaxpint.sensitivity.chi2_threshold` with ``dof = 2 * n_psr``).

    Parameters
    ----------
    sc_all : (npsr, 2) array
        Per-pulsar matched filters ``(S, C)`` (:func:`quadrature_blocks`).
    gram_all : (npsr, 2, 2) array
        Per-pulsar quadrature Grams ``G``.

    Returns
    -------
    scalar
        ``2F_p``.
    """
    return jax.vmap(lambda sc, G: sc @ jnp.linalg.solve(G, sc))(sc_all, gram_all).sum()


def fstat_p_pvalue(stat: float, n_psr: int) -> float:
    """Analytic p-value of an observed ``2F_p`` under the null ``chi^2(2 * n_psr)``.

    The upper-tail (survival-function) probability that noise alone produces a value at
    least as large as ``stat``.  Threshold at a false-alarm probability via
    :func:`jaxpint.sensitivity.chi2_threshold` with ``dof = 2 * n_psr``.

    Parameters
    ----------
    stat : float
        Observed ``2F_p`` (:func:`fstat_p`).
    n_psr : int
        Number of pulsars (the null has ``2 * n_psr`` degrees of freedom).

    Returns
    -------
    float
        The p-value ``P[chi^2(2 n_psr) >= stat]``.
    """
    return float(chi2.sf(stat, 2 * n_psr))


def phase_shift_background(
    sc_all: Float[Array, "npsr 2"],
    gram_all: Float[Array, "npsr 2 2"],
    positions: Float[Array, "npsr 3"],
    cos_gwtheta: Float[Array, " npix"],
    gwphi: Float[Array, " npix"],
    n_real: int,
    key: Array,
) -> Float[Array, " n_real"]:
    """Sky-maximized ``2F`` under random per-pulsar phase shifts (the null).

    Each realization rotates every pulsar's ``(S, C)`` by an independent random angle
    -- destroying the inter-pulsar phase coherence -- and recomputes the sky-max
    ``2F``.  The antenna grid and Gram ``M(sky)`` are fixed (templates unchanged), so
    they are precomputed once.

    Parameters
    ----------
    sc_all, gram_all, positions, cos_gwtheta, gwphi
        As in :func:`fstat_skymap`.
    n_real : int
        Number of null realizations.
    key : PRNGKey
        JAX random key.

    Returns
    -------
    (n_real,) array
        Sky-maximized ``2F`` for each phase-shifted null realization.
    """
    F_pix = _antenna_grid(positions, cos_gwtheta, gwphi)
    S, C = sc_all[:, 0], sc_all[:, 1]

    def one(k):
        phi = jax.random.uniform(k, (sc_all.shape[0],), minval=0.0, maxval=2.0 * jnp.pi)
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        sc_rot = jnp.stack([cphi * S - sphi * C, sphi * S + cphi * C], axis=1)
        return jnp.max(_two_f_map(F_pix, sc_rot, gram_all))

    return jax.lax.map(one, jax.random.split(key, n_real))


def sky_scramble_background(
    sc_all: Float[Array, "npsr 2"],
    gram_all: Float[Array, "npsr 2 2"],
    cos_gwtheta: Float[Array, " npix"],
    gwphi: Float[Array, " npix"],
    n_real: int,
    key: Array,
) -> Float[Array, " n_real"]:
    """Sky-maximized ``2F`` under random pulsar sky positions (the null).

    Each realization redraws isotropic pulsar unit vectors -- destroying the geometric
    (antenna) coherence between the true pulsar array and the source direction -- and
    recomputes the sky-max ``2F``.  The matched filters ``(S, C)`` are position-
    independent (Earth-term quadratures), so only the antenna fold is redrawn.

    Parameters
    ----------
    sc_all, gram_all, cos_gwtheta, gwphi
        As in :func:`fstat_skymap` (no ``positions`` -- they are scrambled).
    n_real : int
        Number of null realizations.
    key : PRNGKey
        JAX random key.

    Returns
    -------
    (n_real,) array
        Sky-maximized ``2F`` for each sky-scrambled null realization.
    """
    npsr = sc_all.shape[0]

    def one(k):
        v = jax.random.normal(k, (npsr, 3))
        pos = v / jnp.linalg.norm(v, axis=1, keepdims=True)
        F_pix = _antenna_grid(pos, cos_gwtheta, gwphi)
        return jnp.max(_two_f_map(F_pix, sc_all, gram_all))

    return jax.lax.map(one, jax.random.split(key, n_real))


def pvalue(stat: float, background: Float[Array, " n_real"]) -> float:
    """One-sided p-value: fraction of the ``background`` at least as large as ``stat``."""
    return float((jnp.asarray(background) >= stat).mean())
