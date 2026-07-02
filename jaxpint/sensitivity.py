"""Frequentist F-statistic detection sensitivity for continuous gravitational waves.

The frequentist detection arm -- as opposed to the Bayesian upper-limit /
credible-area machinery in :mod:`jaxpint.bayes`.  The detection statistic ``2F`` is
``chi^2(dof)`` under the null and ``ncx2(dof, lambda)`` under a signal of
noncentrality ``lambda`` (``dof`` = the amplitude-span rank, 4 for the Earth-term
orientation span).  Detection at false-alarm probability ``fap`` requires
``2F > chi2_threshold(fap, dof)``; the strain sensitivity ``h0_min`` is the amplitude
at which the *orientation-averaged* detection probability reaches ``beta``.

The F-statistic for continuous-wave detection in pulsar timing arrays was proposed
by Ellis, Siemens & Creighton (2012), ApJ 756, 175 -- arXiv:1204.4218
(https://arxiv.org/abs/1204.4218).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.typing import ArrayLike
from jaxtyping import Array, Float
from scipy.stats import chi2, ncx2

__all__ = ["chi2_threshold", "detection_probability", "h0_min_from_lambda"]


def chi2_threshold(fap: float, dof: int = 4) -> float:
    """2F detection threshold at false-alarm probability ``fap``.

    The upper-tail ``1 - fap`` quantile of the null ``chi^2(dof)`` distribution of
    ``2F`` -- the value the null exceeds with probability ``fap``.

    Parameters
    ----------
    fap : float
        False-alarm probability (the upper-tail mass under the null).
    dof : int
        Degrees of freedom of the null ``chi^2`` (4 for the Earth-term orientation span).

    Returns
    -------
    float
        The 2F detection threshold.
    """
    return float(chi2.ppf(1.0 - fap, dof))


def detection_probability(
    threshold: float, lam: ArrayLike, dof: ArrayLike = 4
) -> Array:
    """Detection probability ``P[2F > threshold | signal]``.

    The survival function of ``ncx2(dof, lam)`` at ``threshold``.

    Parameters
    ----------
    threshold : float
        2F detection threshold (from :func:`chi2_threshold`).
    lam : array-like
        Noncentrality ``lambda`` of the signal; may be array-valued.  ``lam = 0``
        recovers the null ``chi^2(dof)`` upper tail.
    dof : array-like
        Degrees of freedom; broadcasts against ``lam``.

    Returns
    -------
    Array
        Detection probability, broadcast over ``lam`` / ``dof``.
    """
    return jnp.asarray(ncx2.sf(threshold, dof, lam))


def h0_min_from_lambda(
    threshold: float,
    lam1: Float[Array, "... n_theta"],
    dof: ArrayLike = 4,
    beta: float = 0.95,
    *,
    tol: float = 1e-3,
    lo: float = 1e-18,
    hi: float = 1e-10,
    max_grow: int = 200,
    max_iter: int = 80,
) -> Float[Array, "..."]:
    """Strain ``h0`` at which the orientation-averaged detection probability is ``beta``.

    Solves, per case, ``mean_theta P[ncx2(dof, h0^2 * lam1) > threshold] = beta`` by
    geometric bisection (``P_det`` is monotone in ``h0``, which spans decades).

    Parameters
    ----------
    threshold : float
        2F detection threshold (from :func:`chi2_threshold`).
    lam1 : (..., n_theta) or (n_theta,) array
        Per-orientation noncentralities of a **unit-strain** (``h0 = 1``) source; the
        detection probability is averaged over the last (orientation) axis.  A leading
        axis (e.g. sky pixels) is handled vectorized.
    dof : int or array
        Degrees of freedom; broadcasts to the leading shape.
    beta : float
        Target orientation-averaged detection probability.
    tol : float
        Relative bracket width at which the bisection stops.
    lo, hi : float
        Initial ``h0`` bracket; ``hi`` is grown geometrically until it detects.

    Returns
    -------
    (...) array
        ``h0_min`` matching the leading shape of ``lam1`` (0-d when ``lam1`` is 1-D).
    """
    lam = jnp.asarray(lam1)
    lead = lam.shape[:-1]  # (); () for a single 1-D case -> a 0-d result
    dof_col = jnp.asarray(dof)[
        ..., None
    ]  # trailing theta axis so per-case dof broadcasts

    def pdet(h0: Array) -> Array:  # averaged detection probability, shape `lead`
        p = ncx2.sf(threshold, dof_col, (h0[..., None] ** 2) * lam)
        return jnp.asarray(p).mean(-1)

    lo_a = jnp.full(lead, lo)
    hi_a = jnp.full(lead, hi)
    for _ in range(max_grow):  # ensure hi detects everywhere
        under = pdet(hi_a) < beta
        if not bool(under.any()):
            break
        hi_a = jnp.where(under, hi_a * 4.0, hi_a)
    for _ in range(max_iter):  # geometric bisection
        mid = jnp.sqrt(lo_a * hi_a)
        ok = pdet(mid) >= beta
        hi_a = jnp.where(ok, mid, hi_a)
        lo_a = jnp.where(ok, lo_a, mid)
        if bool(jnp.all(hi_a / lo_a - 1.0 < tol)):
            break
    return jnp.sqrt(lo_a * hi_a)
