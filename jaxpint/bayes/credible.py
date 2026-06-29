"""Credible intervals, upper limits, and credible regions from posteriors.

Generic, signal-model-free Bayesian credible-bound primitives: extract an upper
limit from a (truncated-Gaussian, Gaussian-mixture, or grid-tabulated) posterior,
or a 2-D credible-region area from a Gaussian/Laplace posterior.

All functions assume an improper-uniform prior on the parameter (and, for the
upper-limit kernels, a non-negativity constraint ``x >= 0``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, Float

__all__ = [
    "truncated_gaussian_upper_limit",
    "mixture_truncated_gaussian_upper_limit",
    "grid_credible_upper_limit",
    "gaussian_credible_area",
]


def truncated_gaussian_upper_limit(
    mu: Float[Array, ""],
    sigma: Float[Array, ""],
    level: float = 0.95,
) -> Float[Array, ""]:
    r"""``level`` upper limit of ``N(mu, sigma^2)`` truncated to ``x >= 0``.

    Closed form for a uniform prior on ``x >= 0`` and a Gaussian likelihood, in
    the ``x_max -> inf`` limit:

    .. math::
        x^{UL} = \mu + \sigma\,\Phi^{-1}\!\big(p + (1-p)\,\Phi(-\mu/\sigma)\big),

    with ``p = level``.  Exact while the data constrain ``x`` (``sigma`` small);
    see the callers for the finite-``x_max`` caveat.
    """
    assert(level > 0)
    assert (level < 1.0)
    phi_lo = ndtr(-mu / sigma)  # Phi(-mu/sigma): mass below the x=0 wall
    q = level + (1.0 - level) * phi_lo
    return mu + sigma * ndtri(q)


def mixture_truncated_gaussian_upper_limit(
    mu: Float[Array, " n"],
    sigma: Float[Array, " n"],
    log_weights: Float[Array, " n"],
    level: float = 0.95,
    n_iter: int = 60,
) -> Float[Array, ""]:
    r"""``level`` upper limit of a weighted mixture of truncated (``x >= 0``) Gaussians.

    The posterior is ``p(x) \propto sum_k w_k N(x; mu_k, sigma_k^2)`` on
    ``x >= 0``; its CDF is the weight-normalized sum of per-component
    truncated-normal CDFs, and the ``level`` quantile is found by bisection.

    Parameters
    ----------
    mu, sigma : (n,) arrays
        Per-component mean and standard deviation.
    log_weights : (n,) array
        Per-component log mixture weight, up to a common additive constant
        (normalized internally by subtracting the max — cancels in the CDF
        ratio).
    level : float
        Credible level (default 0.95).
    n_iter : int
        Bisection iterations (default 60 -> float64-tight).

    Notes
    -----
    Reduces to :func:`truncated_gaussian_upper_limit` for a single component.
    """
    w = jnp.exp(log_weights - jnp.max(log_weights))
    phi_lo = ndtr(-mu / sigma)  # mass below the x=0 wall, per component
    phi_hi = ndtr(mu / sigma)  # component mass over [0, inf)
    denom = jnp.sum(w * phi_hi)

    def cdf(H):
        num = jnp.sum(w * (ndtr((H - mu) / sigma) - phi_lo))
        return num / denom

    # The mixture quantile is below max_k(mu_k) + a few sigma; bracket generously.
    H_hi0 = jnp.max(mu) + 12.0 * jnp.max(sigma)

    def body(_, bounds):
        lo, hi = bounds
        mid = 0.5 * (lo + hi)
        go_up = cdf(mid) < level
        return (jnp.where(go_up, mid, lo), jnp.where(go_up, hi, mid))

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (jnp.zeros_like(H_hi0), H_hi0))
    return 0.5 * (lo + hi)


def grid_credible_upper_limit(
    grid: Float[Array, " n"],
    log_post: Float[Array, " n"],
    level: float = 0.95,
) -> Float[Array, ""]:
    """``level`` upper limit from a log-posterior tabulated on a sorted grid.

    Normalizes ``exp(log_post)`` to a CDF over ``grid`` (assumed monotonically
    increasing) and interpolates the ``level`` quantile.  ``grid`` must span the
    ``level`` mass.  Use this when the posterior is not a (mixture of) Gaussian(s)
    and only a numerical evaluation is available.
    """
    w = jnp.exp(log_post - jnp.max(log_post))
    cdf = jnp.cumsum(w)
    cdf = cdf / cdf[-1]
    return jnp.interp(jnp.asarray(level, dtype=grid.dtype), cdf, grid)


def gaussian_credible_area(
    det_cov: Float[Array, ""],
    level: float = 0.9,
) -> Float[Array, ""]:
    r"""``level``-credible 2-D area of a Gaussian posterior, in the parameters' units.

    For ``N(., Sigma)`` with covariance determinant ``det_cov = det(Sigma)`` the
    ``level``-credible ellipse has area

    .. math::
        A = \pi \cdot \Delta\chi^2 \cdot \sqrt{\det \Sigma},
        \qquad \Delta\chi^2 = -2\ln(1 - \text{level})

    (``Δχ²`` for 2 degrees of freedom).  Pass ``det_cov = inf`` for a degenerate
    / unphysical posterior to propagate ``inf`` rather than raising.
    """
    dchi2 = -2.0 * jnp.log(1.0 - level)
    return jnp.pi * dchi2 * jnp.sqrt(det_cov)
