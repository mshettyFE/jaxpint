"""Upper limits and credible/confidence regions from tabulated or Gaussian densities.

Generic, signal-model-free bound primitives: extract an upper limit from a
(truncated-Gaussian, Gaussian-mixture, or grid-tabulated) density, or a 2-D
region area from a Gaussian/Laplace one.

The functions are arm-neutral -- the identical math serves both inference
vocabularies.  Read through Bayesian glasses, the density is a posterior under
an improper-uniform prior (with, for the upper-limit kernels, a non-negativity
constraint ``x >= 0``) and the regions are *credible* regions.  Read through
frequentist glasses, :func:`gaussian_credible_area` applied to an inverse
Fisher matrix is a Wilks *confidence* ellipse (which is how
:mod:`jaxpint.pta.cw_localization` uses it), and the grid quantiles apply to
any normalized score surface.  Names keep the ``credible`` vocabulary for
continuity with the literature the implementations follow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr, ndtri
from jaxtyping import Array, Float

if TYPE_CHECKING:
    from numpyro.distributions import Distribution

__all__ = [
    "truncated_gaussian_upper_limit",
    "mixture_truncated_gaussian_upper_limit",
    "grid_credible_upper_limit",
    "masked_log_prior",
    "gaussian_credible_area",
    "credible_level_map",
    "credible_region_area",
]


def masked_log_prior(
    x: Float[Array, " n"],
    prior: "Distribution",
) -> Float[Array, " n"]:
    """Prior log-density at grid points ``x``, ``-inf`` outside the support.

    numpyro's ``log_prob`` does **not** self-mask, so before folding a bounded
    prior into grid weights, points outside its support must be set to ``-inf``
    by hand.  A ``None`` support (numpyro's unconstrained base distribution)
    means every point is in support.
    """
    support = prior.support
    in_support = (
        jnp.ones(x.shape, dtype=bool)
        if support is None
        else jnp.asarray(support(x), dtype=bool)
    )
    return jnp.where(in_support, prior.log_prob(x), -jnp.inf)


def truncated_gaussian_upper_limit(
    mu: Float[Array, ""],
    sigma: Float[Array, ""],
    level: float = 0.95,
) -> Float[Array, ""]:
    r"""``level`` quantile of ``N(mu, sigma^2)`` truncated to ``x >= 0``.

    Closed form, from inverting the truncated-normal CDF:

    .. math::
        x^{UL} = \mu + \sigma\,\Phi^{-1}\!\big(p + (1-p)\,\Phi(-\mu/\sigma)\big),

    with ``p = level``.  Pure distribution math; its use as a Bayesian upper
    limit corresponds to a Gaussian likelihood under a uniform prior on
    ``x >= 0`` in the ``x_max -> inf`` limit — exact while the data constrain
    ``x`` (``sigma`` small); see the callers for the finite-``x_max`` caveat.
    """
    assert level > 0
    assert level < 1.0
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
    r"""``level`` quantile of a weighted mixture of truncated (``x >= 0``) Gaussians.

    The density is ``p(x) \propto sum_k w_k N(x; mu_k, sigma_k^2)`` on
    ``x >= 0``; its CDF is the weight-normalized sum of per-component
    truncated-normal CDFs, and the ``level`` quantile is found by bisection.
    (In the CW upper-limit callers the mixture arises from marginalizing a
    Gaussian likelihood over source orientations, making the quantile a
    Bayesian upper limit — but the function itself just inverts the CDF of
    the mixture it is handed.)

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
    log_density: Float[Array, " n"],
    level: float = 0.95,
) -> Float[Array, ""]:
    """``level`` quantile of a log-density tabulated on a sorted **uniform** grid.

    Normalizes ``exp(log_density)`` to a CDF over ``grid`` (assumed
    monotonically increasing) and interpolates the ``level`` quantile.
    ``grid`` must span the ``level`` mass.  Use this when the density is not a
    (mixture of) Gaussian(s) and only a numerical evaluation is available.

    The bare ``cumsum`` weights every grid point equally, i.e. the quadrature
    measure is uniform grid spacing — on a non-uniform grid the quantile would
    be measure-distorted.  (Under Bayesian use this is also where the implicit
    flat prior lives: feed ``log posterior = log likelihood`` on a uniform grid
    and the result is the flat-prior credible upper limit.)
    """
    w = jnp.exp(log_density - jnp.max(log_density))
    cdf = jnp.cumsum(w)
    cdf = cdf / cdf[-1]
    return jnp.interp(jnp.asarray(level, dtype=grid.dtype), cdf, grid)


def gaussian_credible_area(
    det_cov: Float[Array, ""],
    level: float = 0.9,
) -> Float[Array, ""]:
    r"""``level`` 2-D ellipse area of a Gaussian density, in the parameters' units.

    For ``N(., Sigma)`` with covariance determinant ``det_cov = det(Sigma)`` the
    ``level`` ellipse has area

    .. math::
        A = \pi \cdot \Delta\chi^2 \cdot \sqrt{\det \Sigma},
        \qquad \Delta\chi^2 = -2\ln(1 - \text{level})

    (``Δχ²`` is exactly the 2-dof chi-square quantile).  The same number is the
    Bayesian *credible* area of a Gaussian posterior and the frequentist Wilks
    *confidence* area from an inverse Fisher matrix — this is the one function
    in the module where the two vocabularies coincide identically, which is why
    Fisher localization consumes it directly.  Pass ``det_cov = inf`` for a
    degenerate / unphysical density to propagate ``inf`` rather than raising.
    """
    dchi2 = -2.0 * jnp.log(1.0 - level)
    return jnp.pi * dchi2 * jnp.sqrt(det_cov)


def credible_level_map(
    log_density: Float[Array, " n_pix"],
) -> Float[Array, " n_pix"]:
    r"""Highest-density credible level of each pixel of a tabulated density.

    Greedy HPD construction (Singer & Price 2016 [cm_sp16]_): rank pixels by
    density and assign each the cumulative normalized mass of all *strictly
    denser* pixels (exclusive convention -- the densest pixel gets ``0``).  The
    smallest ``level`` region is then ``{level_map < level}``.

    Assumes equal-area pixels, so a pixel's mass is ``\propto
    exp(log_density)``.  The 2-D / map analog of the CDF used by
    :func:`grid_credible_upper_limit`.

    This is the least arm-neutral function in the module: the output is
    cumulative probability mass over *parameter* space, which is intrinsically
    a Bayesian (credible-region) calibration -- under a flat prior on
    equal-area pixels, ``log_density = log likelihood`` makes this the
    posterior HPD level.  A frequentist Wilks region ranks pixels in the
    *identical order* (by density) but calibrates levels via the
    likelihood-ratio chi-square instead of mass; that would be a sibling
    function, not this one.

    Parameters
    ----------
    log_density : (n_pix,) array
        Unnormalized log-density over a pixelized parameter space (e.g. a
        HEALPix sky map).

    Returns
    -------
    (n_pix,) array
        Per-pixel exclusive HPD level in ``[0, 1)``.

    References
    ----------
    .. [cm_sp16] Singer & Price (2016), PRD 93, 024013.
    """
    p = jnp.exp(log_density - jnp.max(log_density))
    p = p / jnp.sum(p)
    order = jnp.argsort(-p)  # descending density
    excl = jnp.cumsum(p[order]) - p[order]  # mass strictly above each pixel
    return jnp.zeros_like(excl).at[order].set(excl)


def credible_region_area(
    log_density: Float[Array, " n_pix"],
    pixel_area: Float[Array, ""],
    level: float = 0.9,
) -> Float[Array, ""]:
    r"""Area of the smallest ``level`` HPD region of a pixelized density.

    Counts the pixels in the HPD region ``{credible_level_map < level}`` and
    multiplies by ``pixel_area`` (constant -- equal-area pixels, e.g.
    ``healpy.nside2pixarea(nside)``).  Returns an area in the units of
    ``pixel_area`` (steradians if ``pixel_area`` is).  The map analog of
    :func:`grid_credible_upper_limit`; for a Gaussian density it converges to
    :func:`gaussian_credible_area` as the pixel scale shrinks.  Inherits the
    mass-based (credible, not Wilks) level calibration of
    :func:`credible_level_map` -- see the note there.

    Parameters
    ----------
    log_density : (n_pix,) array
        Unnormalized log-density over equal-area pixels.
    pixel_area : scalar
        Area of one pixel.
    level : float
        Credible level (default 0.9).

    Notes
    -----
    Equal-area pixels only.  This generalizes to a **mixed-resolution** map (e.g.
    adaptively-refined HEALPix leaves of differing ``nside``) without an API break:
    pass a per-leaf ``area`` array, separate *density* from *mass* in
    :func:`credible_level_map` -- rank by ``exp(log_density)`` (density) but
    accumulate ``exp(log_density)·area`` (mass) -- and return ``Σ area`` over the
    included leaves here.  The constant-area case is exactly that with ``area``
    constant (density ∝ mass; ``Σ area = n_in · pixel_area``).  Deferred until the
    adaptive map tier provides real mixed-resolution leaves to test against.
    """
    levels = credible_level_map(log_density)
    n_in = jnp.sum(levels < level)
    return n_in * pixel_area
