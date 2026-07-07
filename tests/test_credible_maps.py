"""Tests for the 2-D HPD credible-region map primitives (jaxpint.stats.regions)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxpint.stats.regions import (
    credible_level_map,
    credible_region_area,
    gaussian_credible_area,
)


def test_flat_map_area_is_level_times_sky():
    # Flat posterior: the HPD region holding `level` mass is `level` of the sphere.
    npix = 3072
    pix_area = 4 * jnp.pi / npix
    logp = jnp.zeros(npix)
    for level in (0.5, 0.68, 0.9):
        a = credible_region_area(logp, pix_area, level)
        assert jnp.allclose(a, level * 4 * jnp.pi, rtol=2e-3)


def test_delta_map_is_one_pixel():
    npix = 3072
    pix_area = 4 * jnp.pi / npix
    logp = jnp.full(npix, -1e3).at[100].set(0.0)
    a = credible_region_area(logp, pix_area, 0.68)
    assert jnp.allclose(a, pix_area, rtol=1e-12)


def test_level_map_properties():
    logp = jax.random.normal(jax.random.PRNGKey(0), (500,))
    lev = credible_level_map(logp)
    # densest pixel -> exactly 0; all levels in [0, 1); nondecreasing as density drops.
    assert jnp.allclose(lev[jnp.argmax(logp)], 0.0, atol=1e-12)
    assert lev.min() >= 0.0
    assert lev.max() < 1.0
    order = jnp.argsort(-logp)
    assert jnp.all(jnp.diff(lev[order]) >= -1e-12)


def test_gaussian_limit_matches_analytic():
    # 2-D isotropic Gaussian on a flat equal-area grid -> area = pi * dchi2 * sigma^2,
    # i.e. gaussian_credible_area(det Sigma = sigma^4).  Converges as the grid refines.
    N, L, sigma = 401, 6.0, 1.0
    xs = jnp.linspace(-L, L, N)
    dx = xs[1] - xs[0]
    X, Y = jnp.meshgrid(xs, xs)
    logp = (-0.5 * (X**2 + Y**2) / sigma**2).ravel()
    for level in (0.68, 0.95):
        a = credible_region_area(logp, dx * dx, level)
        ana = gaussian_credible_area(jnp.asarray(sigma**4), level)
        assert jnp.allclose(a, ana, rtol=2e-2)
