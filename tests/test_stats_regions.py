"""Tests for the generic credible-bound primitives (:mod:`jaxpint.stats.regions`).

These are signal-model-free Bayesian kernels (upper limits from truncated /
mixture / grid posteriors, 2-D Gaussian credible areas); the CW routines in
:mod:`jaxpint.pta` are thin callers of them.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from jax.scipy.special import ndtr

from jaxpint.stats import (
    gaussian_credible_area,
    grid_credible_upper_limit,
    mixture_truncated_gaussian_upper_limit,
    truncated_gaussian_upper_limit,
)

jax.config.update("jax_enable_x64", True)

# Φ⁻¹(0.975): the one-sided 95% z-score, used as an implementation-independent
# reference for the zero-mean truncated-Gaussian upper limit.
_Z_975 = 1.959963984540054


def _trunc_normal_cdf(H, mu, sigma):
    """CDF of N(mu, sigma²) truncated to x >= 0 (x_max -> inf)."""
    lo = ndtr(-mu / sigma)
    return (ndtr((H - mu) / sigma) - lo) / (1.0 - lo)


class TestTruncatedGaussianUpperLimit:
    def test_zero_mean_is_z_sigma(self):
        # mu = 0 → UL = sigma · Φ⁻¹((1+level)/2) = sigma · Φ⁻¹(0.975) at level 0.95.
        for sigma in (0.5, 2.0, 100.0):
            ul = float(
                truncated_gaussian_upper_limit(
                    jnp.asarray(0.0), jnp.asarray(sigma), 0.95
                )
            )
            npt.assert_allclose(ul, _Z_975 * sigma, rtol=1e-9)

    def test_cdf_at_ul_equals_level(self):
        # Defining property: the truncated-normal CDF at the UL equals `level`.
        for mu, sigma, level in [(0.0, 1.0, 0.95), (3.0, 1.0, 0.9), (-2.0, 0.5, 0.68)]:
            ul = truncated_gaussian_upper_limit(
                jnp.asarray(mu), jnp.asarray(sigma), level
            )
            npt.assert_allclose(
                float(_trunc_normal_cdf(ul, mu, sigma)),
                level,
                rtol=1e-9,
                err_msg=f"mu={mu}, sigma={sigma}, level={level}",
            )

    def test_monotonic_in_level(self):
        f = lambda lv: float(
            truncated_gaussian_upper_limit(jnp.asarray(1.0), jnp.asarray(0.7), lv)
        )
        assert f(0.68) < f(0.9) < f(0.95) < f(0.99)


class TestMixtureUpperLimit:
    def test_single_component_matches_truncated(self):
        mu, sigma, level = 2.0, 0.8, 0.95
        mix = float(
            mixture_truncated_gaussian_upper_limit(
                jnp.array([mu]), jnp.array([sigma]), jnp.array([0.0]), level
            )
        )
        single = float(
            truncated_gaussian_upper_limit(jnp.asarray(mu), jnp.asarray(sigma), level)
        )
        npt.assert_allclose(mix, single, rtol=1e-9)

    def test_identical_components_match_single(self):
        # A mixture of identical components is that single component.
        mu, sigma, level = 1.5, 1.2, 0.9
        n = 5
        mix = float(
            mixture_truncated_gaussian_upper_limit(
                jnp.full(n, mu), jnp.full(n, sigma), jnp.zeros(n), level
            )
        )
        single = float(
            truncated_gaussian_upper_limit(jnp.asarray(mu), jnp.asarray(sigma), level)
        )
        npt.assert_allclose(mix, single, rtol=1e-9)

    def test_mixture_cdf_at_ul_equals_level(self):
        mu = jnp.array([0.0, 2.0, 5.0])
        sigma = jnp.array([1.0, 0.5, 2.0])
        log_w = jnp.array([0.0, -0.3, 0.7])
        level = 0.95
        ul = mixture_truncated_gaussian_upper_limit(mu, sigma, log_w, level)
        # Evaluate the mixture's truncated-normal CDF at the returned UL.
        w = jnp.exp(log_w - jnp.max(log_w))
        lo = ndtr(-mu / sigma)
        num = jnp.sum(w * (ndtr((ul - mu) / sigma) - lo))
        den = jnp.sum(w * ndtr(mu / sigma))
        npt.assert_allclose(float(num / den), level, rtol=1e-9)


class TestGridUpperLimit:
    def test_uniform_posterior(self):
        # Flat posterior on [0, H] → level quantile is level·H.
        H = 4.0
        grid = jnp.linspace(0.0, H, 100001)
        ul = float(grid_credible_upper_limit(grid, jnp.zeros_like(grid), 0.9))
        npt.assert_allclose(ul, 0.9 * H, rtol=1e-4)

    def test_matches_truncated_gaussian(self):
        # A finely-gridded (un-normalized) truncated Gaussian must reproduce the
        # closed-form truncated_gaussian_upper_limit — cross-checks two kernels.
        mu, sigma, level = 3.0, 1.0, 0.9
        grid = jnp.linspace(0.0, mu + 12.0 * sigma, 200001)
        log_post = -0.5 * ((grid - mu) / sigma) ** 2
        grid_ul = float(grid_credible_upper_limit(grid, log_post, level))
        analytic = float(
            truncated_gaussian_upper_limit(jnp.asarray(mu), jnp.asarray(sigma), level)
        )
        npt.assert_allclose(grid_ul, analytic, rtol=1e-3)


class TestGaussianCredibleArea:
    def test_matches_formula(self):
        for det_cov, level in [(0.25, 0.9), (1.0, 0.95), (4.0, 0.68)]:
            got = float(gaussian_credible_area(jnp.asarray(det_cov), level))
            ref = np.pi * (-2.0 * np.log(1.0 - level)) * np.sqrt(det_cov)
            npt.assert_allclose(got, ref, rtol=1e-12, err_msg=f"det_cov={det_cov}")

    def test_infinite_det_propagates(self):
        assert bool(jnp.isinf(gaussian_credible_area(jnp.asarray(jnp.inf), 0.9)))

    def test_area_scales_with_sqrt_det(self):
        # Σ -> 4Σ (det -> 16 det) quadruples the area (√16 = 4).
        a = float(gaussian_credible_area(jnp.asarray(1.0), 0.9))
        a16 = float(gaussian_credible_area(jnp.asarray(16.0), 0.9))
        npt.assert_allclose(a16, 4.0 * a, rtol=1e-12)
