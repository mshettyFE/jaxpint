"""Tests for jaxpint.stats.grids (numerical grid marginalization).

The correctness tests use an *independent* oracle -- a naive linear-space sum
(numpy), or the analytic Gaussian-integral closed form -- rather than recomputing
the ``logsumexp`` the implementation uses.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr

from jaxpint.stats.grids import grid_log_marginal, grid_log_profile


def test_uniform_marginal_equals_naive_linear_space_mean():
    # log of the arithmetic mean of the likelihoods, computed naively in numpy
    # (no logsumexp) -- the definition the function must satisfy.
    ll = jnp.array([0.3, -1.2, 2.0, 0.5])
    ref = float(np.log(np.mean(np.exp(np.asarray(ll)))))
    assert jnp.isclose(grid_log_marginal(ll), ref)


def test_weighted_marginal_equals_naive_weighted_sum():
    # log of the prior-weighted likelihood sum, computed naively in numpy.
    ll = jnp.array([0.0, 1.0, -0.5, 2.0])
    w = np.array([0.1, 0.4, 0.3, 0.2])  # a (normalized) prior over the grid
    ref = float(np.log(np.sum(np.exp(np.asarray(ll)) * w)))
    assert jnp.isclose(grid_log_marginal(ll, jnp.log(jnp.asarray(w))), ref)


def test_marginal_converges_to_analytic_gaussian_integral():
    # The semantic claim: numerically marginalizing a Gaussian likelihood over a
    # uniform prior on [a, b] must converge to the closed-form integral solution
    mu, sigma, a, b, n = 0.4, 0.7, -2.0, 3.0, 4000
    grid = jnp.linspace(a, b, n)
    logL = -0.5 * ((grid - mu) / sigma) ** 2  # unnormalized Gaussian log-likelihood
    integral = (
        sigma
        * np.sqrt(2 * np.pi)
        * float(ndtr((b - mu) / sigma) - ndtr((a - mu) / sigma))
    )
    expected = float(np.log(integral / (b - a)))
    assert jnp.isclose(grid_log_marginal(logL), expected, atol=1e-3)


def test_constant_grid_returns_the_constant():
    # log mean of n equal values is that value (the weights must cancel)
    assert jnp.isclose(grid_log_marginal(jnp.full((5,), 1.7)), 1.7)


def test_flat_weights_recover_the_uniform_default():
    # the convenience None-branch must equal the general weighted branch fed
    # explicit uniform weights log(1/n) -- a consistency check between code paths.
    ll = jnp.array([0.2, -0.7, 1.1])
    n = ll.shape[0]
    flat = jnp.full((n,), -jnp.log(float(n)))
    assert jnp.isclose(grid_log_marginal(ll), grid_log_marginal(ll, flat))


def test_marginal_never_exceeds_profile():
    # mean exp <= max exp, so log-marginal <= log-profile for any input
    ll = jax.random.normal(jax.random.key(0), (50,)) * 3.0
    assert grid_log_marginal(ll) <= grid_log_profile(ll) + 1e-9


def test_dominant_point_picks_out_the_mode():
    # one point dwarfs the rest: profile -> that point, marginal -> point - log n
    ll = jnp.array([-1e3, 5.0, -1e3, -1e3])
    assert jnp.isclose(grid_log_profile(ll), 5.0)
    assert jnp.isclose(grid_log_marginal(ll), 5.0 - jnp.log(4.0), atol=1e-6)


def test_reduces_over_last_axis_with_a_batch():
    ll = jnp.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])  # (n_batch=2, n_grid=3)
    m, p = grid_log_marginal(ll), grid_log_profile(ll)
    assert m.shape == (2,) and p.shape == (2,)
    # batched result agrees with calling each row separately
    assert jnp.allclose(
        m, jnp.array([grid_log_marginal(ll[0]), grid_log_marginal(ll[1])])
    )
    assert jnp.allclose(
        p, jnp.array([grid_log_profile(ll[0]), grid_log_profile(ll[1])])
    )


def test_jit_compatible():
    ll = jnp.array([0.5, -0.5, 1.5, 0.0])
    assert jnp.isclose(jax.jit(grid_log_marginal)(ll), grid_log_marginal(ll))
    assert jnp.isclose(jax.jit(grid_log_profile)(ll), grid_log_profile(ll))
