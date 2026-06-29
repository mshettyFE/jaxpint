"""Tests for jaxpint.bayes.priors: Prior ABC and concrete distributions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import pytest

from jaxpint.bayes import Gaussian, ImproperPrior, Prior, Uniform


jax.config.update("jax_enable_x64", True)


# ===========================================================================
# Construction & validation
# ===========================================================================


class TestUniformConstruction:
    def test_basic(self):
        p = Uniform(0.1, 10.0)
        assert p.low == 0.1
        assert p.high == 10.0
        assert p.is_proper
        assert p.support() == (0.1, 10.0)

    def test_low_must_be_less_than_high(self):
        with pytest.raises(ValueError, match="low < high"):
            Uniform(10.0, 5.0)

    def test_low_equal_high_rejected(self):
        with pytest.raises(ValueError, match="low < high"):
            Uniform(1.0, 1.0)


class TestGaussianConstruction:
    def test_basic(self):
        p = Gaussian(mu=0.973, sigma=0.20)
        assert p.mu == 0.973
        assert p.sigma == 0.20
        assert p.is_proper

    def test_sigma_must_be_positive(self):
        with pytest.raises(ValueError, match="sigma > 0"):
            Gaussian(mu=0.0, sigma=0.0)
        with pytest.raises(ValueError, match="sigma > 0"):
            Gaussian(mu=0.0, sigma=-1.0)


class TestImproperConstruction:
    def test_basic(self):
        p = ImproperPrior()
        assert not p.is_proper
        assert p.support() == (-jnp.inf, jnp.inf)


# ===========================================================================
# log_prob correctness
# ===========================================================================


class TestUniformLogProb:
    def test_inside_returns_neg_log_width(self):
        p = Uniform(0.0, 5.0)
        assert float(p.log_prob(jnp.array(2.0))) == pytest.approx(-jnp.log(5.0))

    def test_at_endpoints_in_support(self):
        p = Uniform(0.0, 5.0)
        assert jnp.isfinite(p.log_prob(jnp.array(0.0)))
        assert jnp.isfinite(p.log_prob(jnp.array(5.0)))

    def test_outside_is_neg_inf(self):
        p = Uniform(0.0, 5.0)
        assert float(p.log_prob(jnp.array(-1.0))) == -jnp.inf
        assert float(p.log_prob(jnp.array(6.0))) == -jnp.inf

    def test_vectorized_input(self):
        p = Uniform(0.0, 5.0)
        xs = jnp.array([-1.0, 0.0, 2.5, 5.0, 6.0])
        out = p.log_prob(xs)
        assert out.shape == xs.shape
        assert float(out[0]) == -jnp.inf
        assert jnp.isfinite(out[1])
        assert jnp.isfinite(out[2])
        assert jnp.isfinite(out[3])
        assert float(out[4]) == -jnp.inf


class TestGaussianLogProb:
    def test_at_mean_is_peak(self):
        p = Gaussian(mu=1.0, sigma=2.0)
        peak = float(p.log_prob(jnp.array(1.0)))
        # Peak = -0.5 log(2 pi) - log(sigma)
        expected = -0.5 * float(jnp.log(2 * jnp.pi)) - float(jnp.log(2.0))
        assert peak == pytest.approx(expected)

    def test_one_sigma_offset(self):
        p = Gaussian(mu=0.0, sigma=1.0)
        peak = float(p.log_prob(jnp.array(0.0)))
        offset = float(p.log_prob(jnp.array(1.0)))
        # log p(mu + sigma) = log p(mu) - 0.5
        assert offset == pytest.approx(peak - 0.5)

    def test_normalization(self):
        # Numerical integration of exp(log_prob) over wide range should ~ 1.
        p = Gaussian(mu=0.0, sigma=1.0)
        xs = jnp.linspace(-10, 10, 4001)
        dx = float(xs[1] - xs[0])
        integral = float(jnp.sum(jnp.exp(p.log_prob(xs))) * dx)
        assert integral == pytest.approx(1.0, abs=1e-3)

    def test_symmetry(self):
        p = Gaussian(mu=2.0, sigma=0.5)
        assert float(p.log_prob(jnp.array(2.5))) == pytest.approx(
            float(p.log_prob(jnp.array(1.5)))
        )


class TestImproperLogProb:
    def test_zero_everywhere(self):
        p = ImproperPrior()
        for x in [-1e10, 0.0, 1.0, 1e10]:
            assert float(p.log_prob(jnp.array(x))) == 0.0

    def test_vectorized(self):
        p = ImproperPrior()
        out = p.log_prob(jnp.array([1.0, 2.0, 3.0]))
        assert out.shape == (3,)
        assert jnp.all(out == 0.0)


# ===========================================================================
# Sampling
# ===========================================================================


class TestSampling:
    def test_uniform_in_support(self):
        p = Uniform(2.0, 5.0)
        samples = p.sample(jr.key(0), shape=(1000,))
        assert samples.shape == (1000,)
        assert jnp.all(samples >= 2.0)
        assert jnp.all(samples <= 5.0)

    def test_gaussian_mean_and_std(self):
        p = Gaussian(mu=3.0, sigma=2.0)
        samples = p.sample(jr.key(42), shape=(10000,))
        assert float(jnp.mean(samples)) == pytest.approx(3.0, abs=0.1)
        assert float(jnp.std(samples)) == pytest.approx(2.0, rel=0.05)

    def test_improper_sample_raises(self):
        p = ImproperPrior()
        with pytest.raises(NotImplementedError, match="improper"):
            p.sample(jr.key(0))


# ===========================================================================
# JAX integration: jit, grad, vmap, pytree
# ===========================================================================


class TestJaxIntegration:
    def test_jit_uniform(self):
        p = Uniform(0.0, 5.0)
        jit_lp = jax.jit(p.log_prob)
        assert float(jit_lp(jnp.array(2.0))) == float(p.log_prob(jnp.array(2.0)))

    def test_jit_gaussian(self):
        p = Gaussian(mu=1.0, sigma=0.5)
        jit_lp = jax.jit(p.log_prob)
        assert float(jit_lp(jnp.array(1.5))) == float(p.log_prob(jnp.array(1.5)))

    def test_grad_gaussian_at_mean_is_zero(self):
        p = Gaussian(mu=2.0, sigma=0.5)
        grad = float(jax.grad(p.log_prob)(jnp.array(2.0)))
        assert grad == pytest.approx(0.0, abs=1e-10)

    def test_grad_gaussian_offset(self):
        p = Gaussian(mu=0.0, sigma=1.0)
        # d/dx [-0.5 x^2] = -x at x = 1.5
        grad = float(jax.grad(p.log_prob)(jnp.array(1.5)))
        assert grad == pytest.approx(-1.5)

    def test_grad_improper_is_zero(self):
        p = ImproperPrior()
        grad = float(jax.grad(p.log_prob)(jnp.array(42.0)))
        assert grad == 0.0

    def test_vmap_gaussian(self):
        p = Gaussian(mu=0.0, sigma=1.0)
        xs = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = jax.vmap(p.log_prob)(xs)
        assert out.shape == xs.shape

    def test_pytree_registration(self):
        # equinox.Module ⇒ pytree. Static-field priors have no leaves but
        # still flatten/unflatten cleanly.
        p = Gaussian(mu=1.0, sigma=2.0)
        leaves, treedef = jtu.tree_flatten(p)
        rebuilt = jtu.tree_unflatten(treedef, leaves)
        assert isinstance(rebuilt, Gaussian)
        assert rebuilt.mu == 1.0
        assert rebuilt.sigma == 2.0


# ===========================================================================
# Polymorphism
# ===========================================================================


class TestPolymorphism:
    def test_all_subclass_prior(self):
        for p in [Uniform(0, 1), Gaussian(0, 1), ImproperPrior()]:
            assert isinstance(p, Prior)

    def test_proper_flag(self):
        assert Uniform(0, 1).is_proper is True
        assert Gaussian(0, 1).is_proper is True
        assert ImproperPrior().is_proper is False

    def test_log_prob_shape_preserved_across_types(self):
        # log_prob has the same call signature and is shape-preserving across
        # every Prior subclass (this is a dispatch/broadcasting check, not a
        # Uniform-specific one -- it exercises Uniform, Gaussian and Improper).
        priors = [Uniform(0, 1), Gaussian(0.5, 0.1), ImproperPrior()]
        x = jnp.array(0.5)
        results = [p.log_prob(x) for p in priors]
        assert all(r.shape == x.shape for r in results)
