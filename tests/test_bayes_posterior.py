"""Tests for jaxpint.bayes.posterior: log-prior summation and likelihood composition."""

import jax
import jax.numpy as jnp
import pytest

from jaxpint.bayes import (
    Gaussian,
    ImproperPrior,
    Uniform,
    combine_log_prob,
    log_prior_sum,
)


jax.config.update("jax_enable_x64", True)


# ===========================================================================
# log_prior_sum
# ===========================================================================


class TestLogPriorSum:
    def test_basic_sum(self):
        priors = {"a": Gaussian(0, 1), "b": Gaussian(0, 1)}
        params = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        # 2 * peak of a unit Gaussian
        peak = -0.5 * float(jnp.log(2 * jnp.pi))
        assert float(log_prior_sum(priors, params)) == pytest.approx(2 * peak)

    def test_improper_contributes_zero(self):
        priors = {"a": ImproperPrior(), "b": Gaussian(0, 1)}
        params = {"a": jnp.array(99.0), "b": jnp.array(0.0)}
        peak = -0.5 * float(jnp.log(2 * jnp.pi))
        assert float(log_prior_sum(priors, params)) == pytest.approx(peak)

    def test_outside_uniform_is_neg_inf(self):
        priors = {"a": Uniform(0, 1)}
        params = {"a": jnp.array(2.0)}
        assert float(log_prior_sum(priors, params)) == -jnp.inf

    def test_extras_in_params_ignored(self):
        priors = {"a": Gaussian(0, 1)}
        params = {"a": jnp.array(0.0), "extra": jnp.array(99.0)}
        peak = -0.5 * float(jnp.log(2 * jnp.pi))
        assert float(log_prior_sum(priors, params)) == pytest.approx(peak)

    def test_jit_friendly(self):
        priors = {"a": Gaussian(0, 1), "b": Uniform(-2, 2)}

        @jax.jit
        def f(params):
            return log_prior_sum(priors, params)

        params = {"a": jnp.array(0.0), "b": jnp.array(0.5)}
        assert jnp.isfinite(f(params))


# ===========================================================================
# combine_log_prob
# ===========================================================================


class TestCombineLogProb:
    def test_combines_likelihood_and_priors(self):
        priors = {"a": Gaussian(0, 1)}

        def log_lik(*, params):
            return -0.5 * params["a"] ** 2  # peaks at 0

        log_post = combine_log_prob(log_lik, priors)
        peak = float(log_post(params={"a": jnp.array(0.0)}))
        # peak = log L(0) + log_prior(0)
        gauss_peak = -0.5 * float(jnp.log(2 * jnp.pi))
        assert peak == pytest.approx(0.0 + gauss_peak)

    def test_keyword_only_params(self):
        priors = {"a": Gaussian(0, 1)}

        def log_lik(*, params):
            return jnp.float64(0.0)

        log_post = combine_log_prob(log_lik, priors)
        # Calling without `params` keyword raises
        with pytest.raises(TypeError, match="params"):
            log_post()

    def test_jittable(self):
        priors = {"a": Uniform(-1, 1)}

        def log_lik(*, params):
            return -params["a"] ** 2

        log_post = jax.jit(combine_log_prob(log_lik, priors))
        v = log_post(params={"a": jnp.array(0.5)})
        assert jnp.isfinite(v)

    def test_grad_through_posterior(self):
        priors = {"x": Gaussian(0, 1)}

        def log_lik(*, params):
            return -0.5 * (params["x"] - 1) ** 2  # peaks at 1

        log_post = combine_log_prob(log_lik, priors)
        # Gradient at x=0: dL/dx = -(x-1) = 1, dprior/dx = -x = 0 → total grad = 1
        grad = jax.grad(lambda x: log_post(params={"x": x}))
        assert float(grad(jnp.array(0.0))) == pytest.approx(1.0)
        # At MAP (x = 0.5, the geometric average): both grads should sum to zero
        assert float(grad(jnp.array(0.5))) == pytest.approx(0.0)
