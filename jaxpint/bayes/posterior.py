"""Posterior helpers for JaxPINT.

The posterior log-density is the sum of the log-likelihood and the log-prior
(up to a normalising constant).  This
module provides composition utilities that turn

    likelihood : params_dict -> log L
    priors     : dict[str, Prior]

into

    log_posterior : params_dict -> log L + sum_i log_prior_i(params_dict[i])

The resulting callable is what one would pass to a sampler (NumPyro,
blackjax, emcee).  It is JIT-compatible and differentiable in the values
of the parameter dict.
"""

from __future__ import annotations

from typing import Callable, Mapping

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.bayes.priors import Prior


__all__ = ["combine_log_prob", "log_prior_sum"]


def log_prior_sum(
    priors: Mapping[str, Prior],
    params: Mapping[str, Float[Array, "..."]],
) -> Float[Array, ""]:
    """Sum ``log_prob(params[name])`` over every (name, prior) in ``priors``.

    Parameters
    ----------
    priors
        Mapping from parameter names to :class:`Prior` instances.
    params
        Mapping from parameter names to current values.  Must contain at
        least every key in ``priors``; extras are ignored.

    Returns
    -------
    scalar
        Sum of the per-parameter log-priors.

    Notes
    -----
    The order of summation is the iteration order of ``priors``.  For
    JIT-traced inputs the result is a scalar JAX array; the iteration is
    Python-side, so ``priors`` should be a fixed dict at trace time.
    """
    total = jnp.float64(0.0)
    for name, prior in priors.items():
        total = total + prior.log_prob(params[name])
    return total


def combine_log_prob(
    log_likelihood: Callable[..., Float[Array, ""]],
    priors: Mapping[str, Prior],
    *,
    params_kw: str = "params",
) -> Callable[..., Float[Array, ""]]:
    """Wrap a log-likelihood callable into an unnormalized log-posterior.

    Parameters
    ----------
    log_likelihood
        Callable returning a scalar log-likelihood.  May accept any
        signature, but must take the parameter dictionary by keyword
        ``params_kw`` (default ``"params"``).
    priors
        Mapping from parameter names to :class:`Prior` instances.
    params_kw
        Name of the keyword argument carrying the parameter dictionary.

    Returns
    -------
    callable
        Returns a function with the same signature as ``log_likelihood``
        whose value is ``log_likelihood(...) + log_prior_sum(priors, params)``.

    Notes
    -----
    The returned function is the *unnormalized* log-posterior — the
    ``-log Z`` evidence term is omitted because every standard inference
    operation (MCMC ratios, gradient-based MAP, Hessian) is invariant to
    additive constants.  Code that needs the evidence (Bayes factors,
    nested sampling) computes it separately.
    """

    def log_post(*args, **kwargs):
        params = kwargs.get(params_kw)
        if params is None:
            raise TypeError(
                f"combine_log_prob: expected keyword argument "
                f"'{params_kw}' carrying the parameter dict."
            )
        return log_likelihood(*args, **kwargs) + log_prior_sum(priors, params)

    return log_post
