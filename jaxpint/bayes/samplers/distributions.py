"""Custom numpyro distributions for PTA priors.

Only distributions that numpyro does not already ship live here — anything
numpyro provides natively is used directly (e.g. enterprise's ``TruncNormal``
is exactly ``numpyro.distributions.TruncatedNormal(loc, scale, low, high)``,
same truncated-renormalization convention; pinned against enterprise in
tests/enterprise_checks).

**Arbitrary user priors (enterprise's ``UserParameter``):** there is no
adapter class, deliberately — numpyro's ``Distribution`` ABC *is* the
extension point, and :class:`LinearExp` below is the template to copy.  The
enterprise facets map directly: prior pdf → ``log_prob``, sampler →
``sample``, PPF → ``icdf``.  The contract that requires actual care: the
log-density must be JAX-differentiable on the support, and ``support`` must
be declared with the correct numpyro constraint — NUTS samples in
unconstrained space and derives the change-of-variables Jacobian from that
declaration, so a wrong ``support`` silently skews the posterior (see the
prior-only NUTS recovery test in test_bayes_samplers_priors.py, and the KS
tests there for how to validate a new distribution's shape).  ``icdf`` is
optional unless the prior must participate in inverse-CDF (hypercube)
transforms for external nested samplers.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array, random
from numpyro.distributions import Distribution, constraints
from numpyro.distributions.util import validate_sample

__all__ = ["LinearExp"]

_LN10 = jnp.log(10.0)


class LinearExp(Distribution):
    r"""Uniform in the linear quantity while sampling its log10.

    Enterprise's ``parameter.LinearExp`` — the standard upper-limit prior
    (GWB ``log10_A``, CW ``log10_h``): if ``x ~ LinearExp(pmin, pmax)`` then
    ``10**x ~ Uniform(10**pmin, 10**pmax)``, i.e.

    .. math::
        p(x) = \frac{\ln(10)\,10^x}{10^{p_\max} - 10^{p_\min}},
        \qquad p_\min \le x \le p_\max.

    Compared to a Uniform prior on ``x`` (the detection convention), this
    puts prior mass ∝ the linear amplitude, so the posterior upper limit is
    not dominated by prior volume at tiny amplitudes.

    ``cdf``/``icdf`` are analytic, so this composes with inverse-CDF
    (hypercube) transforms for nested sampling.
    """

    arg_constraints = {"pmin": constraints.real, "pmax": constraints.real}
    reparametrized_params = ["pmin", "pmax"]

    pmin: Array
    pmax: Array

    def __init__(self, pmin=-18.0, pmax=-11.0, *, validate_args=None):
        self.pmin = jnp.asarray(pmin)
        self.pmax = jnp.asarray(pmax)
        batch_shape = jnp.broadcast_shapes(self.pmin.shape, self.pmax.shape)
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.pmin, self.pmax)

    def sample(self, key, sample_shape=()):
        assert key is not None
        u = random.uniform(key, shape=sample_shape + self.batch_shape)
        return self.icdf(u)

    @validate_sample
    def log_prob(self, value):
        norm = 10.0**self.pmax - 10.0**self.pmin
        return jnp.log(_LN10) + value * _LN10 - jnp.log(norm)

    def cdf(self, value):
        lo = 10.0**self.pmin
        return (10.0**value - lo) / (10.0**self.pmax - lo)

    def icdf(self, q):
        lo = 10.0**self.pmin
        return jnp.log10(lo + q * (10.0**self.pmax - lo))
