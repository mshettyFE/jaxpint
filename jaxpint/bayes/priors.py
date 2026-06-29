"""Prior distributions for JaxPINT's Bayesian-inference layer.

A :class:`Prior` is a small, JIT-compatible object that holds the parameters
of a single-parameter prior distribution and exposes its log-density via
:meth:`Prior.log_prob`.  Concrete subclasses provide the standard shapes
encountered in pulsar-timing-array analysis: :class:`Uniform`,
:class:`Gaussian`, and :class:`ImproperPrior` (the discovery-equivalent
flat improper prior).

Priors are *not* keyed by parameter name — they hold only the prior's
shape and values.  Association with parameter names lives one level up,
in the user-constructed ``dict[str, Prior]`` that downstream helpers
(:mod:`jaxpint.bayes.defaults`, :func:`combine_log_prob`,
``jaxpint.bayes.marginal``) consume.

All priors are :class:`equinox.Module` subclasses, so they are
automatically registered as JAX pytrees and compose with ``jax.jit``,
``jax.grad``, and ``jax.vmap``.

"""

from __future__ import annotations

from typing import Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float


__all__ = [
    "Prior",
    "Uniform",
    "Gaussian",
    "ImproperPrior",
]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Prior(eqx.Module):
    """Abstract base class for single-parameter prior distributions.

    Subclasses must implement :meth:`log_prob`.  They may override
    :meth:`support`, :attr:`is_proper`, and :meth:`sample` as appropriate.

    Notes
    -----
    Priors hold only the shape and values of the distribution
    (e.g. ``Gaussian.mu``, ``Gaussian.sigma``).  They do **not** carry the
    name of the parameter they apply to; the user-facing prior dictionary
    (``dict[str, Prior]``) is what associates names with shapes.

    The :attr:`is_proper` flag distinguishes priors that integrate to a
    finite value (and so admit sampling, evidence calculations, etc.)
    from improper priors like :class:`ImproperPrior`, which can be used
    for analytic marginalization or as MCMC priors but break operations
    requiring normalization.
    """

    @property
    def is_proper(self) -> bool:
        """Whether the prior integrates to a finite value.

        Defaults to ``True``.  Override in subclasses that represent
        improper priors.
        """
        return True

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        """Evaluate ``log p(x)`` (proper log-density, including any constant).

        Parameters
        ----------
        x
            Parameter value(s) at which to evaluate.  Scalar or array.

        Returns
        -------
        Float Array
            Log-density, with the same shape as ``x``.  Returns ``-inf``
            outside the prior's support.
        """
        raise NotImplementedError(f"{type(self).__name__}.log_prob not implemented")

    def support(self) -> Tuple[float, float]:
        """Return the closed interval ``(low, high)`` over which ``log_prob`` is finite.

        Defaults to ``(-inf, +inf)``.  Subclasses with bounded support
        (e.g. :class:`Uniform`) should override.
        """
        return (-jnp.inf, jnp.inf)

    def sample(
        self,
        key: Array,
        shape: Tuple[int, ...] = (),
    ) -> Float[Array, "..."]:
        """Draw a sample from the prior.

        Parameters
        ----------
        key
            JAX PRNG key.
        shape
            Output shape.  Defaults to scalar.

        Returns
        -------
        Float Array
            Sample(s) from the prior.

        Raises
        ------
        NotImplementedError
            If the prior is improper (cannot sample from a non-normalizable
            distribution) or if no ``sample`` implementation is provided
            by the subclass.
        """
        if not self.is_proper:
            raise NotImplementedError(
                f"Cannot sample from improper prior {type(self).__name__}"
            )
        raise NotImplementedError(f"{type(self).__name__}.sample not implemented")

    def __repr__(self) -> str:
        # equinox.Module's default repr is verbose; keep it concise here so
        # `validate_priors` and similar diagnostics produce readable output.
        cls = type(self).__name__
        fields = ", ".join(
            f"{name}={getattr(self, name)}" for name in self.__dataclass_fields__
        )
        return f"{cls}({fields})"


# ---------------------------------------------------------------------------
# Uniform
# ---------------------------------------------------------------------------


class Uniform(Prior):
    """Uniform prior on the closed interval ``[low, high]``.

    Parameters
    ----------
    low, high : float
        Lower and upper bounds.  Must satisfy ``low < high``.

    Notes
    -----
    ``log_prob(x) = -log(high - low)`` inside the interval, ``-inf``
    outside.  The constant ``-log(high - low)`` is included so that the
    returned value is the proper log-density (matters for evidence and
    cross-prior comparisons; cancels out in MCMC ratios).

    Examples
    --------
    >>> p = Uniform(0.1, 10.0)                      # standard EFAC range
    >>> p.log_prob(jnp.array(1.0))                  # = -log(9.9)
    """

    low: float = eqx.field(static=True)
    high: float = eqx.field(static=True)

    def __post_init__(self):
        if self.low >= self.high:
            raise ValueError(
                f"Uniform requires low < high; got low={self.low}, high={self.high}"
            )

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        x = jnp.asarray(x)
        in_support = (x >= self.low) & (x <= self.high)
        log_density = -jnp.log(self.high - self.low)
        return jnp.where(in_support, log_density, -jnp.inf)

    def support(self) -> Tuple[float, float]:
        return (self.low, self.high)

    def sample(
        self,
        key: Array,
        shape: Tuple[int, ...] = (),
    ) -> Float[Array, "..."]:
        return jr.uniform(key, shape=shape, minval=self.low, maxval=self.high)


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class Gaussian(Prior):
    """Gaussian (normal) prior :math:`\\mathcal{N}(\\mu, \\sigma^2)`.

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    sigma : float
        Standard deviation.  Must be strictly positive.

    Notes
    -----
    ``log_prob(x) = -0.5 ((x - mu) / sigma)**2 - 0.5 log(2 pi) - log(sigma)``.
    Includes the full normalization, so the returned value is a proper
    log-density.

    Used for informative priors derived from independent measurements
    (par-file fits, VLBI distances, etc.).  When marginalized analytically
    inside ``jaxpint.bayes.marginal``, ``sigma**2`` enters
    the Woodbury regularizer in place of the improper ``1e40``.

    Examples
    --------
    >>> # PX prior from par-file fit values
    >>> p = Gaussian(mu=0.973, sigma=0.20)
    >>> p.log_prob(jnp.array(1.0))
    """

    mu: float = eqx.field(static=True)
    sigma: float = eqx.field(static=True)

    def __post_init__(self):
        if self.sigma <= 0:
            raise ValueError(f"Gaussian requires sigma > 0; got sigma={self.sigma}")

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        x = jnp.asarray(x)
        z = (x - self.mu) / self.sigma
        return -0.5 * z * z - 0.5 * jnp.log(2.0 * jnp.pi) - jnp.log(self.sigma)

    def sample(
        self,
        key: Array,
        shape: Tuple[int, ...] = (),
    ) -> Float[Array, "..."]:
        return self.mu + self.sigma * jr.normal(key, shape=shape)


# ---------------------------------------------------------------------------
# Improper (flat) prior
# ---------------------------------------------------------------------------


class ImproperPrior(Prior):
    """Improper flat prior over the entire real line.

    The "discovery-equivalent" prior.  Has no normalization (does not
    integrate to a finite value), so :meth:`Prior.sample` raises and any
    operation requiring normalization (evidence, predictive sampling)
    should reject it.  :meth:`log_prob` returns ``0.0`` everywhere — the
    constant is arbitrary and irrelevant for MCMC ratios, gradients, and
    MAP optimization.

    Use this for parameters that should be analytically marginalized
    with no prior information (the standard treatment of timing-model
    parameters in NANOGrav workflows).  Inside the
    ``jaxpint.bayes.marginal`` functions, an :class:`ImproperPrior`
    triggers the ``Phi = 1e40`` Woodbury regularizer that produces the
    discovery-equivalent flat-prior projection.

    Notes
    -----
    A bounded improper prior (uniform over a finite range with
    unspecified normalization) is just a :class:`Uniform` — that case is
    not represented here.  :class:`ImproperPrior` is specifically the
    over-all-reals flat prior.
    """

    @property
    def is_proper(self) -> bool:
        return False

    def log_prob(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        x = jnp.asarray(x)
        return jnp.zeros_like(x)
