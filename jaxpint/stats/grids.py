"""Numerical (grid) marginalization of a nuisance from a tabulated likelihood.

The *numerical* counterpart of :mod:`jaxpint.bayes.marginal` (the analytic Woodbury
marginalization of timing parameters).  Reach for these when the log-likelihood is
**not quadratic** in the nuisance, so the Gaussian/Woodbury closed form does not
apply -- e.g. a continuous-wave source's pulsar-term phase, where the nuisance
enters through ``cos delta / sin delta`` (see :mod:`jaxpint.pta.incoherent_ul`).  The
likelihood is evaluated on a grid of the nuisance and then reduced: **integrate**
it (the Bayesian marginal, :func:`grid_log_marginal`) or **maximize** it (the
frequentist profile, :func:`grid_log_profile`).

These are deliberately prior-free: the marginalization measure is supplied as
explicit log quadrature weights, not a prior distribution.  (Prior specification
and sampling live in ``jaxpint.bayes.samplers``; analytic marginalization
targets are a plain ``set[str]`` passed to :func:`jaxpint.bayes.marginalize_pta`.)
"""

from __future__ import annotations

from typing import Optional

import jax.numpy as jnp
from jax.scipy.special import logsumexp
from jaxtyping import Array, Float

__all__ = ["grid_log_marginal", "grid_log_profile"]


def grid_log_marginal(
    log_like: Float[Array, "... n"],
    log_weights: Optional[Float[Array, "... n"]] = None,
) -> Float[Array, "..."]:
    r"""Log-marginal of a log-likelihood tabulated on a nuisance grid.

    Returns the log-integral over the nuisance,

    .. math::
        \log \int L(\theta)\,\pi(\theta)\,\mathrm{d}\theta
        \;\approx\; \operatorname{logsumexp}_k\!\big(\ell_k + w_k\big),

    reducing over the **last** axis (a leading batch/pulsar axis broadcasts).

    Parameters
    ----------
    log_like : (..., n) array
        Log-likelihood evaluated at each of the ``n`` nuisance grid points.
    log_weights : (..., n) array, optional
        Log of the quadrature×prior measure at each grid point,
        ``log(π(θ_k)·Δθ_k)`` (e.g. trapezoidal weights times a prior density).
        ``None`` (default) is the flat case -- a uniform prior on a uniform grid --
        and reduces to the average ``logsumexp(ℓ) − log n``.

    Notes
    -----
    The numerical counterpart of :func:`jaxpint.bayes.marginalize_single_pulsar`:
    use it when ``L`` is not quadratic in the nuisance (no Gaussian closed form).
    For a quadratic ``L`` and a (improper-)uniform / Gaussian prior the closed-form
    analytic marginalization is exact and cheaper -- prefer it there.
    """
    if log_weights is None:
        n = log_like.shape[-1]
        return logsumexp(log_like, axis=-1) - jnp.log(n)
    return logsumexp(log_like + log_weights, axis=-1)


def grid_log_profile(
    log_like: Float[Array, "... n"],
) -> Float[Array, "..."]:
    r"""Profile (max over the nuisance grid) of a tabulated log-likelihood.

    The frequentist twin of :func:`grid_log_marginal` -- maximize the nuisance
    instead of integrating it -- reducing over the **last** axis.  Sharper but
    alias-prone (it locks onto the single best-fitting grid point); for
    localization it is a diagnostic, not the credible-region map.
    """
    return jnp.max(log_like, axis=-1)
