"""Arm-neutral statistical numerics shared by both inference arms.

This is the layer *below* :mod:`jaxpint.bayes` and ``jaxpint.frequentist`` (and
below :mod:`jaxpint.pta`, whose CW products consume it): grid reductions and
distribution/region primitives that carry no prior, no likelihood, and no
model — only jax.  The same math frequently serves both vocabularies (a
Gaussian ellipse area is a Bayesian *credible* region under a flat prior and a
frequentist *confidence* region from a Fisher matrix; a grid ``logsumexp`` is a
Bayesian marginal while the matching ``max`` is a frequentist profile), which
is exactly why it lives here rather than in either arm.

Keeping this package a **leaf** (no jaxpint imports) is load-bearing: it is
what lets ``bayes/`` and ``frequentist/`` both sit above ``pta/`` without any
circular package dependency.
"""

from __future__ import annotations

from jaxpint.stats.grids import (
    grid_log_marginal,
    grid_log_profile,
)
from jaxpint.stats.regions import (
    credible_level_map,
    credible_region_area,
    gaussian_credible_area,
    grid_credible_upper_limit,
    mixture_truncated_gaussian_upper_limit,
    truncated_gaussian_upper_limit,
)

__all__ = [
    # Grid reductions (marginal = Bayesian integrate, profile = frequentist max)
    "grid_log_marginal",
    "grid_log_profile",
    # Upper limits from tabulated / (mixture-)Gaussian posteriors
    "truncated_gaussian_upper_limit",
    "mixture_truncated_gaussian_upper_limit",
    "grid_credible_upper_limit",
    # Credible / confidence regions
    "gaussian_credible_area",
    "credible_level_map",
    "credible_region_area",
]
