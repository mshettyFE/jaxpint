"""Bayesian-inference layer for JaxPINT.

Importing from :mod:`jaxpint.bayes` opts in to Bayesian-flavoured machinery — priors,
analytic / numerical marginalization, posterior helpers — that build on
top of those likelihoods.

The presence of ``from jaxpint.bayes import ...`` in user code is meant
to act as a visible flag that the script is making Bayesian assumptions
(e.g., choice of prior shape, marginalization over nuisance parameters).
Pure likelihood scans don't need this subpackage.

"""

from __future__ import annotations

from jaxpint.bayes.defaults import (
    NANOGRAV_NOISE_DEFAULTS,
    collect_param_names,
    cw_phi_psr_priors,
    cw_priors,
    distance_priors,
    from_par_file,
    noise_priors_simple,
    timing_priors,
)
from jaxpint.bayes.marginal import (
    marginalize_pta,
    marginalize_single_pulsar,
)
from jaxpint.bayes.posterior import combine_log_prob, log_prior_sum
from jaxpint.bayes.priors import (
    Gaussian,
    ImproperPrior,
    Prior,
    Uniform,
)
from jaxpint.bayes.validate import (
    PriorValidationError,
    validate_priors,
)


__all__ = [
    # Priors
    "Prior",
    "Uniform",
    "Gaussian",
    "ImproperPrior",
    # Bulk-prior factories
    "NANOGRAV_NOISE_DEFAULTS",
    "timing_priors",
    "distance_priors",
    "from_par_file",
    "cw_priors",
    "cw_phi_psr_priors",
    "noise_priors_simple",
    # Validation
    "PriorValidationError",
    "validate_priors",
    "collect_param_names",
    # Posterior composition
    "log_prior_sum",
    "combine_log_prob",
    # Marginalization (analytic)
    "marginalize_single_pulsar",
    "marginalize_pta",
]
# Grid reductions and credible/confidence-region primitives moved to
# jaxpint.stats (arm-neutral numerics consumed by pta/ and both inference
# arms; keeping them here made bayes <-> pta a circular package dependency).
