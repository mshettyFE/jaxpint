"""Bayesian-inference layer for JaxPINT.

Importing from :mod:`jaxpint.bayes` opts in to Bayesian-flavoured machinery —
analytic marginalization of nuisance (timing-model) parameters that builds on
top of the likelihoods.

The presence of ``from jaxpint.bayes import ...`` in user code is meant
to act as a visible flag that the script is making Bayesian assumptions
(here: analytic marginalization over nuisance parameters).  Pure likelihood
scans don't need this subpackage.

Analytic Bayesian CW upper limits live in the ``jaxpint.bayes.cw_upper_limit``
(strain, closed-form / grid-marginalized) and ``jaxpint.bayes.incoherent_ul``
(distance, incoherent) submodules.  They carry explicit prior assumptions, so
they belong on the Bayesian side rather than in the arm-neutral
``jaxpint.pta``; import them by their full module path.

Prior specification and sampling live under ``jaxpint.bayes.samplers``
(NumPyro, opt-in); that is where prior-distribution objects and the
composition helpers are exposed.

"""

from __future__ import annotations

from jaxpint.bayes.marginal import (
    marginalize_pta,
    marginalize_single_pulsar,
)


__all__ = [
    # Marginalization (analytic)
    "marginalize_single_pulsar",
    "marginalize_pta",
]
# Grid reductions and credible/confidence-region primitives moved to
# jaxpint.stats (arm-neutral numerics consumed by pta/ and both inference
# arms; keeping them here made bayes <-> pta a circular package dependency).
