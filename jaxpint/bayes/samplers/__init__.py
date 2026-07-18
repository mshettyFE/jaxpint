"""NumPyro sampler integration for JaxPINT (opt-in).

Install the
``sampling`` extra (``pip install 'jaxpint[sampling]'``) to use it.  The core
likelihood and analytic-marginalization path (``jaxpint.bayes``) does not need
it.

Two layers:

- **Prior assembly** (``priors``) -- the framework-agnostic composition helpers
  that build a ``PriorSpec`` of ``{fqn: numpyro Distribution}``.
- **Model + runner** (``numpyro``) -- the NumPyro model wrapping a
  (marginalized) JaxPINT likelihood via ``numpyro.factor`` and the NUTS driver.
"""

from __future__ import annotations

from jaxpint.bayes.samplers.numpyro import (
    build_pta_clogL_model,
    build_pta_model,
    build_single_pulsar_model,
    run_nuts,
)
from jaxpint.bayes.samplers.priors import (
    PRIOR_DEFAULTS,
    PriorResolutionError,
    PriorSpec,
    PulsarBundle,
    collect_free_fqns,
    cw_phi_psr_priors,
    cw_priors,
    distance_priors,
    free_spectrum_priors,
    from_par_file,
    noise_priors_simple,
    resolve_priors,
    timing_marg_set,
)

__all__ = [
    # Prior assembly
    "PriorSpec",
    "PulsarBundle",
    "PriorResolutionError",
    "PRIOR_DEFAULTS",
    "noise_priors_simple",
    "free_spectrum_priors",
    "distance_priors",
    "from_par_file",
    "cw_priors",
    "cw_phi_psr_priors",
    "timing_marg_set",
    "resolve_priors",
    "collect_free_fqns",
    # Model + runner
    "build_single_pulsar_model",
    "build_pta_model",
    "build_pta_clogL_model",
    "run_nuts",
]
