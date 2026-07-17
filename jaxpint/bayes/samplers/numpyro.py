"""NumPyro model builders + NUTS runner for JaxPINT likelihoods.

Flow::

    from jaxpint.bayes import marginalize_pta
    from jaxpint.bayes.samplers import (
        noise_priors_simple, cw_priors, timing_marg_set, resolve_priors,
        collect_free_fqns,
    )
    from jaxpint.bayes.samplers.numpyro import build_pta_model, run_nuts

    g, _, skels = marginalize_pta(over=timing_marg_set(psrs), config=..., ...)
    spec   = noise_priors_simple(psrs) | cw_priors()
    priors = resolve_priors(collect_free_fqns(names, skels, gp), spec)
    model, init = build_pta_model(g, priors, skels, gp, names)
    idata, mcmc = run_nuts(model, init=init, key=key)

**Packing bridge:** NumPyro samples a ``{site_name: scalar}`` dict; the
likelihood wants ``ParameterVector`` / ``GlobalParams``.  Because every slot is
scalar and ``free_names()`` ↔ ``with_free_values(...)`` are exact inverses in
the same order, reconstruction is a stack + ``with_free_values``.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Optional, Union, overload

import jax.numpy as jnp
from jax.typing import ArrayLike
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

from jaxpint.bayes.samplers.priors import PriorResolutionError
from jaxpint.types import GlobalParams, ParameterVector


__all__ = [
    "build_single_pulsar_model",
    "build_pta_model",
    "run_nuts",
]


# ---------------------------------------------------------------------------
# Packing bridge  (sampled {name: scalar}  ->  ParameterVector / GlobalParams)
# ---------------------------------------------------------------------------


def _pack_free(
    skeleton: ParameterVector, sampled: Mapping[str, ArrayLike], prefix: str = ""
) -> ParameterVector:
    """Reconstruct a ParameterVector from sampled free-parameter values.

    ``sampled`` is keyed by site name (``f"{prefix}_{bare}"`` for a PTA pulsar,
    bare name for a single pulsar).  Marginalized/frozen slots keep their
    skeleton (fiducial) values.
    """
    free = skeleton.free_names()
    if not free:
        return skeleton
    key = (lambda b: f"{prefix}_{b}") if prefix else (lambda b: b)
    vec = jnp.stack([sampled[key(b)] for b in free])
    return skeleton.with_free_values(vec)


def _pack_global(
    skeleton: GlobalParams, sampled: Mapping[str, ArrayLike]
) -> GlobalParams:
    """Reconstruct GlobalParams from sampled values (all global names sampled)."""
    if not skeleton.names:
        return skeleton
    vec = jnp.stack([sampled[n] for n in skeleton.names])
    return skeleton.with_values(vec)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def build_single_pulsar_model(
    likelihood: Callable[[ParameterVector], jnp.ndarray],
    priors: Mapping[str, dist.Distribution],
    skeleton: ParameterVector,
) -> tuple[Callable[[], None], dict[str, jnp.ndarray]]:
    """Build a NumPyro model for a single (marginalized) pulsar likelihood.

    Sites are the bare free-parameter names of ``skeleton``; ``priors`` must
    contain a distribution for each.  Returns ``(model, init_values)`` where
    ``init_values`` seeds the sampler at the skeleton's fiducial ``y_fid``.
    """
    free = skeleton.free_names()
    _check_priors(free, priors, label="single-pulsar")

    def model():
        sampled = {b: numpyro.sample(b, priors[b]) for b in free}
        params = _pack_free(skeleton, sampled)
        numpyro.factor("logL", likelihood(params))

    init = dict(zip(free, [v for v in skeleton.free_values()]))
    return model, init


def build_pta_model(
    likelihood: Callable[[GlobalParams, tuple[ParameterVector, ...]], jnp.ndarray],
    priors: Mapping[str, dist.Distribution],
    reduced_skeletons: tuple[ParameterVector, ...],
    global_skeleton: GlobalParams,
    pulsar_names: tuple[str, ...],
) -> tuple[Callable[[], None], dict[str, jnp.ndarray]]:
    """Build a NumPyro model for a (marginalized) PTA likelihood.

    Per-pulsar sites are ``f"{pulsar_name}_{bare}"`` for each pulsar's free
    parameters; global sites are the names in ``global_skeleton``.  ``priors``
    must cover every site.  Returns ``(model, init_values)`` seeding the sampler
    at the fiducial ``y_fid``.
    """
    site_names: list[str] = []
    init: dict[str, jnp.ndarray] = {}
    for prefix, skel in zip(pulsar_names, reduced_skeletons):
        for b, v in zip(skel.free_names(), skel.free_values()):
            site_names.append(f"{prefix}_{b}")
            init[f"{prefix}_{b}"] = v
    for n, v in zip(global_skeleton.names, global_skeleton.values):
        site_names.append(n)
        init[n] = v
    _check_priors(site_names, priors, label="PTA")

    def model():
        sampled: dict[str, ArrayLike] = {}
        pulsar_params = []
        for prefix, skel in zip(pulsar_names, reduced_skeletons):
            for b in skel.free_names():
                site = f"{prefix}_{b}"
                sampled[site] = numpyro.sample(site, priors[site])
            pulsar_params.append(_pack_free(skel, sampled, prefix))
        for n in global_skeleton.names:
            sampled[n] = numpyro.sample(n, priors[n])
        gp = _pack_global(global_skeleton, sampled)
        numpyro.factor("logL", likelihood(gp, tuple(pulsar_params)))

    return model, init


def _check_priors(site_names, priors: Mapping[str, dist.Distribution], *, label: str):
    # Same "no silent prior" contract as resolve_priors, enforced here because
    # build_*_model is public and may be handed a prior mapping that never went
    # through resolve_priors. Raise the *same* exception type so a caller's
    # `except PriorResolutionError` covers the gap wherever it is first caught.
    missing = [s for s in site_names if s not in priors]
    if missing:
        raise PriorResolutionError(
            f"build_{label}_model: no prior distribution for sampled site(s) "
            f"{sorted(missing)}. Resolve priors covering every free parameter "
            f"(see jaxpint.bayes.samplers.resolve_priors)."
        )


# ---------------------------------------------------------------------------
# NUTS runner
# ---------------------------------------------------------------------------


@overload
def run_nuts(
    model: Callable[[], None],
    *,
    init: Optional[Mapping[str, ArrayLike]] = ...,
    key,
    num_warmup: int = ...,
    num_samples: int = ...,
    num_chains: int = ...,
    max_tree_depth: int = ...,
    target_accept_prob: float = ...,
    dense_mass: bool = ...,
    chain_method: str = ...,
    progress_bar: bool = ...,
    extra_fields: tuple[str, ...] = ...,
    return_arviz: Literal[True] = ...,
) -> tuple[Any, MCMC]: ...
@overload
def run_nuts(
    model: Callable[[], None],
    *,
    init: Optional[Mapping[str, ArrayLike]] = ...,
    key,
    num_warmup: int = ...,
    num_samples: int = ...,
    num_chains: int = ...,
    max_tree_depth: int = ...,
    target_accept_prob: float = ...,
    dense_mass: bool = ...,
    chain_method: str = ...,
    progress_bar: bool = ...,
    extra_fields: tuple[str, ...] = ...,
    return_arviz: Literal[False],
) -> MCMC: ...
def run_nuts(
    model: Callable[[], None],
    *,
    init: Optional[Mapping[str, ArrayLike]] = None,
    key,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    max_tree_depth: int = 8,
    target_accept_prob: float = 0.8,
    dense_mass: bool = False,
    chain_method: str = "vectorized",
    progress_bar: bool = True,
    extra_fields: tuple[str, ...] = (),
    return_arviz: bool = True,
) -> Union[tuple[Any, MCMC], MCMC]:
    """Run NUTS on a JaxPINT NumPyro ``model``, initialized at the fiducial.

    ``init`` (from the model builder) seeds every chain at the marginalization
    linearization point ``y_fid`` — critical for stiff PTA posteriors; without
    it NUTS starts from the prior and rarely mixes.

    ``extra_fields`` is forwarded to ``MCMC.run`` -- pass e.g.
    ``("accept_prob", "num_steps", "diverging")`` to recover per-sample NUTS
    diagnostics afterwards via ``mcmc.get_extra_fields()``.

    Returns ``(idata, mcmc)`` when ``return_arviz`` (default), else the ``mcmc``
    object.  ``idata`` is an ArviZ ``InferenceData`` (r-hat / ESS / divergences
    via ``arviz.summary``); site values are labeled by their FQN.
    """
    numpyro.enable_x64()  # redundant if jaxpint already set it; harmless

    init_strategy = init_to_value(values=dict(init)) if init else None
    kernel = NUTS(
        model,
        init_strategy=init_strategy,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
    )
    mcmc.run(key, extra_fields=extra_fields)

    if not return_arviz:
        return mcmc
    import arviz as az

    return az.from_numpyro(mcmc), mcmc
