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

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.infer import HMCGibbs, MCMC, NUTS, init_to_value

from jaxpint.bayes.samplers.priors import PriorResolutionError
from jaxpint.pta.conditional import conditional_gwb, sample_conditional
from jaxpint.pta.likelihood import PTAConfig, joint_prior_cholesky
from jaxpint.types import GlobalParams, ParameterVector


__all__ = [
    "build_single_pulsar_model",
    "build_pta_model",
    "build_pta_clogL_model",
    "build_pta_clogL_whitened_model",
    "make_conditional_gibbs_fn",
    "run_nuts",
    "run_clogL_gibbs",
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


def build_pta_clogL_model(
    clogL: Callable[
        [GlobalParams, tuple[ParameterVector, ...], ArrayLike], jnp.ndarray
    ],
    priors: Mapping[str, dist.Distribution],
    reduced_skeletons: tuple[ParameterVector, ...],
    global_skeleton: GlobalParams,
    pulsar_names: tuple[str, ...],
    coefficient_init: ArrayLike,
    *,
    coefficient_site: str = "gwb_coefficients",
) -> tuple[Callable[[], None], dict[str, jnp.ndarray]]:
    r"""Build the conjugate clogL model — the substrate for exact-Gibbs sampling.

    A NumPyro model with the hyperparameter sites plus one coefficient site
    carrying a *flat* (improper-uniform) prior, and the full
    :func:`~jaxpint.pta.pta_clogL` (data **and** the built-in Gaussian
    coefficient prior ``N(0, \Phi_\mathrm{joint})``) as the factor.

    **Intended runner: :func:`run_clogL_gibbs`**, whose θ-step needs exactly
    this factor (``p(\theta | a, r) \propto \mathrm{clogL}(\theta, a)\,p(\theta)``,
    which includes the θ-dependent coefficient prior) and whose coefficient
    step is an *exact* Gaussian draw.  You *can* run plain :func:`run_nuts` on
    this model, but the coefficients then have a Neal's-funnel geometry — for
    an all-HMC single kernel prefer the non-centered
    :func:`build_pta_clogL_whitened_model` instead.

    Parameters
    ----------
    clogL
        Closure ``(global_params, pulsar_params, coefficients) -> scalar`` —
        typically ``lambda g, pp, c: pta_clogL(g, pp, config, c)``.  The
        coefficient argument must be in the ``(k, p, b)`` layout of
        :func:`~jaxpint.pta.conditional.conditional_gwb`.
    priors, reduced_skeletons, global_skeleton, pulsar_names
        As for :func:`build_pta_model`.  ``priors`` must cover every
        hyperparameter site (not the coefficient site).
    coefficient_init : (n_joint,) array
        Initial coefficient values; also fixes the coefficient site's length.
        Use ``conditional_gwb(...).mean`` — the exact conditional posterior
        mean at the fiducial hyperparameters, which both seeds the sampler
        (analogous to the hyperparameters' fiducial ``y_fid``) and defines the
        coefficient dimension.  A flat coefficient prior with no informed start
        mixes poorly, so this is required rather than optional.
    coefficient_site : str
        Name of the coefficient sample site (default ``"gwb_coefficients"``).

    Returns
    -------
    (model, init) : tuple
        As for :func:`build_pta_model`; ``init`` additionally seeds
        ``coefficient_site`` at ``coefficient_init``.
    """
    coefficient_init = jnp.asarray(coefficient_init)
    n_coefficients = coefficient_init.shape[0]

    site_names: list[str] = []
    init: dict[str, jnp.ndarray] = {}
    for prefix, skel in zip(pulsar_names, reduced_skeletons):
        for b, v in zip(skel.free_names(), skel.free_values()):
            site_names.append(f"{prefix}_{b}")
            init[f"{prefix}_{b}"] = v
    for n, v in zip(global_skeleton.names, global_skeleton.values):
        site_names.append(n)
        init[n] = v
    # Coefficients use a flat prior, so they are NOT in site_names / priors.
    _check_priors(site_names, priors, label="PTA clogL")
    init[coefficient_site] = coefficient_init

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
        # Flat prior over the n_joint coefficients; their real Gaussian prior
        # is folded into the clogL factor below (do not double-count it).
        coeffs = numpyro.sample(
            coefficient_site,
            dist.ImproperUniform(constraints.real, (), (n_coefficients,)),
        )
        numpyro.factor("clogL", clogL(gp, tuple(pulsar_params), coeffs))

    return model, init


def build_pta_clogL_whitened_model(
    clogL_data: Callable[
        [GlobalParams, tuple[ParameterVector, ...], ArrayLike], jnp.ndarray
    ],
    config: PTAConfig,
    priors: Mapping[str, dist.Distribution],
    reduced_skeletons: tuple[ParameterVector, ...],
    global_skeleton: GlobalParams,
    pulsar_names: tuple[str, ...],
    coefficient_init: ArrayLike,
    *,
    coefficient_site: str = "gwb_coefficients",
    whitened_site: str = "gwb_coefficients_white",
) -> tuple[Callable[[], None], dict[str, jnp.ndarray]]:
    r"""Build a whitened (non-centered) joint-NUTS model over the GP coefficients.

    The reparameterized counterpart of :func:`build_pta_clogL_model` for the
    **fully joint-NUTS** path.  Instead of sampling the coefficients ``a``
    directly under their ``\theta``-dependent Gaussian prior — a
    Neal's-funnel geometry that cripples HMC — it samples an isotropic
    ``z ~ N(0, I)`` and sets ``a = L(\theta)\, z`` with ``L`` the Cholesky of
    ``\Phi_\mathrm{joint}(\theta)`` (:func:`~jaxpint.pta.joint_prior_cholesky`).
    The likelihood factor is the *data-only*
    :func:`~jaxpint.pta.pta_clogL_data` (the Gaussian coefficient prior is
    supplied by ``z``'s isotropic prior through the whitening map, so it must
    **not** also be in the factor).  ``a`` is recorded as a
    ``numpyro.deterministic`` under ``coefficient_site``.

    This targets exactly the same ``(\theta, a)`` posterior as the conjugate
    :func:`build_pta_clogL_model` (up to the change-of-variables Jacobian),
    but with a geometry NUTS can traverse.  Prefer the exact-Gibbs
    :func:`run_clogL_gibbs` when it applies; reach for this when you need a
    single all-HMC kernel, or as the base for a non-Gaussian coefficient
    prior via :func:`~jaxpint.pta.pta_clogL_data`.

    Parameters
    ----------
    clogL_data
        Closure ``(global_params, pulsar_params, coefficients) -> scalar`` —
        typically ``lambda g, pp, a: pta_clogL_data(g, pp, config, a)``.
    config
        The PTA config whose ``\Phi_\mathrm{joint}(\theta)`` defines the
        whitening map; ``config.correlated_injectors`` must be non-empty.
    priors, reduced_skeletons, global_skeleton, pulsar_names
        As for :func:`build_pta_clogL_model`.
    coefficient_init : (n_joint,) array
        Fiducial coefficients (``conditional_gwb(...).mean``); fixes the
        dimension and seeds the whitened variable at ``z = L(\theta_fid)^{-1}
        \hat a`` so the chain starts at the same physical point as the
        conjugate / Gibbs samplers.
    coefficient_site, whitened_site
        Names of the deterministic coefficient site and the sampled whitened
        site (defaults ``"gwb_coefficients"`` / ``"gwb_coefficients_white"``).

    Returns
    -------
    (model, init) : tuple
        ``init`` seeds the hyperparameters at ``y_fid`` and ``whitened_site``
        at ``L(\theta_fid)^{-1} \hat a``.  Run with :func:`run_nuts`.
    """
    coefficient_init = jnp.asarray(coefficient_init)
    n_coefficients = coefficient_init.shape[0]

    site_names: list[str] = []
    init: dict[str, jnp.ndarray] = {}
    for prefix, skel in zip(pulsar_names, reduced_skeletons):
        for b, v in zip(skel.free_names(), skel.free_values()):
            site_names.append(f"{prefix}_{b}")
            init[f"{prefix}_{b}"] = v
    for n, v in zip(global_skeleton.names, global_skeleton.values):
        site_names.append(n)
        init[n] = v
    # Whitened variable uses an isotropic N(0, I) prior — not in priors.
    _check_priors(site_names, priors, label="PTA clogL whitened")

    # Seed z at L(theta_fid)^{-1} a_hat so the chain starts where the conjugate
    # and Gibbs samplers do (the conditional mean), not at a = 0.
    L_fid = joint_prior_cholesky(global_skeleton, config)
    init[whitened_site] = jax.scipy.linalg.solve_triangular(
        L_fid, coefficient_init, lower=True
    )

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
        # Non-centered: isotropic z, then a = L(theta) z realizes a ~ N(0, Phi_joint).
        z = numpyro.sample(
            whitened_site, dist.Normal(0.0, 1.0).expand([n_coefficients]).to_event(1)
        )
        L = joint_prior_cholesky(gp, config)
        coeffs = numpyro.deterministic(coefficient_site, L @ z)
        numpyro.factor("clogL_data", clogL_data(gp, tuple(pulsar_params), coeffs))

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
# Runners
# ---------------------------------------------------------------------------


def _drive_mcmc(
    kernel,
    *,
    key,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    chain_method: str,
    progress_bar: bool,
    extra_fields: tuple[str, ...],
    return_arviz: bool,
) -> Union[tuple[Any, MCMC], MCMC]:
    """Run an MCMC ``kernel`` and (optionally) hand back an ArviZ ``InferenceData``.

    The shared tail of :func:`run_nuts` and :func:`run_clogL_gibbs`: build
    ``MCMC``, run it, and convert to ArviZ unless ``return_arviz`` is False.
    """
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
    return _drive_mcmc(
        kernel,
        key=key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
        extra_fields=extra_fields,
        return_arviz=return_arviz,
    )


def make_conditional_gibbs_fn(
    config: PTAConfig,
    reduced_skeletons: tuple[ParameterVector, ...],
    global_skeleton: GlobalParams,
    pulsar_names: tuple[str, ...],
    *,
    coefficient_site: str = "gwb_coefficients",
) -> Callable[[Any, Mapping, Mapping], dict]:
    r"""Build an exact-Gibbs update for the GWB coefficients.

    The coefficient conditional ``a | \theta, r`` is *exactly* Gaussian —
    ``N(\hat a(\theta), \Sigma(\theta))`` = :func:`~jaxpint.pta.conditional_gwb`
    — so it can be drawn directly with :func:`~jaxpint.pta.sample_conditional`
    rather than explored by HMC.  This returns the ``gibbs_fn`` that
    :func:`run_clogL_gibbs` (via NumPyro's ``HMCGibbs``) calls each sweep:
    reconstruct ``\theta`` from the HMC sites, form the conditional at that
    ``\theta``, and return one exact coefficient draw.  Pairing it with NUTS on
    ``\theta`` gives a rejection-free coefficient step and typically far better
    mixing than the fully-joint NUTS of :func:`build_pta_clogL_model` alone.

    Parameters
    ----------
    config
        The same :class:`~jaxpint.pta.PTAConfig` the model's ``clogL`` closes
        over; ``config.correlated_injectors`` must be non-empty.
    reduced_skeletons, global_skeleton, pulsar_names
        As passed to :func:`build_pta_clogL_model` — used to repack the sampled
        HMC-site scalars back into ``ParameterVector`` / ``GlobalParams``.
    coefficient_site
        The coefficient site name (must match the model's, default
        ``"gwb_coefficients"``).

    Returns
    -------
    gibbs_fn : callable
        ``(rng_key, gibbs_sites, hmc_sites) -> {coefficient_site: draw}``, the
        signature NumPyro's ``HMCGibbs`` expects.
    """

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        pulsar_params = tuple(
            _pack_free(skel, hmc_sites, prefix)
            for prefix, skel in zip(pulsar_names, reduced_skeletons)
        )
        gp = _pack_global(global_skeleton, hmc_sites)
        cond = conditional_gwb(gp, pulsar_params, config)
        return {coefficient_site: sample_conditional(rng_key, cond)}

    return gibbs_fn


def run_clogL_gibbs(
    model: Callable[[], None],
    gibbs_fn: Callable[[Any, Mapping, Mapping], dict],
    *,
    gibbs_sites: tuple[str, ...] = ("gwb_coefficients",),
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
    """Run HMC-within-Gibbs: exact coefficient draws + NUTS on the hyperparameters.

    ``model`` and ``init`` are the outputs of :func:`build_pta_clogL_model`;
    ``init`` seeds both the hyperparameters (at ``y_fid``) and the coefficient
    site (at ``conditional_gwb(...).mean``).
    """
    numpyro.enable_x64()

    init_strategy = init_to_value(values=dict(init)) if init else None
    inner = NUTS(
        model,
        init_strategy=init_strategy,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
        dense_mass=dense_mass,
    )
    kernel = HMCGibbs(inner, gibbs_fn=gibbs_fn, gibbs_sites=list(gibbs_sites))
    return _drive_mcmc(
        kernel,
        key=key,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
        progress_bar=progress_bar,
        extra_fields=extra_fields,
        return_arviz=return_arviz,
    )
