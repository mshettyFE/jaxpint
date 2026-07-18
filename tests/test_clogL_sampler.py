"""Coefficient-sampling hook: numpyro model over explicit GP coefficients.

``build_pta_clogL_model`` adds one flat-prior coefficient site and a
``pta_clogL`` factor so NUTS samples the correlated-signal Fourier
coefficients as explicit latents.  Tests here pin the *wiring* (the model's
log density is exactly ``clogL + hyperprior``, i.e. the improper coefficient
site contributes zero and its Gaussian prior comes only from the factor) and
smoke-run the full NUTS path; they do not assert convergence quality.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("numpyro")
import numpyro.distributions as dist
from numpyro.infer.util import log_density

from jaxpint.bayes.samplers import (
    build_pta_clogL_model,
    collect_free_fqns,
    make_conditional_gibbs_fn,
    run_clogL_gibbs,
    run_nuts,
)
from jaxpint.pta import conditional_gwb, pta_clogL, sample_conditional

from tests.test_conditional import _hd_config

PULSAR_NAMES = ("P0", "P1")


def _fiducial_sites(pulsar_names, reduced_skeletons, global_skeleton):
    """{site_name: fiducial_value} for every hyperparameter site."""
    fid = {}
    for prefix, skel in zip(pulsar_names, reduced_skeletons):
        for b, v in zip(skel.free_names(), skel.free_values()):
            fid[f"{prefix}_{b}"] = v
    for n, v in zip(global_skeleton.names, global_skeleton.values):
        fid[n] = v
    return fid


def _build(config, gp, pps):
    """Build the clogL model seeded at the conditional mean, with Normal priors."""
    fid = _fiducial_sites(PULSAR_NAMES, pps, gp)
    priors = {
        s: dist.Normal(fid[s], 1.0) for s in collect_free_fqns(PULSAR_NAMES, pps, gp)
    }
    cond_mean = conditional_gwb(gp, pps, config).mean
    clogL = lambda g, pp, c: pta_clogL(g, pp, config, c)  # noqa: E731
    model, init = build_pta_clogL_model(clogL, priors, pps, gp, PULSAR_NAMES, cond_mean)
    return model, init, priors, cond_mean


# ---------------------------------------------------------------------------
# Wiring: model log density == clogL + hyperprior (improper coeff site is zero)
# ---------------------------------------------------------------------------


def test_model_log_density_is_clogL_plus_hyperprior():
    gp, pps, config, _ = _hd_config()
    model, init, priors, cond_mean = _build(config, gp, pps)

    # init carries every latent: hyperparams at fiducial + coeffs at the mean,
    # whose length is set by coefficient_init.
    assert init["gwb_coefficients"].shape == cond_mean.shape
    lp, _ = log_density(model, (), {}, init)

    # Expected: the clogL factor (data + coeff prior) plus each hyperparameter's
    # Normal log-prob; the coefficient site is improper-uniform (contributes 0).
    expected = float(pta_clogL(gp, pps, config, init["gwb_coefficients"]))
    for s, d in priors.items():
        expected += float(d.log_prob(init[s]))
    npt.assert_allclose(float(lp), expected, rtol=1e-9)


def test_flat_coeff_prior_not_double_counted():
    """Shifting only the coefficients changes the model logp by exactly Δ(clogL)."""
    gp, pps, config, _ = _hd_config()
    model, init, _, _ = _build(config, gp, pps)

    bumped = dict(init)
    bumped["gwb_coefficients"] = init["gwb_coefficients"] + 0.3
    lp0, _ = log_density(model, (), {}, init)
    lp1, _ = log_density(model, (), {}, bumped)

    dclogL = float(
        pta_clogL(gp, pps, config, bumped["gwb_coefficients"])
        - pta_clogL(gp, pps, config, init["gwb_coefficients"])
    )
    npt.assert_allclose(float(lp1) - float(lp0), dclogL, rtol=1e-9)


# ---------------------------------------------------------------------------
# End-to-end NUTS smoke
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_pta_clogL_nuts_smoke():
    gp, pps, config, _ = _hd_config()
    model, init, _, cond_mean = _build(config, gp, pps)

    mcmc = run_nuts(
        model,
        init=init,
        key=jax.random.PRNGKey(0),
        num_warmup=20,
        num_samples=20,
        num_chains=1,
        progress_bar=False,
        return_arviz=False,
    )
    draws = mcmc.get_samples()["gwb_coefficients"]
    assert draws.shape == (20, cond_mean.shape[0])
    assert bool(jnp.all(jnp.isfinite(draws)))


# ---------------------------------------------------------------------------
# Exact-Gibbs (HMC-within-Gibbs) coefficient draws
# ---------------------------------------------------------------------------


def test_gibbs_fn_is_exact_conditional_draw():
    """The Gibbs update == an exact sample_conditional draw at the same theta."""
    gp, pps, config, _ = _hd_config()
    gibbs_fn = make_conditional_gibbs_fn(config, pps, gp, PULSAR_NAMES)

    # HMC sites at the fiducial theta must repack to exactly (gp, pps), so the
    # Gibbs draw must equal the direct conditional draw with the same key.
    fid = _fiducial_sites(PULSAR_NAMES, pps, gp)
    key = jax.random.PRNGKey(0)
    out = gibbs_fn(key, {}, fid)
    expected = sample_conditional(key, conditional_gwb(gp, pps, config))

    assert out["gwb_coefficients"].shape == expected.shape
    npt.assert_allclose(
        np.asarray(out["gwb_coefficients"]), np.asarray(expected), rtol=1e-12
    )


@pytest.mark.slow
def test_clogL_gibbs_end_to_end():
    gp, pps, config, _ = _hd_config()
    model, init, _, cond_mean = _build(config, gp, pps)
    gibbs_fn = make_conditional_gibbs_fn(config, pps, gp, PULSAR_NAMES)

    mcmc = run_clogL_gibbs(
        model,
        gibbs_fn,
        init=init,
        key=jax.random.PRNGKey(0),
        num_warmup=20,
        num_samples=20,
        num_chains=1,
        progress_bar=False,
        return_arviz=False,
    )
    samples = mcmc.get_samples()
    draws = samples["gwb_coefficients"]
    assert draws.shape == (20, cond_mean.shape[0])
    assert bool(jnp.all(jnp.isfinite(draws)))
    # The NUTS half actually moved the hyperparameters (Gibbs only touches coeffs).
    assert "gwb_log10_A" in samples and bool(jnp.all(jnp.isfinite(samples["gwb_log10_A"])))
