"""End-to-end smoke test for the NumPyro sampler layer.

Exercises the full plumbing — marginalize → build model → NUTS → ArviZ — on a
synthetic single pulsar.  Asserts the machinery runs and round-trips (finite,
correctly-labeled draws); it does NOT assert convergence quality (50 draws).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("numpyro")
pytest.importorskip("arviz")
import numpyro.distributions as dist

from jaxpint.bayes import marginalize_pta, marginalize_single_pulsar
from jaxpint.bayes.samplers.numpyro import (
    build_pta_model,
    build_single_pulsar_model,
    run_nuts,
)
from jaxpint.bayes.samplers.priors import (
    PriorResolutionError,
    collect_free_fqns,
    resolve_priors,
)

from tests.helpers import make_simple_pulsar


@pytest.mark.slow
def test_single_pulsar_nuts_end_to_end():
    toa_data, tm, nm, params = make_simple_pulsar(150, f0=100.0, f1=-1e-14, seed=3)

    # Marginalize every free timing param except F0; sample F0.
    over = {n for n in params.free_names() if n != "F0"}
    assert over, "expected >1 free timing param so something is marginalized"

    likelihood, marged, skel = marginalize_single_pulsar(
        over=over,
        toa_data=toa_data,
        timing_model=tm,
        noise_model=nm,
        fiducial_params=params,
        validate_linearity=False,
    )
    assert marged == frozenset(over)
    assert skel.free_names() == ("F0",)  # only F0 left to sample

    # Broad prior on F0 around its fiducial; likelihood dominates.
    f0_fid = float(params.param_value("F0"))
    priors = {"F0": dist.Normal(f0_fid, 1e-6)}

    model, init = build_single_pulsar_model(likelihood, priors, skel)
    # init seeds the sampler at the fiducial y_fid.
    assert set(init) == {"F0"}
    assert float(init["F0"]) == pytest.approx(f0_fid)

    idata, mcmc = run_nuts(
        model,
        init=init,
        key=jax.random.PRNGKey(0),
        num_warmup=50,
        num_samples=50,
        num_chains=1,
        progress_bar=False,
    )

    # --- Plumbing assertions ---
    samples = mcmc.get_samples()
    assert "F0" in samples
    assert samples["F0"].shape == (50,)
    assert bool(jnp.all(jnp.isfinite(samples["F0"])))
    # Draws sit near the fiducial (likelihood is sharply peaked there).
    assert abs(float(jnp.mean(samples["F0"])) - f0_fid) < 1e-3

    # ArviZ handoff produced an InferenceData labeled by site name.
    assert "F0" in idata.posterior


def test_packing_bridge_roundtrip():
    """The packing bridge reconstructs the skeleton exactly when the sampled
    value equals the fiducial free value (no MCMC needed)."""
    from jaxpint.bayes.samplers.numpyro import _pack_free

    toa_data, tm, nm, params = make_simple_pulsar(80, f0=100.0, f1=-1e-14, seed=1)
    over = {n for n in params.free_names() if n != "F0"}
    likelihood, _, skel = marginalize_single_pulsar(
        over=over, toa_data=toa_data, timing_model=tm, noise_model=nm,
        fiducial_params=params, validate_linearity=False,
    )
    # Feed the fiducial free value back in → packed vector == skeleton exactly.
    packed = _pack_free(skel, {"F0": skel.free_values()[0]})
    np.testing.assert_allclose(np.asarray(packed.values), np.asarray(skel.values), rtol=1e-12)
    np.testing.assert_allclose(float(likelihood(packed)), float(likelihood(skel)), rtol=1e-12)


@pytest.mark.slow
def test_pta_nuts_end_to_end():
    """PTA path: sample a per-pulsar free param (F0) AND global GWB
    hyperparameters, with F1 marginalized in each pulsar. Exercises
    collect_free_fqns -> resolve_priors -> build_pta_model -> run_nuts."""
    from jaxpint.pta.likelihood import PTAConfig
    from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
    from jaxpint.types import GlobalParams

    pulsar_names = ("P0", "P1")
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(2, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.asarray(positions)

    tds, tms, nms, pps = [], [], [], []
    for i in range(2):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=30 + 5 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=42 + i
        )
        tds.append(td); tms.append(tm); nms.append(nm); pps.append(pp)

    gwb = HDCorrelatedGWBInjector(
        pulsar_positions=positions,
        n_components=3,
        T_span=365.25 * 86400.0,
        prefix="gwb_",
        initial_values={"log10_A": -14.0, "gamma": 4.33},
    )
    global_params = gwb.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=tuple(tms),
        noise_models=tuple(nms),
        signal_injectors=(),
        correlated_injectors=(gwb,),
    )

    over = {f"{n}_F1" for n in pulsar_names}  # marginalize F1 in each; F0 stays free
    likelihood, marged, reduced_skeletons = marginalize_pta(
        over=over,
        config=config,
        pulsar_names=pulsar_names,
        fiducial_pulsar_params=tuple(pps),
        fiducial_global_params=global_params,
        validate_linearity=False,
    )
    assert marged == frozenset(over)

    # Sampled set = per-pulsar F0 + the two GWB globals.
    free_fqns = collect_free_fqns(pulsar_names, reduced_skeletons, global_params)
    assert set(free_fqns) == {"P0_F0", "P1_F0", "gwb_log10_A", "gwb_gamma"}

    spec = {f"{n}_F0": dist.Normal(200.0 + 10.0 * i, 1e-6)
            for i, n in enumerate(pulsar_names)}
    spec |= {"gwb_log10_A": dist.Uniform(-18.0, -11.0),
             "gwb_gamma": dist.Uniform(0.0, 7.0)}
    priors = resolve_priors(free_fqns, spec)

    model, init = build_pta_model(
        likelihood, priors, reduced_skeletons, global_params, pulsar_names
    )
    assert set(init) == set(free_fqns)

    idata, mcmc = run_nuts(
        model, init=init, key=jax.random.PRNGKey(0),
        num_warmup=50, num_samples=50, num_chains=1, progress_bar=False,
    )

    samples = mcmc.get_samples()
    for site in ("P0_F0", "P1_F0", "gwb_log10_A", "gwb_gamma"):
        assert samples[site].shape == (50,)
        assert bool(jnp.all(jnp.isfinite(samples[site]))), site
        assert site in idata.posterior
    # Priors respected / F0 near fiducial.
    assert abs(float(jnp.mean(samples["P0_F0"])) - 200.0) < 1e-3
    assert bool(jnp.all((samples["gwb_log10_A"] >= -18.0) & (samples["gwb_log10_A"] <= -11.0)))
    assert bool(jnp.all((samples["gwb_gamma"] >= 0.0) & (samples["gwb_gamma"] <= 7.0)))


def test_build_model_missing_prior_raises():
    toa_data, tm, nm, params = make_simple_pulsar(60, f0=100.0, f1=-1e-14, seed=2)
    over = {n for n in params.free_names() if n != "F0"}
    likelihood, _, skel = marginalize_single_pulsar(
        over=over, toa_data=toa_data, timing_model=tm, noise_model=nm,
        fiducial_params=params, validate_linearity=False,
    )
    with pytest.raises(PriorResolutionError, match="no prior distribution"):
        build_single_pulsar_model(likelihood, {}, skel)  # empty priors
