"""Noise-marginalized optimal statistic (NMOS) driver.

``noise_marginalized_os`` must be exactly one OS evaluation per posterior
draw: the reference is an explicit Python loop of
``per_pulsar_gw_blocks`` + ``optimal_statistic`` at each draw's parameters.
The draws vary a per-pulsar white-noise parameter (EFAC) *and* the global
GWB amplitude, so the test catches a driver that fails to thread either
into the blocks or into ``gwnorm``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.frequentist.optimal import noise_marginalized_os, optimal_statistic
from jaxpint.pta.likelihood import PTAConfig, per_pulsar_gw_blocks

from tests.test_optimal_blocks import _two_pulsar_gwb_config

jax.config.update("jax_enable_x64", True)

N_DRAWS = 4


def _make_draws(gp, pps, n_draws=N_DRAWS, seed=0):
    """Batched (global, per-pulsar) parameter draws varying EFAC and log10_A."""
    rng = np.random.default_rng(seed)

    gvals = np.tile(np.asarray(gp.values), (n_draws, 1))
    gvals[:, gp.param_index("gwb_log10_A")] = rng.uniform(-14.5, -13.5, n_draws)
    gp_draws = gp.with_values(jnp.asarray(gvals))

    pp_draws = []
    for pp in pps:
        vals = np.tile(np.asarray(pp.values), (n_draws, 1))
        vals[:, pp.param_index("EFAC1")] = rng.uniform(0.8, 1.2, n_draws)
        pp_draws.append(pp.with_values(jnp.asarray(vals)))
    return gp_draws, tuple(pp_draws), gvals, [np.asarray(p.values) for p in pp_draws]


def test_nmos_matches_explicit_loop():
    """vmapped NMOS == per-draw loop of blocks + OS, and the draws matter."""
    gp, pps, config, _ = _two_pulsar_gwb_config()
    gp_draws, pp_draws, gvals, pvals = _make_draws(gp, pps)

    out = noise_marginalized_os(gp_draws, pp_draws, config)
    assert out.snr.shape == (N_DRAWS,)

    for i in range(N_DRAWS):
        gp_i = gp.with_values(jnp.asarray(gvals[i]))
        pps_i = tuple(
            pp.with_values(jnp.asarray(vals[i])) for pp, vals in zip(pps, pvals)
        )
        blocks = per_pulsar_gw_blocks(gp_i, pps_i, config)
        ref = optimal_statistic(blocks, gp_i.param_value("gwb_log10_A"))
        npt.assert_allclose(
            float(out.a_squared[i]), float(ref.a_squared), rtol=1e-12
        )
        npt.assert_allclose(
            float(out.a_squared_sigma[i]), float(ref.a_squared_sigma), rtol=1e-12
        )
        npt.assert_allclose(float(out.snr[i]), float(ref.snr), rtol=1e-12)

    # The noise draws must actually propagate: distinct draws, distinct OS.
    assert np.std(np.asarray(out.a_squared_sigma)) > 0
    assert np.std(np.asarray(out.snr)) > 0


def test_nmos_batch_size_matches_vmap():
    """The lax.map chunked path (with remainder) equals the vmap path."""
    gp, pps, config, _ = _two_pulsar_gwb_config()
    gp_draws, pp_draws, _, _ = _make_draws(gp, pps)

    ref = noise_marginalized_os(gp_draws, pp_draws, config)
    out = noise_marginalized_os(gp_draws, pp_draws, config, batch_size=3)
    for a, b in zip(out, ref):
        npt.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12)


def test_nmos_requires_single_correlated_injector():
    gp, pps, config, _ = _two_pulsar_gwb_config()
    gp_draws, pp_draws, _, _ = _make_draws(gp, pps)
    bad = PTAConfig(
        toa_data_list=config.toa_data_list,
        timing_models=config.timing_models,
        noise_models=config.noise_models,
        signal_injectors=(),
        correlated_injectors=(),
    )
    with pytest.raises(ValueError, match="exactly one correlated injector"):
        noise_marginalized_os(gp_draws, pp_draws, bad)
