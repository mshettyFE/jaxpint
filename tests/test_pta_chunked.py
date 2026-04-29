"""Tests for the chunked PTA likelihood.

Validates that :func:`pta_logL_chunked` matches :func:`pta_logL` across
chunk-size variants and that summing per-chunk gradients of the
underlying ``_chunk_logL`` matches the gradient of the loop-based
reference.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.noise.white import ScaleToaError
from jaxpint.phase.spin import Spindown
from jaxpint.pta.params import GlobalParams
from jaxpint.pta.likelihood import (
    PTAConfig,
    pta_logL,
    pta_logL_chunked,
    _chunk_logL,
)
from jaxpint.pta.signals.cw import CWInjectorStack

from tests.helpers import make_toa_data, make_params


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_pulsar(n_toas, f0, f1, seed=0, pepoch_int=59000.0):
    """Spindown pulsar with white noise."""
    rng = np.random.default_rng(seed)
    tdb_frac = jnp.array(np.sort(rng.uniform(0.0, 1.0, n_toas)))
    efac_mask = jnp.ones(n_toas, dtype=jnp.bool_)
    equad_mask = jnp.ones(n_toas, dtype=jnp.bool_)

    toa_data = make_toa_data(
        n_toas,
        tdb_int=pepoch_int,
        tdb_frac=tdb_frac,
        error=1e-6,
        flag_masks={"EFAC1": efac_mask, "EQUAD1": equad_mask},
        tzr_tdb_int=pepoch_int,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=jnp.zeros(3),
        tzr_obs_sun_pos=jnp.zeros(3),
    )

    spindown = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    timing_model = TimingModel(
        delay_components=(),
        phase_components=(spindown,),
        phoff_name=None,
    )

    white_noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    noise_model = NoiseModel(white_noise=white_noise, correlated=())

    params = make_params(
        names=("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1"),
        values=(f0, f1, 0.0, 1.0, 0.0),
        frozen_mask=(False, False, True, True, True),
        epoch_int_values={"PEPOCH": pepoch_int},
    )

    return toa_data, timing_model, noise_model, params


def _make_multi_pulsar_setup(n_pulsars=3, n_toas_list=None):
    """Multi-pulsar setup with no signal injectors."""
    if n_toas_list is None:
        n_toas_list = [40 + i * 15 for i in range(n_pulsars)]

    toa_data_list = []
    timing_models = []
    noise_models = []
    pulsar_params = []

    for i in range(n_pulsars):
        td, tm, nm, pp = _make_simple_pulsar(
            n_toas=n_toas_list[i],
            f0=200.0 + i * 10.0,
            f1=-1e-15 * (1 + i * 0.5),
            seed=42 + i,
        )
        toa_data_list.append(td)
        timing_models.append(tm)
        noise_models.append(nm)
        pulsar_params.append(pp)

    return (
        tuple(toa_data_list),
        tuple(timing_models),
        tuple(noise_models),
        tuple(pulsar_params),
        GlobalParams.empty(),
    )


def _make_multi_pulsar_cw_setup(n_pulsars=3, n_cw_sources=1):
    """Multi-pulsar setup with CW signal injection."""
    toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
        _make_multi_pulsar_setup(n_pulsars)
    )

    rng = np.random.default_rng(123)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    new_pulsar_params = []
    for pp in pulsar_params:
        new_pp = make_params(
            names=pp.names + ("PX",),
            values=list(np.array(pp.values)) + [0.5],
            frozen_mask=pp.frozen_mask + (True,),
            epoch_int_values=pp.epoch_int_values,
        )
        new_pulsar_params.append(new_pp)
    pulsar_params = tuple(new_pulsar_params)

    cw_injector = CWInjectorStack(
        pulsar_positions=positions,
        n_sources=n_cw_sources,
    )
    global_params = cw_injector.register_params(global_params)

    return (
        toa_data_list,
        timing_models,
        noise_models,
        (cw_injector,),
        pulsar_params,
        global_params,
    )


# ---------------------------------------------------------------------------
# Numeric equivalence: chunked == loop
# ---------------------------------------------------------------------------


CHUNK_SIZES = (1, 2, 4)


class TestChunkedMatchesLoop:
    """pta_logL_chunked must match pta_logL across chunk-size variants."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_no_injectors(self, chunk_size):
        """Spindown + white noise, no signal injection."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=5)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=chunk_size,
        )

        np.testing.assert_allclose(
            logL_chunked, logL_loop, rtol=1e-12, atol=1e-15,
            err_msg=f"chunk_size={chunk_size} mismatch (no injectors)",
        )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_with_cw_injection(self, chunk_size):
        """With CW signal injection (per-pulsar dispatch via p_global)."""
        (
            toa_data_list, timing_models, noise_models,
            signal_injectors, pulsar_params, global_params,
        ) = _make_multi_pulsar_cw_setup(n_pulsars=5, n_cw_sources=2)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=chunk_size,
        )

        np.testing.assert_allclose(
            logL_chunked, logL_loop, rtol=1e-12, atol=1e-15,
            err_msg=f"chunk_size={chunk_size} mismatch (with CW)",
        )

    def test_chunk_size_equals_n(self):
        """chunk_size = n_pulsars: a single chunk covering everything."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=3)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=3,
        )

        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-12, atol=1e-15)

    def test_chunk_size_exceeds_n(self):
        """chunk_size > n_pulsars: still one chunk."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=3)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=10,
        )

        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-12, atol=1e-15)

    def test_single_pulsar(self):
        """Edge case: single pulsar."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=1)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=1,
        )

        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-12, atol=1e-15)

    def test_different_toa_counts(self):
        """Pulsars with very different TOA counts."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(
                n_pulsars=4, n_toas_list=[20, 60, 120, 200],
            )
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )

        logL_loop = float(pta_logL(global_params, pulsar_params, config))
        logL_chunked = pta_logL_chunked(
            global_params, pulsar_params, config, chunk_size=2,
        )

        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestChunkedValidation:
    """Argument validation."""

    def test_zero_chunk_size_raises(self):
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=2)
        )
        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            pta_logL_chunked(global_params, pulsar_params, config, chunk_size=0)

    def test_negative_chunk_size_raises(self):
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=2)
        )
        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            pta_logL_chunked(global_params, pulsar_params, config, chunk_size=-1)


# ---------------------------------------------------------------------------
# Gradient equivalence via _chunk_logL
# ---------------------------------------------------------------------------


class TestChunkGradients:
    """Per-chunk gradients of ``_chunk_logL`` sum to the loop gradient.

    ``pta_logL_chunked`` returns a Python ``float`` (after the per-chunk
    ``float(...)`` block), so it is intentionally not directly grad-able.
    For differentiable callers, we verify that the underlying jitted
    ``_chunk_logL`` is differentiable and that summing gradients across
    chunks reproduces the gradient of the loop-based reference.  This
    confirms the chunking decomposition is mathematically faithful even
    though the public wrapper drops gradient support.
    """

    def test_grad_global_params_chunks_sum_to_loop(self):
        (
            toa_data_list, timing_models, noise_models,
            signal_injectors, pulsar_params, global_params,
        ) = _make_multi_pulsar_cw_setup(n_pulsars=4, n_cw_sources=1)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
        )

        # Reference gradient: jax.grad on the loop-based pta_logL
        def loop_logL(gp, pp):
            return pta_logL(gp, pp, config)

        ref_grad = jax.grad(loop_logL, argnums=0)(global_params, pulsar_params)

        # Sum gradients of _chunk_logL over chunks of size 2
        chunk_size = 2
        n = len(pulsar_params)
        accumulated = jax.tree.map(jnp.zeros_like, global_params)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)

            def chunk_fn(gp, start=start, end=end):
                return _chunk_logL(
                    gp,
                    pulsar_params[start:end],
                    toa_data_list[start:end],
                    noise_models[start:end],
                    timing_models,
                    signal_injectors,
                    start,
                )

            chunk_grad = jax.grad(chunk_fn)(global_params)
            accumulated = jax.tree.map(jnp.add, accumulated, chunk_grad)

        ref_leaves = jax.tree.leaves(ref_grad)
        acc_leaves = jax.tree.leaves(accumulated)
        assert len(ref_leaves) == len(acc_leaves)
        for ref_leaf, acc_leaf in zip(ref_leaves, acc_leaves):
            np.testing.assert_allclose(
                np.asarray(acc_leaf), np.asarray(ref_leaf),
                rtol=1e-10, atol=1e-14,
            )
