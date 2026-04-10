"""Tests for the batched PTA likelihood (jaxpint.pta.batching).

Validates that pta_logL_batched matches the loop-based pta_logL
from jaxpint.pta.likelihood.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.noise.white import ScaleToaError
from jaxpint.noise.red_noise import PLRedNoise
from jaxpint.phase.spin import Spindown
from jaxpint.pta.params import GlobalParams
from jaxpint.pta.likelihood import PTAConfig, pta_logL
from jaxpint.pta.batching import BatchedPTAConfig, pta_logL_batched
from jaxpint.pta.signals.cw import CWInjectorStack
from jaxpint.types import ParameterVector

from tests.helpers import make_toa_data, make_params


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_pulsar(
    n_toas: int,
    f0: float,
    f1: float,
    dm: float,
    pepoch_int: float = 59000.0,
    tdb_int: float = 59000.0,
    error: float = 1e-6,
    seed: int = 0,
):
    """Create a simple spindown pulsar with white noise.

    Returns (toa_data, timing_model, noise_model, params).
    """
    rng = np.random.default_rng(seed)

    # TOA data
    tdb_frac = jnp.array(np.sort(rng.uniform(0.0, 1.0, n_toas)))
    efac_mask = jnp.ones(n_toas, dtype=jnp.bool_)
    equad_mask = jnp.ones(n_toas, dtype=jnp.bool_)

    toa_data = make_toa_data(
        n_toas,
        tdb_int=tdb_int,
        tdb_frac=tdb_frac,
        error=error,
        flag_masks={"EFAC1": efac_mask, "EQUAD1": equad_mask},
        tzr_tdb_int=pepoch_int,
        tzr_tdb_frac=0.5,
        tzr_freq=jnp.inf,
        tzr_ssb_obs_pos=jnp.zeros(3),
        tzr_obs_sun_pos=jnp.zeros(3),
    )

    # Timing model: simple spindown
    spindown = Spindown(spin_param_names=("F0", "F1"), pepoch_name="PEPOCH")
    timing_model = TimingModel(
        delay_components=(),
        phase_components=(spindown,),
        phoff_name=None,
    )

    # Noise model: white noise only
    white_noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    noise_model = NoiseModel(white_noise=white_noise, correlated=())

    # Parameters
    params = make_params(
        names=("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1"),
        values=(f0, f1, 0.0, 1.0, 0.0),
        frozen_mask=(False, False, True, True, True),
        epoch_int_values={"PEPOCH": pepoch_int},
    )

    return toa_data, timing_model, noise_model, params


def _make_multi_pulsar_setup(n_pulsars=3, n_toas_list=None):
    """Create a multi-pulsar setup with varying TOA counts.

    Returns (toa_data_list, timing_models, noise_models,
             pulsar_params, global_params).
    """
    if n_toas_list is None:
        n_toas_list = [100 + i * 50 for i in range(n_pulsars)]

    toa_data_list = []
    timing_models = []
    noise_models = []
    pulsar_params = []

    for i in range(n_pulsars):
        f0 = 200.0 + i * 10.0
        f1 = -1e-15 * (1 + i * 0.5)
        dm = 15.0 + i * 2.0

        td, tm, nm, pp = _make_simple_pulsar(
            n_toas=n_toas_list[i],
            f0=f0,
            f1=f1,
            dm=dm,
            seed=42 + i,
        )
        toa_data_list.append(td)
        timing_models.append(tm)
        noise_models.append(nm)
        pulsar_params.append(pp)

    global_params = GlobalParams.empty()

    return (
        tuple(toa_data_list),
        tuple(timing_models),
        tuple(noise_models),
        tuple(pulsar_params),
        global_params,
    )


def _make_multi_pulsar_cw_setup(n_pulsars=3, n_cw_sources=1):
    """Multi-pulsar setup with CW signal injection."""
    toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
        _make_multi_pulsar_setup(n_pulsars)
    )

    # Pulsar positions (unit vectors)
    rng = np.random.default_rng(123)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    # Each pulsar needs a PX parameter for distance
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

    # CW injector
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
# Tests: batched matches loop
# ---------------------------------------------------------------------------


class TestBatchedMatchesLoop:
    """Core correctness: pta_logL_batched == pta_logL."""

    def test_simple_no_injectors(self):
        """Simplest case: spindown + white noise, no signal injection."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=3)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            pulsar_params=pulsar_params,
        )

        logL_loop = pta_logL(global_params, pulsar_params, config)
        logL_batched = pta_logL_batched(global_params, pulsar_params, batched_config)

        np.testing.assert_allclose(
            float(logL_batched), float(logL_loop), rtol=1e-10,
            err_msg="Batched logL does not match loop logL",
        )

    def test_with_cw_injection(self):
        """With CW signal injection via CWInjectorStack."""
        (
            toa_data_list, timing_models, noise_models,
            signal_injectors, pulsar_params, global_params,
        ) = _make_multi_pulsar_cw_setup(n_pulsars=3, n_cw_sources=2)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
            pulsar_params=pulsar_params,
        )

        logL_loop = pta_logL(global_params, pulsar_params, config)
        logL_batched = pta_logL_batched(global_params, pulsar_params, batched_config)

        np.testing.assert_allclose(
            float(logL_batched), float(logL_loop), rtol=1e-10,
            err_msg="Batched logL (with CW) does not match loop logL",
        )

    def test_different_toa_counts(self):
        """Pulsars with very different TOA counts."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(
                n_pulsars=4, n_toas_list=[50, 100, 200, 300]
            )
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            pulsar_params=pulsar_params,
        )

        logL_loop = pta_logL(global_params, pulsar_params, config)
        logL_batched = pta_logL_batched(global_params, pulsar_params, batched_config)

        np.testing.assert_allclose(
            float(logL_batched), float(logL_loop), rtol=1e-10,
        )

    def test_single_pulsar(self):
        """Edge case: single pulsar should still work."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=1)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            pulsar_params=pulsar_params,
        )

        logL_loop = pta_logL(global_params, pulsar_params, config)
        logL_batched = pta_logL_batched(global_params, pulsar_params, batched_config)

        np.testing.assert_allclose(
            float(logL_batched), float(logL_loop), rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests: gradient correctness
# ---------------------------------------------------------------------------


class TestBatchedGradients:
    """Gradients of batched path match gradients of loop path."""

    def test_grad_global_params(self):
        """Gradient w.r.t. global_params (CW)."""
        (
            toa_data_list, timing_models, noise_models,
            signal_injectors, pulsar_params, global_params,
        ) = _make_multi_pulsar_cw_setup(n_pulsars=3, n_cw_sources=1)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
            pulsar_params=pulsar_params,
        )

        # Gradient w.r.t. global_params.values
        def loop_fn(gp_values):
            gp = GlobalParams(values=gp_values, names=global_params.names,
                              _name_to_index=global_params._name_to_index)
            return pta_logL(gp, pulsar_params, config)

        def batched_fn(gp_values):
            gp = GlobalParams(values=gp_values, names=global_params.names,
                              _name_to_index=global_params._name_to_index)
            return pta_logL_batched(gp, pulsar_params, batched_config)

        grad_loop = jax.grad(loop_fn)(global_params.values)
        grad_batched = jax.grad(batched_fn)(global_params.values)

        np.testing.assert_allclose(
            np.array(grad_batched), np.array(grad_loop), rtol=1e-8,
            err_msg="Global param gradients do not match",
        )

    def test_grad_pulsar_params(self):
        """Gradient w.r.t. per-pulsar param values."""
        toa_data_list, timing_models, noise_models, pulsar_params, global_params = (
            _make_multi_pulsar_setup(n_pulsars=2)
        )

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        batched_config = BatchedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            pulsar_params=pulsar_params,
        )

        # Differentiate w.r.t. first pulsar's values
        def loop_fn(pv0):
            pp0 = pulsar_params[0]
            pp0_new = ParameterVector(
                values=pv0, frozen_mask=pp0.frozen_mask,
                names=pp0.names, units=pp0.units,
                epoch_int_values=pp0.epoch_int_values,
            )
            return pta_logL(global_params, (pp0_new, pulsar_params[1]), config)

        def batched_fn(pv0):
            pp0 = pulsar_params[0]
            pp0_new = ParameterVector(
                values=pv0, frozen_mask=pp0.frozen_mask,
                names=pp0.names, units=pp0.units,
                epoch_int_values=pp0.epoch_int_values,
            )
            return pta_logL_batched(
                global_params, (pp0_new, pulsar_params[1]), batched_config
            )

        grad_loop = jax.grad(loop_fn)(pulsar_params[0].values)
        grad_batched = jax.grad(batched_fn)(pulsar_params[0].values)

        np.testing.assert_allclose(
            np.array(grad_batched), np.array(grad_loop), rtol=1e-8,
            err_msg="Pulsar param gradients do not match",
        )


# ---------------------------------------------------------------------------
# Tests: padding correctness
# ---------------------------------------------------------------------------


class TestPadding:
    """Padding utilities produce correct results."""

    def test_universal_param_layout(self):
        """Universal layout contains all parameter names."""
        from jaxpint.pta.padding import build_universal_param_layout

        pp0 = make_params(("F0", "F1", "PEPOCH"), (200.0, -1e-15, 0.0))
        pp1 = make_params(("F0", "PEPOCH", "DM"), (210.0, 0.0, 15.0))

        layout = build_universal_param_layout((pp0, pp1))
        assert set(layout.names) == {"F0", "F1", "PEPOCH", "DM"}
        assert layout.n_params == 4

    def test_reindex_preserves_values(self):
        """Reindexed values match originals at correct positions."""
        from jaxpint.pta.padding import (
            build_universal_param_layout,
            reindex_param_values,
        )

        pp0 = make_params(("F0", "F1"), (200.0, -1e-15))
        pp1 = make_params(("F0", "DM"), (210.0, 15.0))

        layout = build_universal_param_layout((pp0, pp1))
        vals0 = reindex_param_values(pp0, layout)
        vals1 = reindex_param_values(pp1, layout)

        # Check that F0 values are correct
        f0_idx = layout._name_to_index["F0"]
        assert float(vals0[f0_idx]) == pytest.approx(200.0)
        assert float(vals1[f0_idx]) == pytest.approx(210.0)

        # Check that missing params get defaults
        dm_idx = layout._name_to_index["DM"]
        assert float(vals0[dm_idx]) == pytest.approx(0.0)  # DM missing from pp0

    def test_pad_toa_data_shape(self):
        """Padded TOAData has correct shapes."""
        from jaxpint.pta.padding import pad_toa_data

        td = make_toa_data(n_toas=50)
        padded = pad_toa_data(td, n_max=100)

        assert padded.n_toas == 100
        assert padded.tdb_int.shape == (100,)
        assert padded.ssb_obs_pos.shape == (100, 3)
        assert padded.error.shape == (100,)

    def test_pad_toa_data_error_padding(self):
        """Padded TOA errors are 1.0 (so Ndiag padding = 1.0)."""
        from jaxpint.pta.padding import pad_toa_data

        td = make_toa_data(n_toas=5, error=1e-6)
        padded = pad_toa_data(td, n_max=10)

        # Original errors preserved
        np.testing.assert_allclose(padded.error[:5], 1e-6)
        # Padded errors are 1.0
        np.testing.assert_allclose(padded.error[5:], 1.0)
