"""Tests for jaxpint.summary: model/PTA text dumps and their cross-checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from jaxpint.pta.likelihood import PTAConfig
from jaxpint.pta.signals.gwb import CURNInjector
from jaxpint.summary import summarize_model, summarize_pta
from jaxpint.types import GlobalParams

from tests.helpers import make_params, make_simple_pulsar, make_toa_data


@pytest.fixture
def simple_pulsar():
    return make_simple_pulsar(20, f0=200.0, f1=-1e-15)


# ===========================================================================
# summarize_model
# ===========================================================================


class TestSummarizeModel:
    def test_lists_components_and_params(self, simple_pulsar):
        toa_data, timing_model, noise_model, params = simple_pulsar
        text = summarize_model(
            timing_model, noise_model, params, toa_data=toa_data, name="J0000+0000"
        )
        print(text)
        assert "J0000+0000" in text
        assert "Spindown" in text
        assert "ScaleToaError" in text
        # Free/frozen accounting: F0, F1 free; PEPOCH, EFAC1, EQUAD1 frozen.
        assert "n=5: 2 free, 3 frozen, 0 marginalized" in text
        # Component param bindings carry live values and status.
        assert "F0=200 [free]" in text
        assert "EFAC1=1 [frozen]" in text
        # Epoch params render the int/frac split.
        assert "59000" in text

    def test_works_without_toa_data(self, simple_pulsar):
        _, timing_model, noise_model, params = simple_pulsar
        text = summarize_model(timing_model, noise_model, params)
        assert "Spindown" in text
        assert "Flag masks" not in text

    def test_orphan_parameter_reported(self, simple_pulsar):
        toa_data, timing_model, noise_model, _ = simple_pulsar
        # EQUAD7 is in the vector but no component reads it (a "typo").
        params = make_params(
            names=("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1", "EQUAD7"),
            values=(200.0, -1e-15, 0.0, 1.0, 0.0, 1e-6),
            frozen_mask=(False, False, True, True, True, True),
            epoch_int_values={"PEPOCH": 59000.0},
        )
        text = summarize_model(timing_model, noise_model, params, toa_data=toa_data)
        assert "not read by any component" in text
        assert "EQUAD7" in text.split("not read by any component")[1]

    def test_zero_count_mask_warning(self, simple_pulsar):
        _, timing_model, noise_model, params = simple_pulsar
        n = 20
        toa_data = make_toa_data(
            n,
            tdb_int=59000.0,
            tdb_frac=jnp.linspace(0.0, 1.0, n),
            error=1e-6,
            flag_masks={
                "EFAC1": jnp.ones(n, dtype=jnp.bool_),
                "EQUAD1": jnp.zeros(n, dtype=jnp.bool_),
            },
            tzr_tdb_int=59000.0,
            tzr_tdb_frac=0.5,
            tzr_freq=jnp.inf,
            tzr_ssb_obs_pos=jnp.zeros(3),
            tzr_obs_sun_pos=jnp.zeros(3),
        )
        text = summarize_model(timing_model, noise_model, params, toa_data=toa_data)
        assert "WARNING: mask(s) matching 0 TOAs" in text
        assert "EQUAD1" in text

    def test_marginalized_counted(self, simple_pulsar):
        toa_data, timing_model, noise_model, params = simple_pulsar
        params = params.with_marginalized(["F1"])
        text = summarize_model(timing_model, noise_model, params, toa_data=toa_data)
        assert "1 free, 3 frozen, 1 marginalized" in text
        assert "F1=-1e-15 [marg]" in text

    def test_covariance_shape_reported(self, simple_pulsar):
        toa_data, timing_model, noise_model, params = simple_pulsar
        text = summarize_model(timing_model, noise_model, params, toa_data=toa_data)
        # White-only model: zero-width U.
        assert "U (20, 0)" in text


# ===========================================================================
# summarize_pta
# ===========================================================================


@pytest.fixture
def small_pta():
    pulsars = [
        make_simple_pulsar(20, f0=200.0, f1=-1e-15, seed=i) for i in range(2)
    ]
    toa_data_list = tuple(p[0] for p in pulsars)
    timing_models = tuple(p[1] for p in pulsars)
    noise_models = tuple(p[2] for p in pulsars)
    pulsar_params = tuple(p[3] for p in pulsars)

    injector = CURNInjector(
        n_components=3,
        T_span=365.25 * 86400.0,
        initial_values={"log10_A": -14.0, "gamma": 4.33},
    )
    global_params = injector.register_params(GlobalParams.empty())

    config = PTAConfig(
        toa_data_list=toa_data_list,
        timing_models=timing_models,
        noise_models=noise_models,
        signal_injectors=(injector,),
    )
    return config, pulsar_params, global_params


class TestSummarizePta:
    def test_overview_and_injectors(self, small_pta):
        config, pulsar_params, global_params = small_pta
        text = summarize_pta(
            config,
            pulsar_params=pulsar_params,
            global_params=global_params,
            pulsar_names=("J0000+0000", "J1111+1111"),
        )
        assert "2 pulsar(s)" in text
        assert "J1111+1111" in text
        assert "CURNInjector" in text
        assert "gwb_log10_A" in text
        # Registered-by cross-check resolves the injector.
        assert "CURNInjector" in text.split("Global parameters")[1]

    def test_unregistered_global_warning(self, small_pta):
        config, pulsar_params, global_params = small_pta
        global_params = global_params.add_params(["stray_param"], [0.0])
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "no injector in this config registers" in text
        assert "stray_param" in text

    def test_prior_coverage(self, small_pta):
        config, pulsar_params, global_params = small_pta
        names = ("J0000+0000", "J1111+1111")
        # Priors cover the globals and pulsar 0's F0/F1 -- but not pulsar 1's.
        priors = {
            "gwb_log10_A": dist.Uniform(-18.0, -11.0),
            "gwb_gamma": dist.Uniform(0.0, 7.0),
            "J0000+0000_F0": dist.Normal(200.0, 1e-6),
            "J0000+0000_F1": dist.Normal(-1e-15, 1e-18),
        }
        text = summarize_pta(
            config,
            pulsar_params=pulsar_params,
            global_params=global_params,
            priors=priors,
            pulsar_names=names,
        )
        assert "<< MISSING >>" in text
        assert "2 expected site(s) have no prior" in text
        assert "Uniform" in text

    def test_config_summary_method(self, small_pta):
        config, pulsar_params, global_params = small_pta
        text = config.summary(
            pulsar_params=pulsar_params, global_params=global_params
        )
        assert "JaxPINT PTA summary" in text

    def test_verbose_embeds_model_dumps(self, small_pta):
        config, pulsar_params, global_params = small_pta
        text = summarize_pta(
            config,
            pulsar_params=pulsar_params,
            global_params=global_params,
            verbose=True,
        )
        assert text.count("JaxPINT model summary") == 2
        assert "Spindown" in text


# ===========================================================================
# Fitter passthrough
# ===========================================================================


class TestFitterSummary:
    def test_wls_fitter_summary(self, simple_pulsar):
        from jaxpint.fitters import WLSFitter

        toa_data, timing_model, noise_model, params = simple_pulsar
        fitter = WLSFitter(timing_model, toa_data, params, noise_model=noise_model)
        text = fitter.summary(name="J0000+0000")
        assert "JaxPINT model summary: J0000+0000" in text
        assert "Spindown" in text
        assert "ScaleToaError" in text

    def test_fitter_summary_without_noise_model(self, simple_pulsar):
        from jaxpint.fitters import WLSFitter

        toa_data, timing_model, _, params = simple_pulsar
        fitter = WLSFitter(timing_model, toa_data, params)
        text = fitter.summary()
        assert "white (Ndiag): (none)" in text
