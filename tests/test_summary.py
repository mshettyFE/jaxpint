"""Tests for jaxpint.summary: model/PTA text dumps and their cross-checks."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from jaxpint.pta.likelihood import PTAConfig
from jaxpint.pta.signals.cw import CWInjector
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
# Injector parameter crediting (required_pulsar_params convention)
# ===========================================================================


def _pta_with_cw(with_px=(True, True), **cw_kwargs):
    """Two-pulsar PTA with one CWInjector; PX per pulsar controlled by with_px."""
    pulsars = [make_simple_pulsar(20, f0=200.0, f1=-1e-15, seed=i) for i in range(2)]
    base_names = ("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1")
    base_values = (200.0, -1e-15, 0.0, 1.0, 0.0)
    base_frozen = (False, False, True, True, True)
    pulsar_params = tuple(
        make_params(
            names=base_names + (("PX",) if has_px else ()),
            values=base_values + ((1.0,) if has_px else ()),
            frozen_mask=base_frozen + ((True,) if has_px else ()),
            epoch_int_values={"PEPOCH": 59000.0},
        )
        for has_px in with_px
    )
    positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    injector = CWInjector(positions, **cw_kwargs)
    global_params = injector.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(p[0] for p in pulsars),
        timing_models=tuple(p[1] for p in pulsars),
        noise_models=tuple(p[2] for p in pulsars),
        signal_injectors=(injector,),
    )
    return config, pulsar_params, global_params


class TestInjectorParamCrediting:
    def test_reads_line_all_pulsars(self):
        config, pulsar_params, global_params = _pta_with_cw()
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "reads per-pulsar: PX (all pulsars)" in text

    def test_reads_line_none_for_curn(self, small_pta):
        config, pulsar_params, global_params = small_pta
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "reads per-pulsar: (none)" in text

    def test_reads_line_respects_pulsar_term_mask(self):
        config, pulsar_params, global_params = _pta_with_cw(
            pulsar_term_mask=(True, False)
        )
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "reads per-pulsar: PX (pulsars 0)" in text

    def test_reads_line_earth_term_only(self):
        config, pulsar_params, global_params = _pta_with_cw(earth_term_only=True)
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "reads per-pulsar: (none)" in text

    def test_missing_declared_param_warns(self):
        config, pulsar_params, global_params = _pta_with_cw(with_px=(True, False))
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        assert "WARNING: reads PX, missing from pulsar_params" in text
        assert "psr1" in text

    def test_verbose_read_by_credits_injector(self):
        config, pulsar_params, global_params = _pta_with_cw()
        text = summarize_pta(
            config,
            pulsar_params=pulsar_params,
            global_params=global_params,
            verbose=True,
        )
        # PX's "read by" column credits the injector instance...
        assert "CWInjector#1" in text
        # ...so PX is no longer an orphan (and nothing else is either).
        assert "not read by any component" not in text

    def test_multiple_instances_disambiguated(self):
        config, pulsar_params, _ = _pta_with_cw()
        positions = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        cw2 = CWInjector(positions, prefix="cw1_")
        config = PTAConfig(
            toa_data_list=config.toa_data_list,
            timing_models=config.timing_models,
            noise_models=config.noise_models,
            signal_injectors=config.signal_injectors + (cw2,),
        )
        global_params = cw2.register_params(
            config.signal_injectors[0].register_params(GlobalParams.empty())
        )
        text = summarize_pta(
            config, pulsar_params=pulsar_params, global_params=global_params
        )
        table = text.split("Global parameters")[1]
        assert "CWInjector#1" in table
        assert "CWInjector#2" in table


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


# ===========================================================================
# Noise-section basis reporting (basis kind, host-side width, ECORR check)
# ===========================================================================

import numpy as np

from jaxpint.noise import NoiseModel
from jaxpint.noise.ecorr import EcorrNoise
from jaxpint.summary import _basis_width
from tests.test_chrom_noise import _make_plchrom
from tests.test_red_noise import _make_plred

_T3YR = 3.0 * 365.25 * 86400.0


class TestNoiseBasisReporting:
    def test_basis_width_is_host_side(self):
        """Width comes from host columns: correct value, no device cache
        populated -- a text dump must not defeat deferred device allocation."""
        comp, params, toa_data, *_ = _make_plred(n_toas=30, n_freqs=4, T=_T3YR)
        assert _basis_width(comp, toa_data, params) == "n_basis=8"
        assert "_columns_jax_cache" not in comp.__dict__

    def test_noise_section_basis_tags(self, simple_pulsar):
        """Static (red) and dynamic (chromatic) components are tagged."""
        _, timing_model, _, _ = simple_pulsar
        red, _, toa_data, *_ = _make_plred(n_toas=20, n_freqs=3, T=_T3YR)
        chrom, *_ = _make_plchrom(n_toas=20, n_freqs=3, T=_T3YR)
        params = make_params(
            ("TNREDAMP", "TNREDGAM", "TNCHROMAMP", "TNCHROMGAM", "TNCHROMIDX"),
            (-13.0, 3.5, -13.0, 3.5, 4.0),
            units=("",) * 5,
        )
        nm = NoiseModel(white_noise=None, correlated=(red, chrom))
        text = summarize_model(timing_model, nm, params, toa_data=toa_data)
        red_line = next(line for line in text.splitlines() if "PLRedNoise" in line)
        chrom_line = next(line for line in text.splitlines() if "PLChromNoise" in line)
        assert "basis=static" in red_line and "n_basis=6" in red_line
        assert "basis=dynamic" in chrom_line and "n_basis=6" in chrom_line

    def test_ecorr_zero_epoch_warning(self, simple_pulsar):
        """An ECORR parameter whose quantization kept no epochs is flagged;
        healthy siblings are not."""
        _, timing_model, _, _ = simple_pulsar
        n = 6
        U = np.zeros((n, 2))
        U[:3, 0] = 1.0
        U[3:, 1] = 1.0
        ec = EcorrNoise(
            ecorr_names=("ECORR1", "ECORR2"),
            quantization_matrix=jnp.asarray(U),
            ecorr_epoch_slices=((0, 2), (2, 2)),  # ECORR2: empty slice
        )
        params = make_params(("ECORR1", "ECORR2"), (1e-6, 1e-6), units=("s", "s"))
        nm = NoiseModel(white_noise=None, correlated=(ec,))
        toa_data = make_toa_data(n_toas=n)
        text = summarize_model(timing_model, nm, params, toa_data=toa_data)
        warning = next(line for line in text.splitlines() if "zero kept epochs" in line)
        assert "ECORR2" in warning
        assert "ECORR1" not in warning

    def test_no_ecorr_warning_when_all_slices_populated(self, simple_pulsar):
        _, timing_model, _, _ = simple_pulsar
        n = 6
        U = np.zeros((n, 2))
        U[:3, 0] = 1.0
        U[3:, 1] = 1.0
        ec = EcorrNoise(
            ecorr_names=("ECORR1", "ECORR2"),
            quantization_matrix=jnp.asarray(U),
            ecorr_epoch_slices=((0, 1), (1, 2)),
        )
        params = make_params(("ECORR1", "ECORR2"), (1e-6, 1e-6), units=("s", "s"))
        nm = NoiseModel(white_noise=None, correlated=(ec,))
        text = summarize_model(
            timing_model, nm, params, toa_data=make_toa_data(n_toas=n)
        )
        assert "zero kept epochs" not in text
