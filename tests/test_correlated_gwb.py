"""Tests for the correlated (cross-pulsar) GWB likelihood.

Validates:
1. Two-tier Woodbury correctness against a dense brute-force solve.
2. Equivalence with CURN (uncorrelated) when Gamma = I.
3. ORF matrix construction.
4. Gradient correctness (finite, non-zero).
5. Compatibility with per-pulsar intrinsic red noise.
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
from jaxpint.pta.correlated_likelihood import (
    CorrelatedPTAConfig,
    pta_logL_correlated,
    _per_pulsar_intermediates,
)
from jaxpint.pta.signals.gwb import (
    CURNInjector,
    fourier_basis,
    powerlaw_psd,
)
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
from jaxpint.pta.signals.orf import hd_orf, dipole_orf
from jaxpint.types import ParameterVector
from jaxpint.fitters import compute_time_residuals

from tests.helpers import make_toa_data, make_params


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_simple_pulsar(n_toas, f0, f1, seed=0, pepoch_int=59000.0):
    """Create a spindown pulsar with white noise."""
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
    """Create a multi-pulsar setup with random sky positions."""
    if n_toas_list is None:
        n_toas_list = [20 + i * 5 for i in range(n_pulsars)]

    rng = np.random.default_rng(42)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

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
        positions,
    )


def _dense_logL(toa_data_list, timing_models, noise_models, pulsar_params,
                gwb_injector, global_params):
    """Brute-force dense log-likelihood for validation.

    Forms the full global covariance matrix and solves directly.
    """
    n_psr = len(toa_data_list)
    S = gwb_injector.get_psd(global_params)
    Gamma = gwb_injector.get_orf_matrix()

    # Collect per-pulsar residuals, noise, and Fourier bases
    residuals = []
    N_diags = []
    F_bases = []

    for p in range(n_psr):
        r = compute_time_residuals(
            timing_models[p], toa_data_list[p], pulsar_params[p]
        )
        residuals.append(r)
        Ndiag, U, Phi = noise_models[p].covariance(
            toa_data_list[p], pulsar_params[p]
        )
        # For this test, build full per-pulsar covariance
        C_p = jnp.diag(Ndiag)
        if U.shape[1] > 0:
            C_p = C_p + U @ jnp.diag(Phi) @ U.T
        N_diags.append(C_p)
        F_bases.append(gwb_injector.get_fourier_basis(toa_data_list[p]))

    # Build full global covariance
    n_toas_list = [td.n_toas for td in toa_data_list]
    n_total = sum(n_toas_list)
    C_global = jnp.zeros((n_total, n_total))

    # Fill block-diagonal per-pulsar noise
    offset_a = 0
    for a in range(n_psr):
        na = n_toas_list[a]
        C_global = C_global.at[
            offset_a:offset_a + na, offset_a:offset_a + na
        ].add(N_diags[a])

        # GWB: cross-pulsar blocks
        offset_b = 0
        for b in range(n_psr):
            nb = n_toas_list[b]
            gwb_block = Gamma[a, b] * F_bases[a] @ jnp.diag(S) @ F_bases[b].T
            C_global = C_global.at[
                offset_a:offset_a + na, offset_b:offset_b + nb
            ].add(gwb_block)
            offset_b += nb
        offset_a += na

    # Concatenate residuals
    r_global = jnp.concatenate(residuals)

    # Dense solve
    L = jnp.linalg.cholesky(C_global)
    alpha = jax.scipy.linalg.cho_solve((L, True), r_global)
    rCr = jnp.dot(r_global, alpha)
    logdetC = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    logL = -0.5 * rCr - 0.5 * logdetC - 0.5 * n_total * jnp.log(2.0 * jnp.pi)
    return logL


# ---------------------------------------------------------------------------
# Tests: dense comparison
# ---------------------------------------------------------------------------


class TestDenseComparison:
    """Two-tier Woodbury matches brute-force dense solve."""

    def test_hd_correlated(self):
        """HD-correlated GWB: two-tier vs dense."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0  # 1 year
        n_components = 5

        gwb_injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=n_components,
            T_span=T_span,
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb_injector.register_params(GlobalParams.empty())

        config = CorrelatedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL_correlated(global_params, pulsar_params, config)
        logL_dense = _dense_logL(
            toa_data_list, timing_models, noise_models,
            pulsar_params, gwb_injector, global_params,
        )

        np.testing.assert_allclose(
            float(logL_corr), float(logL_dense), rtol=1e-8,
            err_msg="Two-tier Woodbury does not match dense solve",
        )

    def test_dipole_orf(self):
        """Dipole ORF: two-tier vs dense."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0
        n_components = 3

        gwb_injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=n_components,
            T_span=T_span,
            orf_func=dipole_orf,
            initial_values={"log10_A": -14.5, "gamma": 3.0},
        )
        global_params = gwb_injector.register_params(GlobalParams.empty())

        config = CorrelatedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL_correlated(global_params, pulsar_params, config)
        logL_dense = _dense_logL(
            toa_data_list, timing_models, noise_models,
            pulsar_params, gwb_injector, global_params,
        )

        np.testing.assert_allclose(
            float(logL_corr), float(logL_dense), rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# Tests: CURN equivalence
# ---------------------------------------------------------------------------


class TestCURNEquivalence:
    """When Gamma = I, correlated likelihood matches uncorrelated pta_logL."""

    def test_identity_orf_matches_curn(self):
        """Gamma = I (identity matrix) should reproduce CURN result.

        CURN gives each pulsar the same PSD independently, which is
        equivalent to Phi_gwb = I kron diag(S) (no cross-pulsar coupling).
        """
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0
        n_components = 5
        log10_A = -14.0
        gamma = 4.33

        # Build correlated injector, then manually override ORF to identity
        gwb_corr = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=n_components,
            T_span=T_span,
            initial_values={"log10_A": log10_A, "gamma": gamma},
        )
        # Override the ORF matrix to be the identity
        n_psr = positions.shape[0]
        gwb_corr._orf_matrix = jnp.eye(n_psr)

        gp_corr = gwb_corr.register_params(GlobalParams.empty())

        config_corr = CorrelatedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_corr,),
        )

        # CURN (uncorrelated, same as Gamma=I in the per-pulsar sum)
        curn = CURNInjector(
            n_components=n_components,
            T_span=T_span,
            initial_values={"log10_A": log10_A, "gamma": gamma},
        )
        gp_curn = curn.register_params(GlobalParams.empty())

        config_curn = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(curn,),
        )

        logL_corr = pta_logL_correlated(gp_corr, pulsar_params, config_corr)
        logL_curn = pta_logL(gp_curn, pulsar_params, config_curn)

        # Gamma=I correlated == CURN uncorrelated
        np.testing.assert_allclose(
            float(logL_corr), float(logL_curn), rtol=1e-8,
            err_msg="Identity-ORF correlated logL does not match CURN logL",
        )


# ---------------------------------------------------------------------------
# Tests: ORF matrix
# ---------------------------------------------------------------------------


class TestORFMatrix:
    """ORF matrix construction correctness."""

    def test_hd_orf_symmetric(self):
        """HD ORF matrix is symmetric."""
        rng = np.random.default_rng(0)
        positions = rng.normal(size=(5, 3))
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        positions = jnp.array(positions)

        injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=3,
            T_span=365.25 * 86400.0,
        )
        Gamma = injector.get_orf_matrix()
        np.testing.assert_allclose(Gamma, Gamma.T, atol=1e-15)

    def test_hd_orf_diagonal(self):
        """HD ORF diagonal is ~0.5 (self-correlation limit)."""
        rng = np.random.default_rng(1)
        positions = rng.normal(size=(4, 3))
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        positions = jnp.array(positions)

        injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=3,
            T_span=365.25 * 86400.0,
        )
        Gamma = injector.get_orf_matrix()
        np.testing.assert_allclose(
            jnp.diag(Gamma), 0.5, atol=1e-6,
            err_msg="HD ORF diagonal should be ~0.5",
        )

    def test_hd_orf_matches_pairwise(self):
        """ORF matrix entries match direct hd_orf() calls."""
        rng = np.random.default_rng(2)
        positions = rng.normal(size=(3, 3))
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        positions = jnp.array(positions)

        injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=3,
            T_span=365.25 * 86400.0,
        )
        Gamma = injector.get_orf_matrix()

        for a in range(3):
            for b in range(3):
                expected = float(hd_orf(positions[a], positions[b]))
                np.testing.assert_allclose(
                    float(Gamma[a, b]), expected, rtol=1e-12,
                )


# ---------------------------------------------------------------------------
# Tests: gradients
# ---------------------------------------------------------------------------


class TestGradients:
    """Gradient correctness for the correlated likelihood."""

    def test_grad_gwb_params(self):
        """Gradients w.r.t. GWB log10_A and gamma are finite and non-zero."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0
        gwb_injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=5,
            T_span=T_span,
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb_injector.register_params(GlobalParams.empty())

        config = CorrelatedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        def logL_fn(gp_values):
            gp = GlobalParams(
                values=gp_values,
                names=global_params.names,
                _name_to_index=global_params._name_to_index,
            )
            return pta_logL_correlated(gp, pulsar_params, config)

        grad = jax.grad(logL_fn)(global_params.values)

        assert jnp.all(jnp.isfinite(grad)), "Gradient contains non-finite values"
        assert not jnp.allclose(grad, 0.0), "Gradient is all zeros"


# ---------------------------------------------------------------------------
# Tests: with per-pulsar red noise
# ---------------------------------------------------------------------------


class TestWithRedNoise:
    """Correlated GWB + per-pulsar intrinsic red noise."""

    def test_red_noise_plus_gwb_matches_dense(self):
        """Two-tier Woodbury with intrinsic red noise matches dense solve."""
        n_pulsars = 3
        n_toas_list = [20, 25, 30]

        rng = np.random.default_rng(99)
        positions = rng.normal(size=(n_pulsars, 3))
        positions /= np.linalg.norm(positions, axis=1, keepdims=True)
        positions = jnp.array(positions)

        T_span = 365.25 * 86400.0
        n_components = 3
        n_red_components = 3

        toa_data_list = []
        timing_models = []
        noise_models = []
        pulsar_params = []

        for i in range(n_pulsars):
            td, tm, _, pp = _make_simple_pulsar(
                n_toas=n_toas_list[i],
                f0=200.0 + i * 10.0,
                f1=-1e-15,
                seed=99 + i,
            )

            # Build Fourier basis for red noise
            toas_s = (
                td.tdb_int.astype(jnp.float64) * 86400.0
                + td.tdb_frac * 86400.0
            )
            freqs_rn = jnp.arange(1, n_red_components + 1) / T_span
            phase = 2.0 * jnp.pi * toas_s[:, None] * freqs_rn[None, :]
            F_rn = jnp.column_stack([jnp.sin(phase), jnp.cos(phase)])
            df_rn = jnp.full(n_red_components, 1.0 / T_span)

            # Add red noise to noise model
            white_noise = ScaleToaError(
                efac_names=("EFAC1",), equad_names=("EQUAD1",)
            )
            red_noise = PLRedNoise(
                fourier_basis=F_rn,
                freqs=freqs_rn,
                freq_bin_widths=df_rn,
                tnredamp_name="TNREDAMP",
                tnredgam_name="TNREDGAM",
            )
            nm = NoiseModel(white_noise=white_noise, correlated=(red_noise,))

            # Add red noise params
            pp = make_params(
                names=pp.names + ("TNREDAMP", "TNREDGAM", "TNREDC"),
                values=list(np.array(pp.values)) + [-13.0, 3.0, n_red_components],
                frozen_mask=pp.frozen_mask + (True, True, True),
                epoch_int_values=pp.epoch_int_values,
            )

            toa_data_list.append(td)
            timing_models.append(tm)
            noise_models.append(nm)
            pulsar_params.append(pp)

        toa_data_list = tuple(toa_data_list)
        timing_models = tuple(timing_models)
        noise_models = tuple(noise_models)
        pulsar_params = tuple(pulsar_params)

        gwb_injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=n_components,
            T_span=T_span,
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb_injector.register_params(GlobalParams.empty())

        config = CorrelatedPTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL_correlated(global_params, pulsar_params, config)
        logL_dense = _dense_logL(
            toa_data_list, timing_models, noise_models,
            pulsar_params, gwb_injector, global_params,
        )

        np.testing.assert_allclose(
            float(logL_corr), float(logL_dense), rtol=1e-8,
            err_msg="Two-tier Woodbury with red noise does not match dense solve",
        )
