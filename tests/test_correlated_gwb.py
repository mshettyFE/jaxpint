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

from jaxpint.noise import NoiseModel
from jaxpint.noise.white import ScaleToaError
from jaxpint.noise.red_noise import PLRedNoise
from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import (
    PTAConfig,
    pta_logL,
)
from jaxpint.pta.signals.gwb import (
    CURNInjector,
)
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
from jaxpint.pta.signals.orf import hd_orf, dipole_orf
from jaxpint.fitters import compute_time_residuals

from tests.helpers import make_simple_pulsar, make_params


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
        td, tm, nm, pp = make_simple_pulsar(
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


def _dense_logL_multi(toa_data_list, timing_models, noise_models, pulsar_params,
                      gwb_injectors, global_params):
    """Brute-force dense log-likelihood for validation, multi-injector capable.

    ``gwb_injectors`` is a tuple of :class:`CorrelatedSignalInjector`.  The
    global covariance is assembled as
    ``C = blockdiag(C_p) + Σ_k Γ_k[a,b] F_{k,a} diag(S_k) F_{k,b}^T``
    across pulsar pairs (a, b), then solved densely via Cholesky.  A
    length-1 tuple reproduces the single-injector formula exactly.
    """
    n_psr = len(toa_data_list)
    K = len(gwb_injectors)

    S_per_k = [cinj.get_psd(global_params) for cinj in gwb_injectors]
    Gamma_per_k = [cinj.get_orf_matrix() for cinj in gwb_injectors]

    # Collect per-pulsar residuals, noise, and Fourier bases (per injector).
    residuals = []
    C_p_list = []
    F_per_k_per_p: list[list] = [[] for _ in range(K)]

    for p in range(n_psr):
        r = compute_time_residuals(
            timing_models[p], toa_data_list[p], pulsar_params[p]
        )
        residuals.append(r)
        Ndiag, U, Phi = noise_models[p].covariance(
            toa_data_list[p], pulsar_params[p]
        )
        # Full per-pulsar covariance, densely.
        C_p = jnp.diag(Ndiag)
        if U.shape[1] > 0:
            C_p = C_p + U @ jnp.diag(Phi) @ U.T
        C_p_list.append(C_p)
        for k, cinj in enumerate(gwb_injectors):
            F_per_k_per_p[k].append(cinj.get_fourier_basis(toa_data_list[p]))

    # Build full global covariance.
    n_toas_list = [td.n_toas for td in toa_data_list]
    n_total = sum(n_toas_list)
    C_global = jnp.zeros((n_total, n_total))

    offset_a = 0
    for a in range(n_psr):
        na = n_toas_list[a]
        # Block-diagonal per-pulsar noise.
        C_global = C_global.at[
            offset_a:offset_a + na, offset_a:offset_a + na
        ].add(C_p_list[a])

        # GWB: cross-pulsar blocks, summed over correlated injectors.
        offset_b = 0
        for b in range(n_psr):
            nb = n_toas_list[b]
            gwb_block = sum(
                Gamma_per_k[k][a, b]
                * F_per_k_per_p[k][a] @ jnp.diag(S_per_k[k]) @ F_per_k_per_p[k][b].T
                for k in range(K)
            )
            C_global = C_global.at[
                offset_a:offset_a + na, offset_b:offset_b + nb
            ].add(gwb_block)
            offset_b += nb
        offset_a += na

    # Concatenate residuals.
    r_global = jnp.concatenate(residuals)

    # Dense solve.
    L = jnp.linalg.cholesky(C_global)
    alpha = jax.scipy.linalg.cho_solve((L, True), r_global)
    rCr = jnp.dot(r_global, alpha)
    logdetC = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    logL = -0.5 * rCr - 0.5 * logdetC - 0.5 * n_total * jnp.log(2.0 * jnp.pi)
    return logL


def _dense_logL(toa_data_list, timing_models, noise_models, pulsar_params,
                gwb_injector, global_params):
    """K=1 wrapper around :func:`_dense_logL_multi`.

    Preserves the single-injector signature used by existing callers.
    """
    return _dense_logL_multi(
        toa_data_list, timing_models, noise_models, pulsar_params,
        (gwb_injector,), global_params,
    )


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

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL(global_params, pulsar_params, config)
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

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL(global_params, pulsar_params, config)
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

        config_corr = PTAConfig(
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

        logL_corr = pta_logL(gp_corr, pulsar_params, config_corr)
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

        config = PTAConfig(
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
            return pta_logL(gp, pulsar_params, config)

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
            td, tm, _, pp = make_simple_pulsar(
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

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_corr = pta_logL(global_params, pulsar_params, config)
        logL_dense = _dense_logL(
            toa_data_list, timing_models, noise_models,
            pulsar_params, gwb_injector, global_params,
        )

        np.testing.assert_allclose(
            float(logL_corr), float(logL_dense), rtol=1e-8,
            err_msg="Two-tier Woodbury with red noise does not match dense solve",
        )


# ---------------------------------------------------------------------------
# Tests: multiple correlated injectors (joint outer-tier solve)
# ---------------------------------------------------------------------------


class TestMultipleCorrelatedInjectors:
    """``pta_logL`` with len(correlated_injectors) >= 2 uses one joint solve."""

    def _make_two_injector_setup(self):
        """Two HD-correlated injectors with distinct prefixes / spectra / sizes."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0

        cinj_a = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=4,
            T_span=T_span,
            orf_func=hd_orf,
            prefix="gwb_hd_",
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        cinj_b = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=3,
            T_span=T_span,
            orf_func=dipole_orf,
            prefix="gwb_dip_",
            initial_values={"log10_A": -14.8, "gamma": 3.0},
        )

        global_params = cinj_a.register_params(GlobalParams.empty())
        global_params = cinj_b.register_params(global_params)

        return (
            toa_data_list, timing_models, noise_models,
            pulsar_params, (cinj_a, cinj_b), global_params,
        )

    def test_two_injectors_matches_dense(self):
        """K=2 joint solve matches the dense brute-force reference."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, cinjs, global_params) = self._make_two_injector_setup()

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=cinjs,
        )

        logL_woodbury = float(pta_logL(global_params, pulsar_params, config))
        logL_dense = float(_dense_logL_multi(
            toa_data_list, timing_models, noise_models,
            pulsar_params, cinjs, global_params,
        ))

        np.testing.assert_allclose(
            logL_woodbury, logL_dense, rtol=1e-8,
            err_msg="K=2 joint outer-tier solve does not match dense reference",
        )

    def test_two_injectors_not_equal_to_sum_of_K1(self):
        """Joint K=2 result differs from the sum of independent K=1 results.

        The pre-fix code added K independent per-injector corrections and
        over-counted ``sum_rCr`` / ``sum_logdetC``.  This test catches a
        regression to that behavior by asserting the joint K=2 number is
        not close to the buggy "sum of K=1 results" expression.
        """
        (toa_data_list, timing_models, noise_models,
         pulsar_params, cinjs, global_params) = self._make_two_injector_setup()

        config_joint = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=cinjs,
        )
        config_a = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(cinjs[0],),
        )
        config_b = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(cinjs[1],),
        )

        logL_joint = float(pta_logL(global_params, pulsar_params, config_joint))
        logL_a = float(pta_logL(global_params, pulsar_params, config_a))
        logL_b = float(pta_logL(global_params, pulsar_params, config_b))

        # The joint K=2 result should not coincide with either K=1 result,
        # nor with their sum — those are the values the pre-fix code would
        # have produced (modulo a constant offset).
        assert abs(logL_joint - logL_a) > 1e-3, (
            f"K=2 joint result {logL_joint:.6f} ≈ K=1(inj_a) {logL_a:.6f}; "
            "joint solve appears to be ignoring the second injector"
        )
        assert abs(logL_joint - logL_b) > 1e-3, (
            f"K=2 joint result {logL_joint:.6f} ≈ K=1(inj_b) {logL_b:.6f}; "
            "joint solve appears to be ignoring the first injector"
        )
        assert abs(logL_joint - (logL_a + logL_b)) > 1e-3, (
            f"K=2 joint result {logL_joint:.6f} ≈ K=1(a) + K=1(b) "
            f"{logL_a + logL_b:.6f}; this is the pre-fix buggy behavior"
        )

    def test_single_injector_unchanged(self):
        """K=1 via the new joint code path still matches the dense reference.

        Regression guard: the joint-solve code path with a length-1 tuple
        must reduce to the original (correct) K=1 behavior.
        """
        (toa_data_list, timing_models, noise_models,
         pulsar_params, positions) = _make_multi_pulsar_setup(n_pulsars=3)

        gwb_injector = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=5,
            T_span=365.25 * 86400.0,
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb_injector.register_params(GlobalParams.empty())

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb_injector,),
        )

        logL_woodbury = float(pta_logL(global_params, pulsar_params, config))
        logL_dense = float(_dense_logL_multi(
            toa_data_list, timing_models, noise_models,
            pulsar_params, (gwb_injector,), global_params,
        ))

        np.testing.assert_allclose(logL_woodbury, logL_dense, rtol=1e-8)

    def test_grad_with_two_injectors(self):
        """``jax.grad`` of K=2 ``pta_logL`` w.r.t. global params is finite."""
        (toa_data_list, timing_models, noise_models,
         pulsar_params, cinjs, global_params) = self._make_two_injector_setup()

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=cinjs,
        )

        grad = jax.grad(pta_logL, argnums=0)(
            global_params, pulsar_params, config
        )
        # GlobalParams gradient: its ``values`` array must be all-finite.
        assert jnp.all(jnp.isfinite(grad.values)), (
            f"grad has non-finite entries: {grad.values}"
        )
        assert jnp.any(grad.values != 0.0), (
            "all gradient entries are zero — likelihood doesn't depend on "
            "global params, which is wrong for K=2"
        )
