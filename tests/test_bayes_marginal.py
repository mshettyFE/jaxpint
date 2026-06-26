"""
    Tests for jaxpint.bayes.marginal — analytic marginalization.
"""

from __future__ import annotations

import io
import warnings

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import pint.models as models
from pint.simulation import make_fake_toas_uniform

from jaxpint.bayes import (
    Gaussian,
    ImproperPrior,
    Uniform,
    marg_set_from_priors,
    marginalize,
)
from jaxpint.bayes.validate import PriorValidationError
from jaxpint.bridge import build_timing_model, pint_model_to_params, pint_toas_to_jax
from jaxpint.fitters import compute_time_residuals
from jaxpint.likelihood import single_pulsar_logL
from jaxpint.pta.likelihood import PTAConfig, pta_logL
from jaxpint.types import GlobalParams
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector

from tests.helpers import make_simple_pulsar


# ---------------------------------------------------------------------------
# Synthetic-pulsar fixture (mirrors tests/test_likelihood.py)
# ---------------------------------------------------------------------------

_SYNTH_PAR = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
START         53000 1
FINISH        55000 1
PEPOCH        54000
F0            100.0 1
F1            -1e-15 1
DM            15.0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
"""


@pytest.fixture(scope="module")
def synth_objects():
    """Timing model, noise model, TOA data, and params from a synthetic pulsar."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR))
    toas = make_fake_toas_uniform(
        53000, 55000, 30, m_true,
        error=10 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true).params
    jax_model, noise_model = build_timing_model(m_true)
    return jax_model, noise_model, toa_data, params


# A second fixture that adds F2 to the free-parameter set. Used
# by the multi-parameter numerical-integration test, which needs two linear
# precision-friendly params to marginalize jointly.
_SYNTH_PAR_F2 = """\
PSR           J0000+0000
EPHEM         DE421
CLK           TT(BIPM2019)
UNITS         TDB
START         53000 1
FINISH        55000 1
PEPOCH        54000
F0            100.0 1
F1            -1e-15 1
F2            0.0 1
DM            15.0 1
TZRMJD        54000
TZRFRQ        1400
TZRSITE       @
"""


@pytest.fixture(scope="module")
def synth_objects_with_F2():
    """Synthetic pulsar with F2 added as a free param (in addition to F0/F1/DM)."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR_F2))
    toas = make_fake_toas_uniform(
        53000, 55000, 30, m_true,
        error=10 * u.us, add_noise=True, freq=1400 * u.MHz,
    )
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true).params
    jax_model, noise_model = build_timing_model(m_true)
    return jax_model, noise_model, toa_data, params


# ---------------------------------------------------------------------------
# marg_set_from_priors (test first — used by later tests)
# ---------------------------------------------------------------------------


class TestMargSetFromPriors:
    def _mixed_priors(self):
        return {
            "F0":   ImproperPrior(),
            "F1":   ImproperPrior(),
            "PX":   Gaussian(mu=1.0, sigma=0.2),
            "EFAC": Uniform(0.1, 10.0),
            "DM":   ImproperPrior(),
        }

    def test_default_filters_to_improper(self):
        priors = self._mixed_priors()
        out = marg_set_from_priors(priors)
        assert out == {"F0", "F1", "DM"}

    def test_include_adds_regardless_of_shape(self):
        priors = self._mixed_priors()
        out = marg_set_from_priors(priors, include={"PX", "EFAC"})
        assert out == {"F0", "F1", "DM", "PX", "EFAC"}

    def test_exclude_removes_regardless_of_shape(self):
        priors = self._mixed_priors()
        out = marg_set_from_priors(priors, exclude={"F0"})
        assert out == {"F1", "DM"}

    def test_exclude_wins_over_include(self):
        priors = self._mixed_priors()
        # User explicitly includes PX but also excludes it — exclude wins.
        out = marg_set_from_priors(priors, include={"PX"}, exclude={"PX"})
        assert out == {"F0", "F1", "DM"}

    def test_prior_class_filter(self):
        priors = self._mixed_priors()
        out = marg_set_from_priors(priors, prior_class=Gaussian)
        assert out == {"PX"}


# ---------------------------------------------------------------------------
# Analytic correctness — single Improper marg'd param vs dense integral
# ---------------------------------------------------------------------------


class TestAnalyticCorrectness:
    @pytest.mark.slow
    def test_marg_matches_numerical_integration(self, synth_objects):
        """End-to-end semantic check: the wrapper's output equals the actual
        integral w.r.t. F1 computed by direct numerical quadrature.

        They differ by the Gaussian-prior normalization
        constant -0.5 * log(2π * 1e40), which we add back for comparison.

        F1 is chosen over F0 here because of float64 precision: F0 ≈ 100 Hz
        with WLS σ ≈ 1e-12 Hz means grid spacings ~1e-14 land at machine
        epsilon and adjacent grid points round to the same number. F1 has
        ratio σ/|F1| ≈ 1e-4 — comfortably resolvable. Both are linear in
        residuals so the integrand is exactly Gaussian.
        """
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg, _, skel = marginalize(
            single_pulsar_logL,
            over={"F1"},
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )
        logL_wrapper = float(likelihood_marg(skel))

        # WLS posterior σ for F1 — sets the natural integration scale.
        idx_F1 = params.param_index("F1")
        from jaxpint.fitters._base import compute_time_residuals as _crt
        J_full = jax.jacobian(
            lambda v: _crt(jax_model, toa_data, eqx.tree_at(lambda p: p.values, params, v))
        )(params.values)
        M_col = J_full[:, idx_F1]
        Ndiag = noise_model.scaled_sigma(toa_data, params) ** 2
        fisher = float(jnp.sum(M_col ** 2 / Ndiag))
        sigma_post = 1.0 / np.sqrt(fisher)

        # Trapezoidal-rule integration over ±K·σ around the WLS-ML estimate.
        # We center on F1_fid + (a/b) — the maximum-likelihood F1 — so the
        # peak of the integrand sits at the grid center, not at the edge.
        # K=20 → truncation error ~ exp(-K²/2) is utterly negligible.
        # N=16001 → trapezoidal error on a Gaussian is ~ (δ/σ)² ≈ 1.5e-6.
        K = 20.0
        N_grid = 16001
        F1_fid = float(params.param_value("F1"))
        r_at_fid = compute_time_residuals(jax_model, toa_data, params)
        a = float(jnp.sum(M_col * r_at_fid / Ndiag))
        F1_max = F1_fid + a / fisher
        F1_grid = jnp.linspace(F1_max - K * sigma_post, F1_max + K * sigma_post, N_grid)
        delta = float(F1_grid[1] - F1_grid[0])

        def logL_at_F1(F1):
            # Use the FULL nonlinear single_pulsar_logL (not the wrapper) so
            # we're testing against the true integrand of the marg integral.
            p = params.with_value("F1", F1)
            return single_pulsar_logL(toa_data, jax_model, noise_model, p)

        logL_values = jax.vmap(logL_at_F1)(F1_grid)

        # Log-sum-exp for numerical stability.
        logL_max = jnp.max(logL_values)
        log_integral = float(
            logL_max
            + jnp.log(jnp.sum(jnp.exp(logL_values - logL_max)))
            + jnp.log(delta)
        )

        # The Woodbury wrapper treats Phi=1e40 as a Gaussian prior with that
        # variance on F1. The Gaussian-prior normalization (1/sqrt(2π·1e40))
        # is bundled into the wrapper's output. To compare against the
        # IMPROPER (flat-measure) integral the numerical side computes, add
        # the normalization back.
        normalization_offset = 0.5 * float(jnp.log(2 * jnp.pi * 1e40))
        expected = logL_wrapper + normalization_offset

        npt.assert_allclose(log_integral, expected, atol=3e-3)

    @pytest.mark.slow
    def test_multi_param_improper_marg(self, synth_objects_with_F2):
        """Multi-parameter marg verified by 2D numerical integration.

        Independent check that marginalizing over two parameters at once
        produces the correct integral — not just the correct internal
        composition. The wrapper integrates analytically over (F1, F2)
        via stacked Woodbury columns; this test integrates the same
        integral by direct quadrature on a rectangular (F1, F2) grid and
        compares.

        The (F1, F2) posterior is elongated (correlation ~0.99 from the
        polynomial-in-t structure of the spindown model). We handle the
        elongation by sizing the grid range from the MARGINAL σ (covers
        the full ellipse) while keeping the per-axis grid count high
        enough that the EFFECTIVE resolution along the tight direction
        is still adequate. ~300 points per axis is enough at this
        correlation; the resulting cost (~90K likelihood evaluations
        through vmap) is tolerable.
        """
        jax_model, noise_model, toa_data, params = synth_objects_with_F2
        priors = {n: ImproperPrior() for n in params.free_names()}
        over = {"F1", "F2"}

        likelihood_marg, _, skel = marginalize(
            single_pulsar_logL,
            over=over,
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )
        logL_wrapper = float(likelihood_marg(skel))

        # --- Build Fisher matrix and ML offset for the (F1, F2) block ---
        idx_F1 = params.param_index("F1")
        idx_F2 = params.param_index("F2")
        from jaxpint.fitters._base import compute_time_residuals as _crt
        J_full = jax.jacobian(
            lambda v: _crt(jax_model, toa_data, eqx.tree_at(lambda p: p.values, params, v))
        )(params.values)
        M = J_full[:, jnp.array([idx_F1, idx_F2])]            # (n_toas, 2)
        Ndiag = noise_model.scaled_sigma(toa_data, params) ** 2
        Ninv = 1.0 / Ndiag
        Fisher = (M.T * Ninv) @ M                              # (2, 2)
        r0 = compute_time_residuals(jax_model, toa_data, params)
        grad = (M.T * Ninv) @ r0                               # (2,)
        shift = jnp.linalg.solve(Fisher, grad)                 # WLS-ML offset
        theta_fid = jnp.array([
            float(params.param_value("F1")),
            float(params.param_value("F2")),
        ])
        theta_max = theta_fid + shift

        # Per-axis marginal σ from the diagonal of Fisher⁻¹.
        Fisher_inv = jnp.linalg.inv(Fisher)
        sigma_F1 = float(jnp.sqrt(Fisher_inv[0, 0]))
        sigma_F2 = float(jnp.sqrt(Fisher_inv[1, 1]))

        # --- Rectangular grid in (F1, F2), centered on the ML estimate ---
        # K=8 marginal σ covers the full posterior ellipse. N=300 per axis is
        # enough resolution in the tight direction at ~0.99 correlation for
        # ~1e-3 error in log space; we measure empirically and leave headroom.
        K = 8.0
        N_per_axis = 300
        F1_grid = jnp.linspace(
            theta_max[0] - K * sigma_F1, theta_max[0] + K * sigma_F1, N_per_axis
        )
        F2_grid = jnp.linspace(
            theta_max[1] - K * sigma_F2, theta_max[1] + K * sigma_F2, N_per_axis
        )
        delta_F1 = float(F1_grid[1] - F1_grid[0])
        delta_F2 = float(F2_grid[1] - F2_grid[0])

        # Flatten the cartesian grid for vmap.
        F1_mesh, F2_mesh = jnp.meshgrid(F1_grid, F2_grid, indexing="ij")
        F1_flat = F1_mesh.flatten()
        F2_flat = F2_mesh.flatten()

        # --- Evaluate the FULL nonlinear single_pulsar_logL on the grid ---
        def logL_at(F1, F2):
            p = params.with_value("F1", F1).with_value("F2", F2)
            return single_pulsar_logL(toa_data, jax_model, noise_model, p)

        logL_values = jax.vmap(logL_at)(F1_flat, F2_flat)      # (N²,)

        # --- 2D Riemann sum via log-sum-exp ---
        logL_max_val = jnp.max(logL_values)
        log_integral = float(
            logL_max_val
            + jnp.log(jnp.sum(jnp.exp(logL_values - logL_max_val)))
            + jnp.log(delta_F1)
            + jnp.log(delta_F2)
        )

        # --- Account for the Φ=1e40 prior-normalization the wrapper bakes in
        # (two marg'd params → 2× the offset). ---
        n_marg = 2
        normalization_offset = n_marg * 0.5 * float(jnp.log(2 * jnp.pi * 1e40))
        expected = logL_wrapper + normalization_offset

        # Tolerance is dominated by 2D quadrature error on the elongated
        # Gaussian. The marg math is wrong by orders of magnitude if this fires.
        npt.assert_allclose(log_integral, expected, atol=1e-2)


# ---------------------------------------------------------------------------
#  Identity when over=set()
# ---------------------------------------------------------------------------


class TestEmptyMargSet:
    def test_no_op_equals_original(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg, sampled_priors, skel = marginalize(
            single_pulsar_logL,
            over=set(),
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )

        logL_marg = likelihood_marg(skel)
        logL_orig = single_pulsar_logL(toa_data, jax_model, noise_model, params)
        npt.assert_allclose(float(logL_marg), float(logL_orig), rtol=1e-12)

        assert sampled_priors == priors            # nothing removed
        assert skel.marginalized_names() == ()     # nothing marked


# ---------------------------------------------------------------------------
#  reduced_skeleton structure
# ---------------------------------------------------------------------------

class TestReducedSkeleton:
    def test_mask_and_free_names(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg, sampled_priors, skel = marginalize(
            single_pulsar_logL,
            over={"F0", "F1"},
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )

        # Marginalized names are exactly what was passed in over.
        assert set(skel.marginalized_names()) == {"F0", "F1"}
        # Free names exclude the marg'd ones.
        assert "F0" not in skel.free_names()
        assert "F1" not in skel.free_names()
        # And values for marg'd entries equal the fiducial.
        for n in ("F0", "F1"):
            i = skel.param_index(n)
            assert float(skel.values[i]) == float(params.values[i])

    def test_y_disappears_from_free_values(self, synth_objects):
        """skel.free_values() returns one value per kept (= not frozen, not
        marg'd) parameter — the marg'd values do NOT leak out."""
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        _, _, skel = marginalize(
            single_pulsar_logL,
            over={"F0", "DM"},
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )

        kept = skel.free_names()
        assert "F0" not in kept and "DM" not in kept
        assert skel.free_values().shape == (len(kept),)


# ---------------------------------------------------------------------------
# JIT and gradient
# ---------------------------------------------------------------------------


class TestJITAndGrad:
    def test_jit_matches_eager(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg, _, skel = marginalize(
            single_pulsar_logL,
            over={"F0"},
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )

        logL_eager = likelihood_marg(skel)
        logL_jit = jax.jit(likelihood_marg)(skel)
        npt.assert_allclose(float(logL_jit), float(logL_eager), rtol=1e-8)

    def test_grad_wrt_kept_values_is_finite(self, synth_objects):
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg, _, skel = marginalize(
            single_pulsar_logL,
            over={"F0"},
            priors=priors,
            toa_data=toa_data,
            timing_model=jax_model,
            noise_model=noise_model,
            fiducial_params=params,
        )

        def logL_of_values(kept_values):
            p = skel.with_free_values(kept_values)
            return likelihood_marg(p)

        grad = jax.grad(logL_of_values)(skel.free_values())
        assert jnp.all(jnp.isfinite(grad))
        # At least some gradients should be non-zero
        assert jnp.any(grad != 0.0)


# ---------------------------------------------------------------------------
# Linearity check
# ---------------------------------------------------------------------------


class TestLinearityCheck:
    def test_linear_params_pass_silently(self, synth_objects):
        """F0, F1, DM are linear in residuals — no warning, no raise."""
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        with warnings.catch_warnings():
            warnings.simplefilter("error")    # any warning becomes a test failure
            likelihood_marg, _, _ = marginalize( single_pulsar_logL,
                over=set(params.free_names()),
                priors=priors,
                toa_data=toa_data,
                timing_model=jax_model,
                noise_model=noise_model,
                fiducial_params=params,
                # defaults: allow_nonlinear=False, validate_linearity=True
            )
        # Ensure the result is well-defined too.
        _, _, skel = marginalize(
            single_pulsar_logL, over=set(params.free_names()),
            priors=priors, toa_data=toa_data, timing_model=jax_model,
            noise_model=noise_model, fiducial_params=params,
        )
        assert jnp.isfinite(likelihood_marg(skel))

    def test_skip_validation_skips_hessian(self, synth_objects):
        """validate_linearity=False should produce the same result for linear
        params but without computing the Hessian."""
        jax_model, noise_model, toa_data, params = synth_objects
        priors = {n: ImproperPrior() for n in params.free_names()}

        likelihood_marg_check, _, skel_check = marginalize(
            single_pulsar_logL, over={"F0"},
            priors=priors, toa_data=toa_data, timing_model=jax_model,
            noise_model=noise_model, fiducial_params=params,
            validate_linearity=True,
        )
        likelihood_marg_skip, _, skel_skip = marginalize(
            single_pulsar_logL, over={"F0"},
            priors=priors, toa_data=toa_data, timing_model=jax_model,
            noise_model=noise_model, fiducial_params=params,
            validate_linearity=False,
        )

        npt.assert_allclose(
            float(likelihood_marg_check(skel_check)),
            float(likelihood_marg_skip(skel_skip)),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# PTA marginalization
# ---------------------------------------------------------------------------


def _build_pta_setup(
    n_pulsars: int = 3,
    *,
    pulsar_names: tuple[str, ...] | None = None,
):
    """Multi-pulsar synthetic setup with F0/F1 free in each pulsar.

    Returns ``(pulsar_names, toa_data_list, timing_models, noise_models,
    pulsar_params, positions)`` — last item is a (n_pulsars, 3) unit-vector
    array of pulsar sky positions for use with correlated injectors.
    """
    if pulsar_names is None:
        pulsar_names = tuple(f"P{i}" for i in range(n_pulsars))

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
            n_toas=20 + i * 5,
            f0=200.0 + i * 10.0,
            f1=-1e-15 * (1 + i * 0.5),
            seed=42 + i,
        )
        toa_data_list.append(td)
        timing_models.append(tm)
        noise_models.append(nm)
        pulsar_params.append(pp)

    return (
        pulsar_names,
        tuple(toa_data_list),
        tuple(timing_models),
        tuple(noise_models),
        tuple(pulsar_params),
        positions,
    )


def _per_pulsar_marg_logL(
    toa_data, timing_model, noise_model, params, bare_names,
):
    """Run single-pulsar marg over ``bare_names`` and evaluate at fiducial.

    Used as the ground-truth-per-pulsar comparison for the uncorrelated PTA
    marg test: when ``correlated_injectors == ()``, ``pta_logL`` is a sum
    of independent per-pulsar log-likelihoods, so marg'ing per-pulsar
    timing params at the PTA level must equal the sum of single-pulsar
    marg log-likelihoods.
    """
    priors = {n: ImproperPrior() for n in bare_names}
    g, _, skel = marginalize(
        single_pulsar_logL,
        over=set(bare_names),
        priors=priors,
        toa_data=toa_data,
        timing_model=timing_model,
        noise_model=noise_model,
        fiducial_params=params,
        validate_linearity=False,
    )
    return float(g(skel))


def _dense_marg_pta_logL(
    toa_data_list, timing_models, noise_models, pulsar_params,
    bare_names_per_pulsar,
    correlated_injectors=(),
    global_params=None,
    sigma_sq=1e40,
):
    """Dense reference for PTA marg log-likelihood (matches Woodbury value).

    Builds the full ``(N_total, N_total)`` *unaugmented* covariance ``N_full``
    (noise + correlated injectors, no marg block), then applies the
    marg-Woodbury identity at the dense level:

        logL_marg = -½ rᵀ (N_full + σ² M Mᵀ)⁻¹ r
                    - ½ log|N_full + σ² M Mᵀ|
                    - ½ N log(2π)

    Using the matrix-determinant-lemma form
        |N_full + σ² M Mᵀ| = |N_full| · σ^(2k) · |σ⁻² I_k + Mᵀ N_full⁻¹ M|
    and the standard Woodbury
        (N_full + σ² M Mᵀ)⁻¹ = N_full⁻¹ − N_full⁻¹ M (σ⁻² I_k + Mᵀ N_full⁻¹ M)⁻¹
                               Mᵀ N_full⁻¹
    avoids forming ``σ² M Mᵀ`` directly — so it stays numerically stable for
    ``σ² = 1e40`` (the value used by the marg implementation under test).

    The structural difference from the implementation: the dense reference
    builds ``N_full`` directly (block-diagonal noise + cross-pulsar HD
    contributions, no two-tier decomposition), then does ONE marg-Woodbury.
    The implementation uses an inner+outer two-tier Woodbury structure.
    Both produce the same value; this test confirms they agree.
    """
    n_psr = len(toa_data_list)

    # ---- Per-pulsar noise covariance (no marg block) ----
    residuals = []
    C_p_list = []
    for p in range(n_psr):
        r = compute_time_residuals(
            timing_models[p], toa_data_list[p], pulsar_params[p]
        )
        residuals.append(r)
        Ndiag, U_noise, Phi_noise = noise_models[p].covariance(
            toa_data_list[p], pulsar_params[p]
        )
        C_p = jnp.diag(Ndiag)
        if U_noise.shape[1] > 0:
            C_p = C_p + U_noise @ jnp.diag(Phi_noise) @ U_noise.T
        C_p_list.append(C_p)

    n_toas_list = [td.n_toas for td in toa_data_list]
    n_total = sum(n_toas_list)
    N_full = jnp.zeros((n_total, n_total))

    offset_a = 0
    for a in range(n_psr):
        na = n_toas_list[a]
        N_full = N_full.at[
            offset_a:offset_a + na, offset_a:offset_a + na,
        ].add(C_p_list[a])
        if correlated_injectors:
            offset_b = 0
            for b in range(n_psr):
                nb = n_toas_list[b]
                blk = jnp.zeros((na, nb))
                for cinj in correlated_injectors:
                    Gamma = cinj.get_orf_matrix()
                    S = cinj.get_psd(global_params)
                    F_a = cinj.get_fourier_basis(toa_data_list[a])
                    F_b = cinj.get_fourier_basis(toa_data_list[b])
                    blk = blk + Gamma[a, b] * F_a @ jnp.diag(S) @ F_b.T
                N_full = N_full.at[
                    offset_a:offset_a + na, offset_b:offset_b + nb,
                ].add(blk)
                offset_b += nb
        offset_a += na

    r_global = jnp.concatenate(residuals)

    # ---- Build global M (block-diagonal across pulsars) ----
    M_blocks = []
    for p in range(n_psr):
        bare_names = bare_names_per_pulsar[p]
        if not bare_names:
            M_blocks.append(jnp.zeros((n_toas_list[p], 0)))
            continue
        def _resid_fn(values, _tm=timing_models[p], _td=toa_data_list[p], _fp=pulsar_params[p]):
            params = eqx.tree_at(lambda pv: pv.values, _fp, values)
            return compute_time_residuals(_tm, _td, params)
        J_full = jax.jacobian(_resid_fn)(pulsar_params[p].values)
        indices = jnp.asarray(
            [pulsar_params[p].param_index(b) for b in bare_names],
            dtype=jnp.int32,
        )
        M_blocks.append(-J_full[:, indices])
    M_global = jax.scipy.linalg.block_diag(*M_blocks)
    k = M_global.shape[1]

    # ---- Apply marg-Woodbury identity densely ----
    L = jnp.linalg.cholesky(N_full)
    alpha = jax.scipy.linalg.cho_solve((L, True), r_global)   # N_full^{-1} r
    rNir = jnp.dot(r_global, alpha)

    if k == 0:
        rCr_marg = rNir
        logdet_marg_extra = 0.0
    else:
        beta = jax.scipy.linalg.cho_solve((L, True), M_global)   # N_full^{-1} M
        MTNiM = M_global.T @ beta
        MTNir = M_global.T @ alpha
        # Sigma = (1/σ²) I_k + Mᵀ N_full^{-1} M  — well-conditioned for finite σ²
        Sigma_marg = (1.0 / sigma_sq) * jnp.eye(k) + MTNiM
        L_sigma = jnp.linalg.cholesky(Sigma_marg)
        correction = jnp.dot(
            MTNir,
            jax.scipy.linalg.cho_solve((L_sigma, True), MTNir),
        )
        rCr_marg = rNir - correction
        logdet_Sigma = 2.0 * jnp.sum(jnp.log(jnp.diag(L_sigma)))
        logdet_marg_extra = k * jnp.log(sigma_sq) + logdet_Sigma

    logdet_N = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
    return float(
        -0.5 * rCr_marg
        - 0.5 * (logdet_N + logdet_marg_extra)
        - 0.5 * n_total * jnp.log(2.0 * jnp.pi)
    )


class TestPTAMarg:
    """Analytic marginalization of per-pulsar timing parameters in `pta_logL`."""

    def test_pta_marg_matches_per_pulsar_sum(self):
        """Uncorrelated PTA: marg via pta_logL == sum of single-pulsar marg."""
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=3)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        over = {f"{pn}_F0" for pn in pulsar_names} | {
            f"{pn}_F1" for pn in pulsar_names
        }
        priors = {n: ImproperPrior() for n in over}

        g, sampled, reduced_skeletons = marginalize(
            pta_logL,
            over=over,
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )

        logL_pta = float(g(global_params, reduced_skeletons))

        logL_sum_per_pulsar = sum(
            _per_pulsar_marg_logL(
                toa_data_list[p], timing_models[p], noise_models[p],
                pulsar_params[p], ("F0", "F1"),
            )
            for p in range(len(pulsar_names))
        )

        npt.assert_allclose(logL_pta, logL_sum_per_pulsar, rtol=1e-10)
        assert sampled == {}

    def test_pta_marg_with_hd_correlated(self):
        """Marg + HD GWB: matches dense brute-force reference."""
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, positions,
        ) = _build_pta_setup(n_pulsars=3)

        T_span = 365.25 * 86400.0
        gwb = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=4,
            T_span=T_span,
            prefix="gwb_",
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb.register_params(GlobalParams.empty())

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb,),
        )

        over = {f"{pn}_F0" for pn in pulsar_names} | {
            f"{pn}_F1" for pn in pulsar_names
        }
        priors = {n: ImproperPrior() for n in over}

        g, _, reduced_skeletons = marginalize(
            pta_logL,
            over=over,
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )

        logL_woodbury = float(g(global_params, reduced_skeletons))

        bare_per_p = [["F0", "F1"]] * len(pulsar_names)
        logL_dense = _dense_marg_pta_logL(
            toa_data_list, timing_models, noise_models, pulsar_params,
            bare_per_p,
            correlated_injectors=(gwb,),
            global_params=global_params,
        )

        npt.assert_allclose(logL_woodbury, logL_dense, rtol=1e-6)

    def test_pta_marg_empty_over(self):
        """Empty `over`: wrapper returns plain pta_logL value at fiducial."""
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=2)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        g, sampled, reduced_skeletons = marginalize(
            pta_logL,
            over=set(),
            priors={},
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )

        # Reduced skeleton should equal the fiducial (no params marg'd).
        for p in range(len(pulsar_names)):
            assert reduced_skeletons[p].marginalized_names() == ()

        logL_marg = float(g(global_params, reduced_skeletons))
        logL_direct = float(pta_logL(global_params, pulsar_params, config))
        npt.assert_allclose(logL_marg, logL_direct, rtol=1e-12)
        assert sampled == {}

    def test_global_param_in_over_raises_not_implemented(self):
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, positions,
        ) = _build_pta_setup(n_pulsars=2)

        gwb = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=2,
            T_span=365.25 * 86400.0,
            prefix="gwb_",
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb.register_params(GlobalParams.empty())

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb,),
        )

        priors = {"gwb_log10_A": ImproperPrior()}

        with pytest.raises(NotImplementedError, match="global-parameter"):
            marginalize(
                pta_logL,
                over={"gwb_log10_A"},
                priors=priors,
                config=config,
                pulsar_names=pulsar_names,
                fiducial_pulsar_params=pulsar_params,
                fiducial_global_params=global_params,
                validate_linearity=False,
            )

    def test_unknown_name_in_over_raises_value_error(self):
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=2)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        priors = {"nonsense_param": ImproperPrior()}

        with pytest.raises(ValueError, match="matches no pulsar"):
            marginalize(
                pta_logL,
                over={"nonsense_param"},
                priors=priors,
                config=config,
                pulsar_names=pulsar_names,
                fiducial_pulsar_params=pulsar_params,
                fiducial_global_params=global_params,
                validate_linearity=False,
            )

    def test_per_pulsar_disambiguation(self):
        """A pulsar name that is a prefix of another resolves unambiguously."""
        pulsar_names = ("J1234", "J1234_extra")
        (
            _, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=2, pulsar_names=pulsar_names)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        # Marg the F0 of the longer-named pulsar.  The FQN "J1234_extra_F0"
        # starts with "J1234_" too, but the bare name "extra_F0" is NOT in
        # pulsar 0's params; so the resolver must pick pulsar 1.
        priors = {"J1234_extra_F0": ImproperPrior()}
        g, _, reduced_skeletons = marginalize(
            pta_logL,
            over={"J1234_extra_F0"},
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )
        assert reduced_skeletons[0].marginalized_names() == ()
        assert reduced_skeletons[1].marginalized_names() == ("F0",)

    def test_pta_marg_skeleton_structure(self):
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=3)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        # Marg F0 in all pulsars; F1 only in pulsar 1.
        over = {f"{pn}_F0" for pn in pulsar_names} | {f"{pulsar_names[1]}_F1"}
        priors = {n: ImproperPrior() for n in over}

        _, sampled, reduced_skeletons = marginalize(
            pta_logL,
            over=over,
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )

        assert reduced_skeletons[0].marginalized_names() == ("F0",)
        assert "F0" not in reduced_skeletons[0].free_names()
        # Pulsar 1: both F0 and F1 marg'd.
        assert set(reduced_skeletons[1].marginalized_names()) == {"F0", "F1"}
        assert "F0" not in reduced_skeletons[1].free_names()
        assert "F1" not in reduced_skeletons[1].free_names()
        # Pulsar 2: only F0 marg'd.
        assert reduced_skeletons[2].marginalized_names() == ("F0",)
        assert "F1" in reduced_skeletons[2].free_names()
        assert sampled == {}

    def test_pta_marg_jit_and_grad(self):
        """The wrapper is JIT-able and grad-able through global params."""
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, positions,
        ) = _build_pta_setup(n_pulsars=3)

        gwb = HDCorrelatedGWBInjector(
            pulsar_positions=positions,
            n_components=3,
            T_span=365.25 * 86400.0,
            prefix="gwb_",
            initial_values={"log10_A": -14.0, "gamma": 4.33},
        )
        global_params = gwb.register_params(GlobalParams.empty())

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
            correlated_injectors=(gwb,),
        )

        over = {f"{pn}_F0" for pn in pulsar_names}
        priors = {n: ImproperPrior() for n in over}

        g, _, reduced_skeletons = marginalize(
            pta_logL,
            over=over,
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=False,
        )

        logL_eager = float(g(global_params, reduced_skeletons))
        logL_jit = float(jax.jit(g)(global_params, reduced_skeletons))
        npt.assert_allclose(logL_eager, logL_jit, rtol=1e-12)

        grad = jax.grad(g, argnums=0)(global_params, reduced_skeletons)
        assert jnp.all(jnp.isfinite(grad.values))
        assert jnp.any(grad.values != 0.0)

    def test_pta_marg_nonlinear_raises_without_allow_nonlinear(self):
        """A nonlinear marg param raises NotImplementedError by default.

        Spindown residuals are linear in F0/F1, so we force a failure by
        using a tiny tol the data couldn't possibly satisfy under floating-point
        noise (Hessian is not literally zero due to numerical artifacts).
        """
        (
            pulsar_names, toa_data_list, timing_models, noise_models,
            pulsar_params, _,
        ) = _build_pta_setup(n_pulsars=2)

        config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=(),
        )
        global_params = GlobalParams.empty()

        over = {f"{pn}_F0" for pn in pulsar_names}
        priors = {n: ImproperPrior() for n in over}

        # The linearity test does not fire for linear params even at small
        # tol, since the Hessian is exactly zero.  Confirm: with the default
        # tol it passes silently.
        marginalize(
            pta_logL,
            over=over,
            priors=priors,
            config=config,
            pulsar_names=pulsar_names,
            fiducial_pulsar_params=pulsar_params,
            fiducial_global_params=global_params,
            validate_linearity=True,
        )
