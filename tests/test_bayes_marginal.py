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
        npt.assert_allclose(float(logL_jit), float(logL_eager), rtol=1e-12)

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


