"""Tests for the GLS fitter and ECORR noise model."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import EcorrNoise
from jaxpint.utils import woodbury_dot, woodbury_solve
from tests.helpers import make_params as _make_params_base


def _make_params(names, values, frozen_names=()):
    frozen_mask = tuple(n in frozen_names for n in names)
    return _make_params_base(names, values, frozen_mask=frozen_mask,
                             units=tuple("s" for _ in names))


# ---------------------------------------------------------------------------
# Quantization matrix construction
# ---------------------------------------------------------------------------


class TestBuildQuantizationMatrix:
    """Tests for _build_quantization_matrix."""

    def test_simple_epochs(self):
        """Two epochs with 2 TOAs each, one singleton excluded."""
        from jaxpint.bridge import _build_quantization_matrix

        # 5 TOAs: 2 in epoch A (~100s), 1 singleton (~200s), 2 in epoch B (~300s)
        times = np.array([100.0, 100.5, 200.0, 300.0, 300.3])
        masks = {"ECORR1": np.ones(5, dtype=bool)}

        U, slices = _build_quantization_matrix(times, masks)

        assert U.shape == (5, 2), f"Expected (5,2) got {U.shape}"
        # Epoch A: TOAs 0,1
        npt.assert_array_equal(U[:, 0], [1, 1, 0, 0, 0])
        # Epoch B: TOAs 3,4
        npt.assert_array_equal(U[:, 1], [0, 0, 0, 1, 1])
        assert slices["ECORR1"] == (0, 2)

    def test_no_qualifying_epochs(self):
        """All singletons -> zero-column U."""
        from jaxpint.bridge import _build_quantization_matrix

        times = np.array([100.0, 200.0, 300.0])
        masks = {"ECORR1": np.ones(3, dtype=bool)}

        U, slices = _build_quantization_matrix(times, masks)
        assert U.shape == (3, 0)
        assert slices["ECORR1"] == (0, 0)

    def test_multiple_ecorrs(self):
        """Two ECORR parameters with disjoint masks."""
        from jaxpint.bridge import _build_quantization_matrix

        # 6 TOAs: first 3 belong to ECORR1, last 3 to ECORR2
        times = np.array([100.0, 100.5, 100.8, 200.0, 200.3, 200.7])
        mask1 = np.array([True, True, True, False, False, False])
        mask2 = np.array([False, False, False, True, True, True])

        U, slices = _build_quantization_matrix(
            times, {"ECORR1": mask1, "ECORR2": mask2}
        )

        # Each ECORR should produce 1 epoch (3 TOAs within 1s)
        assert U.shape == (6, 2)
        assert slices["ECORR1"] == (0, 1)
        assert slices["ECORR2"] == (1, 2)
        npt.assert_array_equal(U[:, 0], [1, 1, 1, 0, 0, 0])
        npt.assert_array_equal(U[:, 1], [0, 0, 0, 1, 1, 1])

    def test_empty_mask(self):
        """ECORR with no matching TOAs."""
        from jaxpint.bridge import _build_quantization_matrix

        times = np.array([100.0, 100.5])
        masks = {"ECORR1": np.zeros(2, dtype=bool)}

        U, slices = _build_quantization_matrix(times, masks)
        assert U.shape == (2, 0)
        assert slices["ECORR1"] == (0, 0)


# ---------------------------------------------------------------------------
# EcorrNoise weights
# ---------------------------------------------------------------------------


class TestEcorrWeights:
    """Tests for EcorrNoise.ecorr_weights."""

    def test_single_ecorr(self):
        """Weights should be ECORR^2 for each epoch."""
        n_toas, n_epochs = 10, 3
        U = jnp.zeros((n_toas, n_epochs))

        ecorr = EcorrNoise(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 3),),
        )

        params = _make_params(("ECORR1",), [2e-6])

        weights = ecorr.ecorr_weights(params)
        expected = jnp.full(3, (2e-6) ** 2)
        npt.assert_allclose(np.array(weights), np.array(expected), rtol=1e-14)

    def test_multiple_ecorrs(self):
        """Two ECORRs with different values fill correct slices."""
        n_toas = 10
        U = jnp.zeros((n_toas, 5))  # 3 epochs for ECORR1, 2 for ECORR2

        ecorr = EcorrNoise(
            ecorr_names=("ECORR1", "ECORR2"),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 3), (3, 5)),
        )

        params = _make_params(("ECORR1", "ECORR2"), [1e-6, 3e-6])

        weights = ecorr.ecorr_weights(params)
        npt.assert_allclose(float(weights[0]), (1e-6) ** 2, rtol=1e-14)
        npt.assert_allclose(float(weights[4]), (3e-6) ** 2, rtol=1e-14)

    def test_jit_compatible(self):
        """ecorr_weights should be JIT-compilable."""
        U = jnp.zeros((5, 2))
        ecorr = EcorrNoise(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 2),),
        )
        params = _make_params(("ECORR1",), [1e-6])

        @jax.jit
        def _weights(ecorr, params):
            return ecorr.ecorr_weights(params)

        result = _weights(ecorr, params)
        assert result.shape == (2,)


# ---------------------------------------------------------------------------
# Woodbury utilities
# ---------------------------------------------------------------------------


class TestWoodburySolve:
    """Tests for woodbury_solve."""

    def test_matches_explicit_inverse(self):
        """woodbury_solve should match explicit C^{-1} B."""
        n, k, m = 20, 3, 5
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, 4)

        Ndiag = jax.random.uniform(keys[0], (n,), minval=0.1, maxval=1.0)
        U = jax.random.normal(keys[1], (n, k))
        Phidiag = jax.random.uniform(keys[2], (k,), minval=0.1, maxval=1.0)
        B = jax.random.normal(keys[3], (n, m))

        # Explicit
        C = jnp.diag(Ndiag) + U @ jnp.diag(Phidiag) @ U.T
        Cinv_B_explicit = jnp.linalg.solve(C, B)

        # Woodbury
        Cinv_B_woodbury = woodbury_solve(Ndiag, U, Phidiag, B)

        npt.assert_allclose(
            np.array(Cinv_B_woodbury), np.array(Cinv_B_explicit), rtol=1e-10
        )

    def test_jit_compatible(self):
        """woodbury_solve should be JIT-compilable."""
        n, k, m = 10, 2, 3
        Ndiag = jnp.ones(n)
        U = jnp.ones((n, k))
        Phidiag = jnp.ones(k)
        B = jnp.ones((n, m))

        result = jax.jit(woodbury_solve)(Ndiag, U, Phidiag, B)
        assert result.shape == (n, m)


# ---------------------------------------------------------------------------
# GLS chi-squared
# ---------------------------------------------------------------------------


class TestGLSChi2:
    """Tests for compute_gls_chi2."""

    def test_matches_explicit(self):
        """GLS chi2 via Woodbury matches explicit r^T C^{-1} r."""
        from jaxpint.fitter import compute_gls_chi2

        n, k = 30, 4
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, 4)

        Ndiag = jax.random.uniform(keys[0], (n,), minval=0.1, maxval=1.0)
        U = jax.random.normal(keys[1], (n, k))
        Phidiag = jax.random.uniform(keys[2], (k,), minval=0.01, maxval=0.5)
        r = jax.random.normal(keys[3], (n,)) * 1e-6

        # Explicit
        C = jnp.diag(Ndiag) + U @ jnp.diag(Phidiag) @ U.T
        chi2_explicit = r @ jnp.linalg.solve(C, r)

        chi2_gls = compute_gls_chi2(r, Ndiag, U, Phidiag)

        npt.assert_allclose(float(chi2_gls), float(chi2_explicit), rtol=1e-10)


# ---------------------------------------------------------------------------
# GLS fitter: fullcov vs augmented agree
# ---------------------------------------------------------------------------


class TestGLSSteps:
    """Tests for gls_step_fullcov and gls_step_augmented."""

    @pytest.fixture
    def synthetic_gls_problem(self):
        """Create a synthetic linear regression with correlated noise."""
        n_toas = 50
        n_free = 4
        n_epochs = 6
        key = jax.random.PRNGKey(99)
        keys = jax.random.split(key, 5)

        # Random design matrix
        M = jax.random.normal(keys[0], (n_toas, n_free)) * 1e-3
        # White noise
        Ndiag = jax.random.uniform(keys[1], (n_toas,), minval=1e-12, maxval=1e-10)
        # Quantization matrix (binary, ~8 TOAs per epoch)
        U_np = np.zeros((n_toas, n_epochs))
        for i in range(n_epochs):
            start = i * 8
            end = min(start + 8, n_toas)
            U_np[start:end, i] = 1.0
        U = jnp.array(U_np)
        # ECORR weights
        Phidiag = jax.random.uniform(keys[2], (n_epochs,), minval=1e-14, maxval=1e-12)
        # Residuals
        residuals = jax.random.normal(keys[3], (n_toas,)) * 1e-6
        threshold = 1e-14 * max(n_toas, n_free)

        return residuals, Ndiag, U, Phidiag, M, threshold

    def test_fullcov_augmented_dpars_agree(self, synthetic_gls_problem):
        """fullcov and augmented approaches give same dpars."""
        from jaxpint.fitter import gls_step_augmented, gls_step_fullcov

        residuals, Ndiag, U, Phidiag, M, threshold = synthetic_gls_problem

        dpars_fc, cov_fc, _ = gls_step_fullcov(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
        dpars_aug, cov_aug, _, _ = gls_step_augmented(
            residuals, Ndiag, U, Phidiag, M, threshold
        )

        npt.assert_allclose(
            np.array(dpars_aug), np.array(dpars_fc), rtol=1e-6,
            err_msg="Augmented and fullcov dpars should agree",
        )
        # Covariance differs slightly due to the uninformative prior in the
        # augmented approach (1e-40 on timing params).  Check relative
        # agreement to ~20% which confirms structural correctness.
        npt.assert_allclose(
            np.array(cov_aug), np.array(cov_fc), rtol=0.2,
            err_msg="Augmented and fullcov covariance should broadly agree",
        )


# ---------------------------------------------------------------------------
# GLS == WLS when no ECORR
# ---------------------------------------------------------------------------


class TestGLSReducesToWLS:
    """GLS fitter with no ECORR should match WLS fitter."""

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        """Create a simple pulsar for testing."""
        import pint.models as pm
        from pint.fitter import WLSFitter as PINTWLSFitter
        from pint.simulation import make_fake_toas_uniform

        par = """
            PSR           J0000+0000
            EPHEM         DE440
            CLK           TT(BIPM2021)
            PEPOCH        55000
            F0            100.0 1
            F1            -1e-15 1
            RAJ           00:00:00.0 1
            DECJ          00:00:00.0 1
            DM            15.0 1
            TZRMJD        55000
            TZRFRQ        1400
            TZRSITE       gbt
        """
        m = pm.get_model(io.StringIO(par))
        toas = make_fake_toas_uniform(54990, 55010, 40, m, add_noise=True)
        return m, toas

    def test_gls_matches_wls(self, synthetic_data):
        """GLSFitter with no ECORR gives same chi2 as WLSFitter."""
        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitter import GLSFitter, WLSFitter

        pint_model, toas = synthetic_data
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model)
        jax_model, noise_model, ecorr_noise = build_timing_model(pint_model)

        # WLS
        params_wls = copy.deepcopy(params)
        wls = WLSFitter(jax_model, toa_data, params_wls, noise_model=noise_model)
        chi2_wls = wls.fit_toas(maxiter=1)

        # GLS with no ECORR
        params_gls = copy.deepcopy(params)
        gls = GLSFitter(
            jax_model, toa_data, params_gls,
            noise_model=noise_model, ecorr_noise=None,
        )
        chi2_gls = gls.fit_toas(maxiter=1)

        npt.assert_allclose(
            chi2_gls, chi2_wls, rtol=1e-10,
            err_msg="GLS without ECORR should match WLS chi2",
        )


# ---------------------------------------------------------------------------
# Integration: quantization matrix matches PINT
# ---------------------------------------------------------------------------


class TestQuantizationVsPINT:
    """Compare quantization matrix against PINT's implementation."""

    def test_matches_pint_ecorr_basis(self):
        """Our quantization matrix matches PINT's get_noise_basis."""
        from pathlib import Path

        import pint.models as pm
        from pint.toa import get_TOAs

        from jaxpint.bridge import _build_quantization_matrix

        # Use PINT's example data with ECORR
        pint_data = Path(pm.__file__).parent.parent / "data" / "examples"
        par_file = pint_data / "B1855+09_NANOGrav_9yv1.gls.par"
        tim_file = pint_data / "B1855+09_NANOGrav_9yv1.tim"

        if not par_file.exists():
            pytest.skip("PINT example data not found")

        m = pm.get_model(str(par_file))
        toas = get_TOAs(str(tim_file), ephem="DE440")
        toas.compute_TDBs()

        # PINT's quantization matrix
        ecorr_comp = m.components.get("EcorrNoise")
        if ecorr_comp is None:
            pytest.skip("No EcorrNoise in test model")

        ecorr_comp.setup()
        pint_U = ecorr_comp.get_noise_basis(toas)

        # Our quantization matrix
        tdb_s = np.float64(np.asarray(toas.table["tdbld"])) * 86400.0
        ecorr_masks = {}
        for ename in sorted(ecorr_comp.ECORRs.keys()):
            param = getattr(m, ename)
            idx = param.select_toa_mask(toas)
            mask = np.zeros(toas.ntoas, dtype=bool)
            if len(idx) > 0:
                mask[idx] = True
            ecorr_masks[ename] = mask

        our_U, slices = _build_quantization_matrix(tdb_s, ecorr_masks)

        assert our_U.shape == pint_U.shape, (
            f"Shape mismatch: ours={our_U.shape}, PINT={pint_U.shape}"
        )
        npt.assert_array_equal(
            our_U, pint_U,
            err_msg="Quantization matrix does not match PINT",
        )


# Need io for StringIO in the fixture
import io
