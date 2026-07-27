"""Tests for the GLS fitter and ECORR noise model."""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
# Need io for StringIO in the fixture
import io

from jaxpint.noise import EcorrNoise
from jaxpint.utils import woodbury_solve
from tests.helpers import make_params_with_frozen_names


# ---------------------------------------------------------------------------
# Quantization matrix construction
# ---------------------------------------------------------------------------


class TestBuildQuantizationMatrix:
    """Tests for build_quantization_matrix."""

    def test_simple_epochs(self):
        """Two epochs with 2 TOAs each, one singleton excluded."""
        from jaxpint.utils import build_quantization_matrix

        # 5 TOAs: 2 in epoch A (~100s), 1 singleton (~200s), 2 in epoch B (~300s)
        times = np.array([100.0, 100.5, 200.0, 300.0, 300.3])
        masks = {"ECORR1": np.ones(5, dtype=bool)}

        U, slices = build_quantization_matrix(times, masks)

        assert U.shape == (5, 2), f"Expected (5,2) got {U.shape}"
        # Epoch A: TOAs 0,1
        npt.assert_array_equal(U[:, 0], [1, 1, 0, 0, 0])
        # Epoch B: TOAs 3,4
        npt.assert_array_equal(U[:, 1], [0, 0, 0, 1, 1])
        assert slices["ECORR1"] == (0, 2)

    def test_no_qualifying_epochs(self):
        """All singletons -> zero-column U."""
        from jaxpint.utils import build_quantization_matrix

        times = np.array([100.0, 200.0, 300.0])
        masks = {"ECORR1": np.ones(3, dtype=bool)}

        U, slices = build_quantization_matrix(times, masks)
        assert U.shape == (3, 0)
        assert slices["ECORR1"] == (0, 0)

    def test_multiple_ecorrs(self):
        """Two ECORR parameters with disjoint masks."""
        from jaxpint.utils import build_quantization_matrix

        # 6 TOAs: first 3 belong to ECORR1, last 3 to ECORR2
        times = np.array([100.0, 100.5, 100.8, 200.0, 200.3, 200.7])
        mask1 = np.array([True, True, True, False, False, False])
        mask2 = np.array([False, False, False, True, True, True])

        U, slices = build_quantization_matrix(
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
        from jaxpint.utils import build_quantization_matrix

        times = np.array([100.0, 100.5])
        masks = {"ECORR1": np.zeros(2, dtype=bool)}

        U, slices = build_quantization_matrix(times, masks)
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

        ecorr = EcorrNoise.from_dense(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 3),),
        )

        params = make_params_with_frozen_names(("ECORR1",), [2e-6])

        weights = ecorr.ecorr_weights(params)
        expected = jnp.full(3, (2e-6) ** 2)
        npt.assert_allclose(np.array(weights), np.array(expected), rtol=1e-14)

    def test_multiple_ecorrs(self):
        """Two ECORRs with different values fill correct slices."""
        n_toas = 10
        U = jnp.zeros((n_toas, 5))  # 3 epochs for ECORR1, 2 for ECORR2

        ecorr = EcorrNoise.from_dense(
            ecorr_names=("ECORR1", "ECORR2"),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 3), (3, 5)),
        )

        params = make_params_with_frozen_names(("ECORR1", "ECORR2"), [1e-6, 3e-6])

        weights = ecorr.ecorr_weights(params)
        npt.assert_allclose(float(weights[0]), (1e-6) ** 2, rtol=1e-14)
        npt.assert_allclose(float(weights[4]), (3e-6) ** 2, rtol=1e-14)

    def test_jit_compatible(self):
        """ecorr_weights should be JIT-compilable."""
        U = jnp.zeros((5, 2))
        ecorr = EcorrNoise.from_dense(
            ecorr_names=("ECORR1",),
            quantization_matrix=U,
            ecorr_epoch_slices=((0, 2),),
        )
        params = make_params_with_frozen_names(("ECORR1",), [1e-6])

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
    """Tests for compute_chi2_cov."""

    def test_matches_explicit(self):
        """GLS chi2 via Woodbury matches explicit r^T C^{-1} r."""
        from jaxpint.fitters import compute_chi2_cov

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

        chi2_gls = compute_chi2_cov(r, Ndiag, U, Phidiag)

        npt.assert_allclose(float(chi2_gls), float(chi2_explicit), rtol=1e-10)


# ---------------------------------------------------------------------------
# GLS fitter: fullcov vs augmented agree
# ---------------------------------------------------------------------------


class TestGLSSteps:
    """Tests for lstsq_step_fullcov and lstsq_step_augmented."""

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
        from jaxpint.fitters._base import lstsq_step_augmented, lstsq_step_fullcov

        residuals, Ndiag, U, Phidiag, M, threshold = synthetic_gls_problem

        dpars_fc, cov_fc, _ = lstsq_step_fullcov(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
        dpars_aug, cov_aug, _, _ = lstsq_step_augmented(
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
    """GLS fitter with no ECORR should match WLS fitter.
    """

    @pytest.fixture(scope="class")
    def synthetic_data(self):
        """Create a simple pulsar for testing."""
        import jax

        import jaxpint.par as jpar
        from jaxpint import build_model
        from jaxpint.simulation import make_fake_toas_uniform

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
        par_result = jpar.get_model(io.StringIO(par))
        toa_data = make_fake_toas_uniform(
            54990.0, 55010.0, 40, par_result,
            obs="gbt", freq_mhz=1400.0, error_us=1.0,
            add_noise=True, key=jax.random.PRNGKey(3),
        )
        jax_model, noise_model = build_model(par_result, toa_data)
        return jax_model, noise_model, toa_data, par_result.params

    @pytest.mark.slow
    def test_gls_matches_wls(self, synthetic_data):
        """GLSFitter with no ECORR gives same chi2 as WLSFitter."""
        from jaxpint.fitters import GLSFitter, WLSFitter

        jax_model, noise_model, toa_data, params = synthetic_data

        # WLS
        params_wls = copy.deepcopy(params)
        wls = WLSFitter(jax_model, toa_data, params_wls, noise_model=noise_model)
        result_wls = wls.fit_toas(maxiter=1)

        # GLS with no ECORR
        params_gls = copy.deepcopy(params)
        gls = GLSFitter(
            jax_model, toa_data, params_gls,
            noise_model=noise_model,
        )
        result_gls = gls.fit_toas(maxiter=1)

        npt.assert_allclose(
            result_gls.chi2, result_wls.chi2, rtol=1e-10,
            err_msg="GLS without ECORR should match WLS chi2",
        )


# ---------------------------------------------------------------------------
# Integration: quantization matrix matches PINT
# ---------------------------------------------------------------------------


class TestQuantizationVsPINT:
    """Compare quantization matrix against PINT's implementation."""

    @pytest.mark.slow
    def test_matches_pint_ecorr_basis(self):
        """Our quantization matrix matches PINT's get_noise_basis."""
        from pathlib import Path

        import pint.models as pm
        from pint.toa import get_TOAs

        from jaxpint.utils import build_quantization_matrix

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

        our_U, slices = build_quantization_matrix(tdb_s, ecorr_masks)

        assert our_U.shape == pint_U.shape, (
            f"Shape mismatch: ours={our_U.shape}, PINT={pint_U.shape}"
        )
        npt.assert_array_equal(
            our_U, pint_U,
            err_msg="Quantization matrix does not match PINT",
        )



# ---------------------------------------------------------------------------
# Full GLS fit vs PINT (covariance parity)
# ---------------------------------------------------------------------------


class TestGLSFitVsPINT:
    """Covariance parity vs PINT's GLSFitter on B1855+09 9yv1.

    The dataset carries EFAC/EQUAD/ECORR and power-law red noise, so this
    exercises the full Woodbury/augmented GLS covariance -- the path that
    ``TestGLSReducesToWLS`` (no ECORR) cannot reach.
    """

    @pytest.fixture(scope="class")
    def b1855(self):
        import pint.models as pint_models
        from pint.config import examplefile
        from pint.toa import get_TOAs

        model = pint_models.get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
        toas = get_TOAs(examplefile("B1855+09_NANOGrav_9yv1.tim"), ephem="DE421")
        return model, toas

    @pytest.fixture(scope="class")
    def pint_gls_fit(self, b1855):
        import copy

        from pint.fitter import GLSFitter as PINTGLSFitter

        model, toas = b1855
        f = PINTGLSFitter(toas, copy.deepcopy(model))
        f.fit_toas(maxiter=1)
        return f

    @pytest.fixture(scope="class")
    def jax_gls_fit(self, b1855):
        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitters import GLSFitter

        model, toas = b1855
        toa_data = pint_toas_to_jax(toas, model=model)
        params = pint_model_to_params(model).params
        jax_model, noise_model = build_timing_model(model, toas=toas)
        fitter = GLSFitter(jax_model, toa_data, params, noise_model=noise_model)
        return fitter.fit_toas(maxiter=1)

    @pytest.mark.slow
    def test_chi2_matches(self, pint_gls_fit, jax_gls_fit):
        npt.assert_allclose(jax_gls_fit.chi2, pint_gls_fit.resids.chi2, rtol=1e-3)

    @pytest.mark.slow
    def test_covariance_matches_pint(self, pint_gls_fit, jax_gls_fit):
        """GLS parameter covariance matches PINT.

        T0/OM are excluded: B1855's low eccentricity makes them a
        near-exact degenerate pair, where JaxPINT's SVD cutoff (zero
        variance along the dropped direction) and PINT's Cholesky (huge
        marginal variance along it) legitimately report different
        marginals.  Everything else -- spin, astrometry, DMX, JUMPs, the
        remaining binary parameters -- must agree.
        """
        from tests.helpers import assert_covariance_matches_pint

        assert_covariance_matches_pint(
            jax_gls_fit,
            pint_gls_fit,
            uncert_rtol=0.02,
            corr_atol=0.025,
            exclude=("T0", "OM"),
        )


# ---------------------------------------------------------------------------
# Ill-conditioned covariance: dense oracle
# ---------------------------------------------------------------------------


class TestGLSCovarianceIllConditioned:
    """Covariance parity against a dense oracle at high condition number.

    ``TestGLSSteps`` uses a random, well-conditioned design whose
    normalized normal matrix has singular values clustered near 1. 
    This class stresses the opposite regime: two near-collinear columns
    (condition number > 1e6, still above the SVD cutoff), where that bug
    class produces covariance errors of order the condition number.
    """

    @pytest.fixture
    def ill_conditioned_gls_problem(self):
        n_toas = 50
        n_epochs = 6
        key = jax.random.PRNGKey(7)
        keys = jax.random.split(key, 4)

        t = jnp.linspace(0.0, 1.0, n_toas)
        ortho = jax.random.normal(keys[0], (n_toas,))
        # Columns 1 and 2 are near-collinear: correlation ~ 1 - 5e-8,
        # putting the smallest singular value of the normalized normal
        # matrix at ~1e-7 -- far above the 1e-14*dim SVD cutoff, far
        # below the well-conditioned regime.
        M = jnp.stack(
            [
                jnp.ones(n_toas),
                t,
                t + 3e-4 * ortho / jnp.linalg.norm(ortho) * jnp.linalg.norm(t),
                jax.random.normal(keys[1], (n_toas,)),
            ],
            axis=1,
        ) * 1e-3

        Ndiag = jax.random.uniform(keys[2], (n_toas,), minval=1e-12, maxval=1e-10)
        U_np = np.zeros((n_toas, n_epochs))
        for i in range(n_epochs):
            U_np[i * 8 : min(i * 8 + 8, n_toas), i] = 1.0
        U = jnp.array(U_np)
        Phidiag = jax.random.uniform(keys[3], (n_epochs,), minval=1e-14, maxval=1e-12)
        residuals = jax.random.normal(jax.random.PRNGKey(11), (n_toas,)) * 1e-6
        threshold = 1e-14 * max(n_toas, 4)
        return residuals, Ndiag, U, Phidiag, M, threshold

    def test_covariance_matches_dense_reference(self, ill_conditioned_gls_problem):
        """fullcov and augmented covariances match dense inv(M^T C^-1 M).

        The dense NumPy reference shares no code with the JAX Woodbury /
        augmented solves.  Anti-vacuity guard: the fixture must actually
        be ill-conditioned, so a future edit cannot quietly return it to
        the benign regime where this test stops testing anything.
        """
        from jaxpint.fitters._base import lstsq_step_augmented, lstsq_step_fullcov

        residuals, Ndiag, U, Phidiag, M, threshold = ill_conditioned_gls_problem

        # Anti-vacuity: normalized normal matrix must be ill-conditioned.
        Mn = np.asarray(M)
        C = np.diag(np.asarray(Ndiag)) + np.asarray(U) @ np.diag(
            np.asarray(Phidiag)
        ) @ np.asarray(U).T
        mtcm = Mn.T @ np.linalg.solve(C, Mn)
        d = np.sqrt(np.diag(mtcm))
        corr = mtcm / np.outer(d, d)
        assert np.linalg.cond(corr) > 1e6, (
            f"fixture no longer ill-conditioned: cond={np.linalg.cond(corr):.3e}"
        )

        cov_ref = np.linalg.inv(mtcm)

        _, cov_fc, _ = lstsq_step_fullcov(
            residuals, Ndiag, U, Phidiag, M, threshold
        )
        _, cov_aug, _, _ = lstsq_step_augmented(
            residuals, Ndiag, U, Phidiag, M, threshold
        )

        # Compare as uncertainties + correlations (the covariance spans
        # many decades along the degenerate direction).
        for label, cov in (("fullcov", cov_fc), ("augmented", cov_aug)):
            cov = np.asarray(cov)
            err = np.sqrt(np.diag(cov))
            ref_err = np.sqrt(np.diag(cov_ref))
            npt.assert_allclose(
                err, ref_err, rtol=1e-6,
                err_msg=f"{label}: uncertainty mismatch vs dense oracle",
            )
            npt.assert_allclose(
                cov / np.outer(err, err),
                cov_ref / np.outer(ref_err, ref_err),
                atol=1e-6,
                err_msg=f"{label}: correlation mismatch vs dense oracle",
            )


class TestQuantizationIndexEquivalence:
    """Indexed ECORR storage must be interchangeable with the dense form."""

    def _times_masks(self, n=40, seed=3):
        rng = np.random.default_rng(seed)
        times = np.sort(rng.uniform(0.0, 20 * 86400.0, n))
        masks = {
            "ECORR1": np.arange(n) % 2 == 0,
            "ECORR2": np.arange(n) % 2 == 1,
        }
        return times, masks

    def test_index_matches_dense_builder(self):
        from jaxpint.utils import (
            build_quantization_index,
            build_quantization_matrix,
            quantization_matrix_from_index,
        )

        times, masks = self._times_masks()
        U, slices_dense = build_quantization_matrix(times, masks, dt=86400.0)
        idx, n_epochs, slices_idx = build_quantization_index(
            times, masks, dt=86400.0
        )
        assert slices_idx == slices_dense
        assert n_epochs == U.shape[1]
        npt.assert_array_equal(quantization_matrix_from_index(idx, n_epochs), U)

    def test_unassigned_rows_are_minus_one(self):
        from jaxpint.utils import NO_EPOCH, build_quantization_index

        # nmin=2 with an isolated TOA -> that TOA's singleton epoch dropped.
        times = np.array([0.0, 10.0, 20.0, 5 * 86400.0])
        masks = {"ECORR1": np.ones(4, dtype=bool)}
        idx, n_epochs, _ = build_quantization_index(times, masks, dt=3600.0)
        assert idx[3] == NO_EPOCH
        assert (idx[:3] >= 0).all()

    def test_overlapping_masks_raise(self):
        from jaxpint.utils import build_quantization_index

        times = np.linspace(0.0, 1000.0, 6)
        masks = {
            "ECORR1": np.ones(6, dtype=bool),
            "ECORR2": np.array([True, True, False, False, False, False]),
        }
        with pytest.raises(ValueError, match="overlap"):
            build_quantization_index(times, masks, dt=1e6)

    def test_component_covariance_matches_dense_reference(self):
        from jaxpint.utils import build_quantization_index
        from tests.helpers import make_params, make_toa_data

        times, masks = self._times_masks()
        idx, n_epochs, eslices = build_quantization_index(times, masks, dt=86400.0)
        ecorr = EcorrNoise(
            ecorr_names=("ECORR1", "ECORR2"),
            epoch_index=idx,
            n_epochs=n_epochs,
            ecorr_epoch_slices=(eslices["ECORR1"], eslices["ECORR2"]),
        )
        params = make_params(
            ("ECORR1", "ECORR2"), (5e-7, 3e-7), units=("s", "s")
        )
        toa_data = make_toa_data(n_toas=len(times))
        _, U, Phi = ecorr.covariance(toa_data, params)
        # Dense reference built independently of the component.
        U_ref = np.zeros((len(times), n_epochs))
        valid = idx >= 0
        U_ref[np.nonzero(valid)[0], idx[valid]] = 1.0
        npt.assert_array_equal(np.asarray(U), U_ref)
        C = np.asarray(U) @ np.diag(np.asarray(Phi)) @ np.asarray(U).T
        C_ref = U_ref @ np.diag(np.asarray(Phi)) @ U_ref.T
        npt.assert_array_equal(C, C_ref)

    def test_from_dense_round_trip(self):
        from jaxpint.utils import build_quantization_matrix

        times, masks = self._times_masks()
        U, eslices = build_quantization_matrix(times, masks, dt=86400.0)
        ecorr = EcorrNoise.from_dense(
            ecorr_names=("ECORR1", "ECORR2"),
            quantization_matrix=U,
            ecorr_epoch_slices=(eslices["ECORR1"], eslices["ECORR2"]),
        )
        from jaxpint.utils import quantization_matrix_from_index

        npt.assert_array_equal(
            quantization_matrix_from_index(
                np.asarray(ecorr.epoch_index), ecorr.n_epochs
            ),
            U,
        )
        assert ecorr.n_epochs == U.shape[1]
        assert ecorr.basis_width() == U.shape[1]

    def test_from_dense_rejects_multihot_rows(self):
        bad = np.zeros((3, 2))
        bad[0, 0] = bad[0, 1] = 1.0
        with pytest.raises(ValueError, match="one"):
            EcorrNoise.from_dense(
                ecorr_names=("ECORR1",),
                quantization_matrix=bad,
                ecorr_epoch_slices=((0, 2),),
            )
