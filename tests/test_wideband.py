"""Tests for wideband timing support.

Compares JaxPINT wideband functionality against PINT using the
J1614-2230 NANOGrav 12yv3 wideband dataset (available via
``pint.config.examplefile``).
"""

from __future__ import annotations

import astropy.units as u
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

pytest.importorskip("pint")  # optional dependency; skip module if absent
import pint.models as pm
from pint.config import examplefile
from pint.toa import get_TOAs
from pint.residuals import Residuals, WidebandTOAResiduals

from jaxpint.bridge import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
)
from jaxpint.fitters import (
    WidebandGLSFitter,
    compute_dm_residuals,
    compute_time_residuals,
    compute_wideband_design_matrix,
    compute_wideband_residuals,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def pint_wb():
    """Load J1614-2230 wideband PINT model and TOAs."""
    par = examplefile("J1614-2230_NANOGrav_12yv3.wb.gls.par")
    tim = examplefile("J1614-2230_NANOGrav_12yv3.wb.tim")
    model = pm.get_model(par)
    toas = get_TOAs(tim, ephem="DE436", bipm_version="BIPM2015")
    return model, toas


@pytest.fixture(scope="module")
def jax_wb(pint_wb):
    """Convert wideband data to JaxPINT objects."""
    pint_model, toas = pint_wb
    toa_data = pint_toas_to_jax(toas, model=pint_model)
    params = pint_model_to_params(pint_model).params
    jax_model, noise_model = build_timing_model(pint_model, toas=toas)
    return jax_model, toa_data, params, noise_model


# ---------------------------------------------------------------------------
# TOA data: wideband fields populated
# ---------------------------------------------------------------------------


class TestWidebandTOAConversion:
    """Verify wideband DM data is correctly converted to JaxPINT."""

    @pytest.mark.slow
    def test_dm_values_present(self, jax_wb):
        _, toa_data, _, _ = jax_wb
        assert toa_data.dm_values is not None

    @pytest.mark.slow
    def test_dm_errors_present(self, jax_wb):
        _, toa_data, _, _ = jax_wb
        assert toa_data.dm_errors is not None

    @pytest.mark.slow
    def test_dm_values_shape(self, jax_wb, pint_wb):
        _, toa_data, _, _ = jax_wb
        _, toas = pint_wb
        assert toa_data.dm_values.shape == (toas.ntoas,)

    @pytest.mark.slow
    def test_dm_errors_shape(self, jax_wb, pint_wb):
        _, toa_data, _, _ = jax_wb
        _, toas = pint_wb
        assert toa_data.dm_errors.shape == (toas.ntoas,)

    @pytest.mark.slow
    def test_dm_values_match_pint(self, jax_wb, pint_wb):
        _, toa_data, _, _ = jax_wb
        _, toas = pint_wb
        pint_dms = toas.get_dms().to(u.pc / u.cm**3).value
        npt.assert_allclose(
            np.array(toa_data.dm_values), pint_dms, rtol=1e-14,
        )

    @pytest.mark.slow
    def test_dm_errors_match_pint(self, jax_wb, pint_wb):
        _, toa_data, _, _ = jax_wb
        _, toas = pint_wb
        pint_dme = toas.get_dm_errors().to(u.pc / u.cm**3).value
        npt.assert_allclose(
            np.array(toa_data.dm_errors), pint_dme, rtol=1e-14,
        )

    @pytest.mark.slow
    def test_dm_values_positive(self, jax_wb):
        _, toa_data, _, _ = jax_wb
        assert jnp.all(toa_data.dm_values > 0)

    @pytest.mark.slow
    def test_dm_errors_positive(self, jax_wb):
        _, toa_data, _, _ = jax_wb
        assert jnp.all(toa_data.dm_errors > 0)


# ---------------------------------------------------------------------------
# Model DM computation: compute_dm matches PINT total_dm
# ---------------------------------------------------------------------------


class TestModelDM:
    """Test that JaxPINT's model.compute_dm matches PINT's total_dm."""

    @pytest.mark.slow
    def test_compute_dm_shape(self, jax_wb, pint_wb):
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        dm = jax_model.compute_dm(toa_data, params)
        assert dm.shape == (toas.ntoas,)

    @pytest.mark.slow
    def test_compute_dm_no_nan(self, jax_wb):
        jax_model, toa_data, params, _ = jax_wb
        dm = jax_model.compute_dm(toa_data, params)
        assert not jnp.any(jnp.isnan(dm))

    @pytest.mark.slow
    def test_compute_dm_matches_pint(self, jax_wb, pint_wb):
        """JaxPINT compute_dm should match PINT total_dm."""
        jax_model, toa_data, params, _ = jax_wb
        pint_model, toas = pint_wb

        jax_dm = np.array(jax_model.compute_dm(toa_data, params))
        pint_dm = pint_model.total_dm(toas).to(u.pc / u.cm**3).value

        npt.assert_allclose(jax_dm, pint_dm, rtol=1e-10)

    @pytest.mark.slow
    def test_dispersion_components_populated(self, jax_wb):
        jax_model, _, _, _ = jax_wb
        assert len(jax_model.dispersion_components) > 0

    @pytest.mark.slow
    def test_compute_dm_differentiable(self, jax_wb):
        """compute_dm should be differentiable via jax.jacobian."""
        jax_model, toa_data, params, _ = jax_wb

        def dm_fn(all_values):
            p = eqx.tree_at(lambda pv: pv.values, params, all_values)
            return jax_model.compute_dm(toa_data, p)

        J = jax.jacobian(dm_fn)(params.values)
        assert J.shape == (toa_data.n_toas, len(params.values))
        assert not jnp.any(jnp.isnan(J))


# ---------------------------------------------------------------------------
# DM residuals
# ---------------------------------------------------------------------------


class TestDMResiduals:
    """Test DM residual computation against PINT."""

    @pytest.mark.slow
    def test_dm_residuals_shape(self, jax_wb, pint_wb):
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        dm_resid = compute_dm_residuals(jax_model, toa_data, params)
        assert dm_resid.shape == (toas.ntoas,)

    @pytest.mark.slow
    def test_dm_residuals_match_pint(self, jax_wb, pint_wb):
        """DM residuals should match PINT's WidebandDMResiduals."""
        jax_model, toa_data, params, _ = jax_wb
        pint_model, toas = pint_wb

        jax_dm_resid = np.array(
            compute_dm_residuals(jax_model, toa_data, params)
        )
        pint_resids = WidebandTOAResiduals(toas, pint_model)
        pint_dm_resid = pint_resids.dm.resids.to(u.pc / u.cm**3).value

        npt.assert_allclose(jax_dm_resid, pint_dm_resid, rtol=1e-10)

    @pytest.mark.slow
    def test_dm_residuals_no_nan(self, jax_wb):
        jax_model, toa_data, params, _ = jax_wb
        dm_resid = compute_dm_residuals(jax_model, toa_data, params)
        assert not jnp.any(jnp.isnan(dm_resid))


# ---------------------------------------------------------------------------
# Time residuals (narrowband part)
# ---------------------------------------------------------------------------


class TestTimeResiduals:
    """Verify that narrowband time residuals match PINT."""

    @pytest.mark.slow
    def test_time_residuals_finite(self, jax_wb):
        """Time residuals should be finite and have the right shape."""
        jax_model, toa_data, params, _ = jax_wb
        time_resid = compute_time_residuals(jax_model, toa_data, params)
        assert time_resid.shape == (toa_data.n_toas,)
        assert not jnp.any(jnp.isnan(time_resid))
        assert not jnp.any(jnp.isinf(time_resid))

    @pytest.mark.slow
    def test_time_residuals_match_pint(self, jax_wb, pint_wb):
        """Time residuals should match PINT to ~100 ns."""
        jax_model, toa_data, params, _ = jax_wb
        pint_model, toas = pint_wb

        jax_resid = np.array(compute_time_residuals(jax_model, toa_data, params))
        pint_resid = Residuals(toas, pint_model).time_resids.to(u.s).value

        npt.assert_allclose(jax_resid, pint_resid, atol=1e-7)


# ---------------------------------------------------------------------------
# Combined wideband residuals
# ---------------------------------------------------------------------------


class TestWidebandResiduals:
    """Test combined [time_resid; dm_resid] vector."""

    @pytest.mark.slow
    def test_shape(self, jax_wb, pint_wb):
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        wb_resid = compute_wideband_residuals(jax_model, toa_data, params)
        assert wb_resid.shape == (2 * toas.ntoas,)

    @pytest.mark.slow
    def test_first_half_is_time(self, jax_wb, pint_wb):
        """First N entries should be time residuals."""
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        wb_resid = compute_wideband_residuals(jax_model, toa_data, params)
        time_resid = compute_time_residuals(jax_model, toa_data, params)

        npt.assert_array_equal(
            np.array(wb_resid[:n]), np.array(time_resid),
        )

    @pytest.mark.slow
    def test_second_half_is_dm(self, jax_wb, pint_wb):
        """Last N entries should be DM residuals."""
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        wb_resid = compute_wideband_residuals(jax_model, toa_data, params)
        dm_resid = compute_dm_residuals(jax_model, toa_data, params)

        npt.assert_array_equal(
            np.array(wb_resid[n:]), np.array(dm_resid),
        )

    @pytest.mark.slow
    def test_no_nan(self, jax_wb):
        jax_model, toa_data, params, _ = jax_wb
        wb_resid = compute_wideband_residuals(jax_model, toa_data, params)
        assert not jnp.any(jnp.isnan(wb_resid))


# ---------------------------------------------------------------------------
# Wideband design matrix
# ---------------------------------------------------------------------------


class TestWidebandDesignMatrix:
    """Test the combined wideband design matrix."""

    @pytest.mark.slow
    def test_shape(self, jax_wb, pint_wb):
        """Default ``include_offset=True`` adds one Offset column.

        Offset column is ``[1, ..., 1, 0, ..., 0]`` — ones for the
        time-residual block, zeros for the DM-residual block, mirroring
        PINT's wideband design-matrix convention.
        """
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas
        M = compute_wideband_design_matrix(jax_model, toa_data, params)
        assert M.shape == (2 * n, params.n_free + 1)
        # Offset column structure
        np.testing.assert_array_equal(np.array(M[:n, 0]), 1.0)
        np.testing.assert_array_equal(np.array(M[n:, 0]), 0.0)
        # Without offset
        M_no = compute_wideband_design_matrix(
            jax_model, toa_data, params, include_offset=False
        )
        assert M_no.shape == (2 * n, params.n_free)

    @pytest.mark.slow
    def test_no_nan(self, jax_wb):
        jax_model, toa_data, params, _ = jax_wb
        M = compute_wideband_design_matrix(jax_model, toa_data, params)
        assert not jnp.any(jnp.isnan(M))

    @pytest.mark.slow
    def test_columns_nonzero(self, jax_wb):
        """Each free parameter should affect at least one residual."""
        jax_model, toa_data, params, _ = jax_wb
        M = compute_wideband_design_matrix(jax_model, toa_data, params)
        col_norms = jnp.linalg.norm(M, axis=0)
        assert jnp.all(col_norms > 0)

    @pytest.mark.slow
    def test_toa_block_matches_narrowband(self, jax_wb, pint_wb):
        """Top half of wideband design matrix should match narrowband."""
        from jaxpint.fitters import compute_design_matrix

        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        M_wb = compute_wideband_design_matrix(jax_model, toa_data, params)
        M_nb = compute_design_matrix(jax_model, toa_data, params)

        npt.assert_allclose(
            np.array(M_wb[:n, :]), np.array(M_nb), rtol=1e-12,
        )

    @pytest.mark.slow
    def test_dm_block_nonzero_for_dm_params(self, jax_wb, pint_wb):
        """DM rows should be nonzero for DM-related parameters."""
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        M_wb = compute_wideband_design_matrix(jax_model, toa_data, params)
        M_dm = M_wb[n:, :]  # DM block

        # At least some columns should be nonzero (DM, DMX, DMJUMP)
        dm_col_norms = jnp.linalg.norm(M_dm, axis=0)
        assert jnp.any(dm_col_norms > 0)


# ---------------------------------------------------------------------------
# Noise model: DM white noise
# ---------------------------------------------------------------------------


class TestDMWhiteNoise:
    """Test ScaleDmError (DMEFAC/DMEQUAD) via the NoiseModel."""

    @pytest.mark.slow
    def test_dm_white_noise_present(self, jax_wb):
        _, _, _, noise_model = jax_wb
        assert noise_model.dm_white_noise is not None

    @pytest.mark.slow
    def test_scaled_dm_sigma_shape(self, jax_wb, pint_wb):
        _, toa_data, params, noise_model = jax_wb
        _, toas = pint_wb
        sigma_dm = noise_model.scaled_dm_sigma(toa_data, params)
        assert sigma_dm.shape == (toas.ntoas,)

    @pytest.mark.slow
    def test_scaled_dm_sigma_positive(self, jax_wb):
        _, toa_data, params, noise_model = jax_wb
        sigma_dm = noise_model.scaled_dm_sigma(toa_data, params)
        assert jnp.all(sigma_dm > 0)

    @pytest.mark.slow
    def test_scaled_dm_sigma_matches_pint(self, jax_wb, pint_wb):
        """Scaled DM sigma should match PINT's scaled_dm_uncertainty."""
        _, toa_data, params, noise_model = jax_wb
        pint_model, toas = pint_wb

        jax_sigma = np.array(noise_model.scaled_dm_sigma(toa_data, params))
        pint_sigma = pint_model.scaled_dm_uncertainty(toas).to(
            u.pc / u.cm**3
        ).value

        npt.assert_allclose(jax_sigma, pint_sigma, rtol=1e-12)

    @pytest.mark.slow
    def test_wideband_covariance_shapes(self, jax_wb, pint_wb):
        _, toa_data, params, noise_model = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        Ndiag_toa, U_toa, Phi_toa, Ndiag_dm = noise_model.wideband_covariance(
            toa_data, params
        )
        assert Ndiag_toa.shape == (n,)
        assert U_toa.shape[0] == n
        assert Ndiag_dm.shape == (n,)


# ---------------------------------------------------------------------------
# DispersionJump (DMJUMP)
# ---------------------------------------------------------------------------


class TestDispersionJump:
    """Test DMJUMP component."""

    @pytest.mark.slow
    def test_dmjump_in_dispersion_components(self, jax_wb):
        from jaxpint.delay.dispersion_jump import DispersionJump

        jax_model, _, _, _ = jax_wb
        has_dmjump = any(
            isinstance(c, DispersionJump)
            for c in jax_model.dispersion_components
        )
        assert has_dmjump

    @pytest.mark.slow
    def test_dmjump_zero_delay(self, jax_wb):
        """DMJUMP should contribute zero timing delay."""
        from jaxpint.delay.dispersion_jump import DispersionJump

        jax_model, toa_data, params, _ = jax_wb
        for comp in jax_model.delay_components:
            if isinstance(comp, DispersionJump):
                delay = comp(
                    toa_data, params, jnp.zeros(toa_data.n_toas)
                )
                npt.assert_array_equal(np.array(delay), 0.0)
                break

    @pytest.mark.slow
    def test_dmjump_nonzero_dm(self, jax_wb):
        """DMJUMP should contribute nonzero DM for some TOAs."""
        from jaxpint.delay.dispersion_jump import DispersionJump

        jax_model, toa_data, params, _ = jax_wb
        for comp in jax_model.dispersion_components:
            if isinstance(comp, DispersionJump):
                dm = comp.compute_dm(
                    toa_data, params, jnp.zeros(toa_data.n_toas)
                )
                assert jnp.any(dm != 0.0)
                break


# ---------------------------------------------------------------------------
# Wideband GLS fitter
# ---------------------------------------------------------------------------


class TestWidebandGLSFitter:
    """Test the wideband GLS fitter."""

    @pytest.mark.slow
    def test_fitter_creates(self, jax_wb):
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        # Constructor stores the inputs it will fit against.
        assert fitter.model is jax_model
        assert fitter.toa_data is toa_data
        assert fitter.params is params
        assert fitter.noise_model is noise_model

    @pytest.mark.slow
    def test_fit_returns_result(self, jax_wb):
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        result = fitter.fit_toas(maxiter=1)
        # A real fit yields a finite, positive chi2 and a finite reduced chi2.
        assert jnp.isfinite(result.chi2)
        assert result.chi2 > 0
        assert jnp.isfinite(result.reduced_chi2)

    @pytest.mark.slow
    def test_result_has_both_residuals(self, jax_wb):
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        result = fitter.fit_toas(maxiter=1)
        assert result.time_residuals.shape == (toa_data.n_toas,)
        assert result.dm_residuals.shape == (toa_data.n_toas,)

    @pytest.mark.slow
    def test_dof_correct(self, jax_wb):
        """``dof`` accounts for the implicit Offset column.

        Same off-by-one as the narrowband case: ``2N - n_free - 1`` when
        the model has no explicit ``PhaseOffset`` component.
        """
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        result = fitter.fit_toas(maxiter=1)
        expected_dof = 2 * toa_data.n_toas - params.n_free - 1
        assert result.dof == expected_dof

    @pytest.mark.slow
    def test_prefit_design_matrix_and_solve(self, jax_wb):
        """Verify the WLS solve step produces finite dpars before update."""
        from jaxpint.fitters._base import _subtract_weighted_mean, wls_step

        jax_model, toa_data, params, noise_model = jax_wb

        sigma_toa = noise_model.scaled_sigma(toa_data, params)
        sigma_dm = noise_model.scaled_dm_sigma(toa_data, params)
        Ndiag = jnp.concatenate([sigma_toa**2, sigma_dm**2])
        sigma_combined = jnp.sqrt(Ndiag)

        time_resid = compute_time_residuals(jax_model, toa_data, params)
        dm_resid = compute_dm_residuals(jax_model, toa_data, params)
        time_resid_ms = _subtract_weighted_mean(time_resid, sigma_toa)
        residuals = jnp.concatenate([time_resid_ms, dm_resid])

        M = compute_wideband_design_matrix(jax_model, toa_data, params)
        threshold = 1e-14 * max(2 * toa_data.n_toas, params.n_free)
        dpars, cov, norms = wls_step(residuals, sigma_combined, M, threshold)

        assert not jnp.any(jnp.isnan(dpars))
        assert not jnp.any(jnp.isnan(jnp.diag(cov)))

    @pytest.mark.slow
    def test_covariance_symmetric(self, jax_wb):
        """Pre-fit covariance from the solve step should be symmetric."""
        from jaxpint.fitters._base import _subtract_weighted_mean, wls_step

        jax_model, toa_data, params, noise_model = jax_wb

        sigma_toa = noise_model.scaled_sigma(toa_data, params)
        sigma_dm = noise_model.scaled_dm_sigma(toa_data, params)
        sigma_combined = jnp.sqrt(
            jnp.concatenate([sigma_toa**2, sigma_dm**2])
        )

        time_resid = _subtract_weighted_mean(
            compute_time_residuals(jax_model, toa_data, params), sigma_toa
        )
        residuals = jnp.concatenate([
            time_resid,
            compute_dm_residuals(jax_model, toa_data, params),
        ])

        M = compute_wideband_design_matrix(jax_model, toa_data, params)
        threshold = 1e-14 * max(2 * toa_data.n_toas, params.n_free)
        _, cov, _ = wls_step(residuals, sigma_combined, M, threshold)

        npt.assert_allclose(np.array(cov), np.array(cov.T), atol=1e-20)

    @pytest.mark.slow
    def test_postfit_residuals_finite(self, jax_wb):
        """Post-fit time and DM residuals should be finite."""
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        result = fitter.fit_toas(maxiter=1)
        assert not jnp.any(jnp.isnan(result.time_residuals))
        assert not jnp.any(jnp.isnan(result.dm_residuals))


# ---------------------------------------------------------------------------
# End-to-end wideband fit: JaxPINT vs PINT
# ---------------------------------------------------------------------------


class TestWidebandFitVsPINT:
    """Compare JaxPINT WidebandGLSFitter against PINT WidebandTOAFitter."""

    @pytest.fixture(scope="class")
    def pint_wb_fit(self, pint_wb):
        """Run PINT's WidebandTOAFitter for 1 iteration."""
        import copy
        from pint.fitter import WidebandTOAFitter

        pint_model, toas = pint_wb
        m = copy.deepcopy(pint_model)
        f = WidebandTOAFitter(toas, m)
        f.fit_toas(maxiter=1)
        return f

    @pytest.fixture(scope="class")
    def jax_wb_fit(self, jax_wb):
        """Run JaxPINT's WidebandGLSFitter for 1 iteration."""
        jax_model, toa_data, params, noise_model = jax_wb
        fitter = WidebandGLSFitter(
            jax_model, toa_data, params, noise_model=noise_model
        )
        return fitter.fit_toas(maxiter=1)

    @pytest.mark.slow
    def test_chi2_matches(self, pint_wb_fit, jax_wb_fit):
        """Post-fit chi2 should agree between JaxPINT and PINT."""
        pint_chi2 = pint_wb_fit.resids.chi2
        jax_chi2 = jax_wb_fit.chi2
        npt.assert_allclose(jax_chi2, pint_chi2, rtol=0.05)

    @pytest.mark.slow
    def test_chi2_decreases(self, pint_wb, jax_wb, jax_wb_fit):
        """Post-fit chi2 should be lower than pre-fit chi2."""
        from jaxpint.fitters._base import _subtract_weighted_mean

        jax_model, toa_data, params, noise_model = jax_wb
        sigma_toa = noise_model.scaled_sigma(toa_data, params)
        sigma_dm = noise_model.scaled_dm_sigma(toa_data, params)
        time_resid = _subtract_weighted_mean(
            compute_time_residuals(jax_model, toa_data, params), sigma_toa
        )
        dm_resid = compute_dm_residuals(jax_model, toa_data, params)
        chi2_pre = float(
            jnp.sum((time_resid / sigma_toa) ** 2)
            + jnp.sum((dm_resid / sigma_dm) ** 2)
        )
        assert jax_wb_fit.chi2 < chi2_pre

    @pytest.mark.slow
    def test_f0_matches(self, pint_wb_fit, jax_wb_fit):
        pint_val = float(pint_wb_fit.model.F0.value)
        jax_val = float(jax_wb_fit.params.param_value("F0"))
        pint_err = float(pint_wb_fit.model.F0.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    @pytest.mark.slow
    def test_f1_matches(self, pint_wb_fit, jax_wb_fit):
        pint_val = float(pint_wb_fit.model.F1.value)
        jax_val = float(jax_wb_fit.params.param_value("F1"))
        pint_err = float(pint_wb_fit.model.F1.uncertainty_value)
        assert abs(jax_val - pint_val) < 3 * pint_err

    @pytest.mark.slow
    def test_uncertainties_positive(self, jax_wb_fit):
        assert jnp.all(jax_wb_fit.parameter_uncertainties > 0)

    @pytest.mark.slow
    def test_covariance_symmetric(self, jax_wb_fit):
        cov = jax_wb_fit.covariance_matrix
        npt.assert_allclose(np.array(cov), np.array(cov.T), atol=1e-20)

    @pytest.mark.slow
    def test_postfit_residuals_finite(self, jax_wb_fit):
        assert not jnp.any(jnp.isnan(jax_wb_fit.time_residuals))
        assert not jnp.any(jnp.isnan(jax_wb_fit.dm_residuals))
