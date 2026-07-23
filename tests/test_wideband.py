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
    """Verify wideband DM data is correctly converted to JaxPINT.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "attr, pint_getter",
        [("dm_values", "get_dms"), ("dm_errors", "get_dm_errors")],
    )
    def test_dm_field_matches_pint(self, jax_wb, pint_wb, attr, pint_getter):
        _, toa_data, _, _ = jax_wb
        _, toas = pint_wb
        pint_vals = getattr(toas, pint_getter)().to(u.pc / u.cm**3).value
        npt.assert_allclose(
            np.array(getattr(toa_data, attr)), pint_vals, rtol=1e-14,
        )


# ---------------------------------------------------------------------------
# Model DM computation: compute_dm matches PINT total_dm
# ---------------------------------------------------------------------------


class TestModelDM:
    """Test that JaxPINT's model.compute_dm matches PINT's total_dm.
    """

    @pytest.mark.slow
    def test_compute_dm_matches_pint(self, jax_wb, pint_wb):
        """JaxPINT compute_dm should match PINT total_dm."""
        jax_model, toa_data, params, _ = jax_wb
        pint_model, toas = pint_wb

        jax_dm = np.array(jax_model.compute_dm(toa_data, params))
        pint_dm = pint_model.total_dm(toas).to(u.pc / u.cm**3).value

        npt.assert_allclose(jax_dm, pint_dm, rtol=1e-10)

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


# ---------------------------------------------------------------------------
# Time residuals (narrowband part)
# ---------------------------------------------------------------------------


class TestTimeResiduals:
    """Verify that narrowband time residuals match PINT."""

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
    def test_structure(self, jax_wb, pint_wb):
        """One computation: 2N shape, [time; dm] layout contract, finite.
        """
        jax_model, toa_data, params, _ = jax_wb
        _, toas = pint_wb
        n = toas.ntoas

        wb_resid = compute_wideband_residuals(jax_model, toa_data, params)
        assert wb_resid.shape == (2 * n,)
        npt.assert_array_equal(
            np.array(wb_resid[:n]),
            np.array(compute_time_residuals(jax_model, toa_data, params)),
        )
        npt.assert_array_equal(
            np.array(wb_resid[n:]),
            np.array(compute_dm_residuals(jax_model, toa_data, params)),
        )
        assert not jnp.any(jnp.isnan(wb_resid))


# ---------------------------------------------------------------------------
# Wideband design matrix
# ---------------------------------------------------------------------------


class TestWidebandDesignMatrix:
    """Test the combined wideband design matrix."""

    @pytest.mark.slow
    def test_structure(self, jax_wb, pint_wb):
        """One computation: shape, offset column, finite, no dead columns.

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
        assert not jnp.any(jnp.isnan(M))
        # Each free parameter affects at least one residual.
        assert jnp.all(jnp.linalg.norm(M, axis=0) > 0)
        # Without offset
        M_no = compute_wideband_design_matrix(
            jax_model, toa_data, params, include_offset=False
        )
        assert M_no.shape == (2 * n, params.n_free)

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
            np.array(M_wb[:n, :]), np.array(M_nb), rtol=1e-11,
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
    """Test ScaleDmError (DMEFAC/DMEQUAD) via the NoiseModel.
    """

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
    """Test DMJUMP component.

    Both tests locate the DispersionJump instance and assert on failure,
    so a missing component cannot pass silently — no separate presence
    test needed.
    """

    @pytest.mark.slow
    def test_dmjump_zero_delay(self, jax_wb):
        """DMJUMP affects DM only -- its timing-delay contribution is zero."""
        from jaxpint.delay.dispersion_jump import DispersionJump

        jax_model, toa_data, params, _ = jax_wb
        # DMJUMP lives in dispersion_components (it is a DispersionDelayComponent),
        # NOT delay_components -- iterating the latter would never match and the
        # assertion would silently never run.
        dmjump = next(
            (c for c in jax_model.dispersion_components
             if isinstance(c, DispersionJump)),
            None,
        )
        assert dmjump is not None, "no DispersionJump in dispersion_components"
        delay = dmjump(toa_data, params, jnp.zeros(toa_data.n_toas))
        npt.assert_array_equal(np.array(delay), 0.0)

    @pytest.mark.slow
    def test_dmjump_nonzero_dm(self, jax_wb):
        """DMJUMP should contribute nonzero DM for some TOAs."""
        from jaxpint.delay.dispersion_jump import DispersionJump

        jax_model, toa_data, params, _ = jax_wb
        dmjump = next(
            (c for c in jax_model.dispersion_components
             if isinstance(c, DispersionJump)),
            None,
        )
        assert dmjump is not None, "no DispersionJump in dispersion_components"
        dm = dmjump.compute_dm(toa_data, params, jnp.zeros(toa_data.n_toas))
        assert jnp.any(dm != 0.0)


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
    def test_dof_correct(self, jax_wb, jax_wb_fit):
        """``dof`` accounts for the implicit Offset column.

        Same off-by-one as the narrowband case: ``2N - n_free - 1`` when
        the model has no explicit ``PhaseOffset`` component.
        """
        _, toa_data, params, _ = jax_wb
        expected_dof = 2 * toa_data.n_toas - params.n_free - 1
        assert jax_wb_fit.dof == expected_dof

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["F0", "F1"])
    def test_fitted_param_matches_pint(self, pint_wb_fit, jax_wb_fit, name):
        pint_param = getattr(pint_wb_fit.model, name)
        pint_val = float(pint_param.value)
        pint_err = float(pint_param.uncertainty_value)
        jax_val = float(jax_wb_fit.params.param_value(name))
        assert abs(jax_val - pint_val) < 3 * pint_err, name

    @pytest.mark.slow
    def test_covariance_matches_pint(self, pint_wb_fit, jax_wb_fit):
        """Wideband parameter covariance matches PINT's WidebandTOAFitter.
        """
        from tests.helpers import assert_covariance_matches_pint

        assert_covariance_matches_pint(
            jax_wb_fit,
            pint_wb_fit,
            uncert_rtol=0.01,
            corr_atol=0.005,
        )
