"""Integration tests for the WLS fitter against PINT.

Uses a synthetic isolated pulsar with only Spindown + DispersionDM
(the components currently ported to JaxPINT) so results are directly
comparable between PINT and JaxPINT.
"""

from __future__ import annotations

import copy
import io
import pathlib

import astropy.units as u
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("pint")  # optional dependency; skip module if absent
import pint.models as models
import pint.toa as toa
from pint.config import examplefile
from pint.fitter import WLSFitter as PINTWLSFitter
from pint.simulation import make_fake_toas_uniform

from jaxpint.bridge import (
    build_timing_model,
    pint_model_to_params,
    pint_toas_to_jax,
)
from jaxpint.fitters import (
    WLSFitter,
    compute_design_matrix,
    compute_time_residuals,
)

from jaxpint.fitters._base import _subtract_weighted_mean


# ---------------------------------------------------------------------------
# Synthetic isolated pulsar (Spindown + DM only)
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
def synthetic_data():
    """Generate synthetic multi-frequency TOAs from a Spindown+DM model."""
    np.random.seed(42)
    m_true = models.get_model(io.StringIO(_SYNTH_PAR))
    # Two frequency bands so DM is well-determined
    toas_lo = make_fake_toas_uniform(
        53000,
        55000,
        30,
        m_true,
        error=10 * u.us,
        add_noise=True,
        freq=1400 * u.MHz,
    )
    toas_hi = make_fake_toas_uniform(
        53000,
        55000,
        30,
        m_true,
        error=10 * u.us,
        add_noise=True,
        freq=2000 * u.MHz,
    )
    toas_lo.merge(toas_hi)
    return m_true, toas_lo


@pytest.fixture(scope="module")
def pint_fit(synthetic_data):
    """Run PINT's WLS fitter on the synthetic data."""
    m_true, toas = synthetic_data
    mc = copy.deepcopy(m_true)
    f = PINTWLSFitter(toas, mc)
    f.fit_toas(maxiter=1)
    return f


@pytest.fixture(scope="module")
def jax_objects(synthetic_data):
    """Convert synthetic data to JaxPINT objects."""
    m_true, toas = synthetic_data
    toa_data = pint_toas_to_jax(toas, model=m_true)
    params = pint_model_to_params(m_true).params
    jax_model, _noise = build_timing_model(m_true)
    return jax_model, toa_data, params


@pytest.fixture(scope="module")
def jax_fit(jax_objects):
    """Run JaxPINT's WLS fitter on the synthetic data."""
    jax_model, toa_data, params = jax_objects
    fitter = WLSFitter(jax_model, toa_data, params)
    return fitter.fit_toas(maxiter=1)


# ---------------------------------------------------------------------------
# NGC6440E (real isolated pulsar, limited comparison)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ngc6440e():
    """Load NGC6440E, freeze astrometry, keep F0/F1/DM free."""
    pint_model = models.get_model(examplefile("NGC6440E.par"))
    toas = toa.get_TOAs(examplefile("NGC6440E.tim"), ephem="DE421")
    return pint_model, toas


# ---------------------------------------------------------------------------
# Tests: synthetic data (strict comparison)
# ---------------------------------------------------------------------------


class TestDesignMatrix:
    """Design matrix shape and properties."""

    def test_shape(self, jax_objects):
        """Default ``include_offset=True`` adds one Offset column (matches PINT).

        Test both shapes to lock in the API.
        """
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        assert M.shape == (toa_data.n_toas, params.n_free + 1)
        M_no_offset = compute_design_matrix(
            jax_model, toa_data, params, include_offset=False
        )
        assert M_no_offset.shape == (toa_data.n_toas, params.n_free)
        # Offset column is the first column and must be ones.
        np.testing.assert_array_equal(np.array(M[:, 0]), 1.0)
        # The remaining columns equal the no-offset matrix.
        np.testing.assert_allclose(
            np.array(M[:, 1:]), np.array(M_no_offset), rtol=0, atol=0
        )

    def test_no_nan(self, jax_objects):
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        assert not jnp.any(jnp.isnan(M))

    def test_columns_nonzero(self, jax_objects):
        """Each free parameter should affect at least one TOA."""
        jax_model, toa_data, params = jax_objects
        M = compute_design_matrix(jax_model, toa_data, params)
        col_norms = jnp.linalg.norm(M, axis=0)
        assert jnp.all(col_norms > 0)


class TestSyntheticFit:
    """Compare JaxPINT WLS fit against PINT on synthetic data."""

    @pytest.mark.slow
    def test_chi2_matches(self, pint_fit, jax_fit):
        pint_chi2 = pint_fit.resids.chi2
        jax_chi2 = jax_fit.chi2
        np.testing.assert_allclose(jax_chi2, pint_chi2, rtol=1e-3)

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["F0", "F1", "DM"])
    def test_fitted_param_matches_pint(self, pint_fit, jax_fit, name):
        pint_param = getattr(pint_fit.model, name)
        pint_val = float(pint_param.value)
        pint_err = float(pint_param.uncertainty_value)
        jax_val = float(jax_fit.params.param_value(name))
        assert abs(jax_val - pint_val) < 3 * pint_err, name

    @pytest.mark.slow
    def test_uncertainties_positive(self, jax_fit):
        assert jnp.all(jax_fit.parameter_uncertainties > 0)

    @pytest.mark.slow
    def test_covariance_symmetric(self, jax_fit):
        cov = jax_fit.covariance_matrix
        np.testing.assert_allclose(np.array(cov), np.array(cov.T), atol=1e-20)

    @pytest.mark.slow
    def test_correlation_diagonal_ones(self, jax_fit):
        corr = jax_fit.correlation_matrix
        np.testing.assert_allclose(np.diag(np.array(corr)), 1.0, atol=1e-12)

    @pytest.mark.slow
    def test_dof(self, synthetic_data, jax_fit):
        """``dof`` accounts for the implicit Offset column.

        When the model has no explicit ``PhaseOffset`` component, the
        constant-residual DOF is absorbed by the Offset column added to M,
        so ``dof = n_toas - n_free - 1`` (matches PINT's accounting).
        """
        _, toas = synthetic_data
        n_toas = len(toas)
        n_free = jax_fit.params.n_free
        assert jax_fit.dof == n_toas - n_free - 1

    @pytest.mark.slow
    def test_reduced_chi2_reasonable(self, jax_fit):
        """Reduced chi2 should be close to 1 for synthetic data."""
        assert 0.5 < jax_fit.reduced_chi2 < 2.0


# ---------------------------------------------------------------------------
# Tests: NGC6440E real data (structural tests)
# ---------------------------------------------------------------------------


class TestNGC6440E:
    """Tests on real NGC6440E data with astrometry."""

    @pytest.mark.slow
    def test_chi2_decreases(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, _noise = build_timing_model(pint_model)

        # Pre-fit chi2
        resid0 = compute_time_residuals(jax_model, toa_data, params)
        resid0 = _subtract_weighted_mean(resid0, toa_data.error)
        chi2_pre = float(jnp.sum((resid0 / toa_data.error) ** 2))

        # Single iteration
        fitter = WLSFitter(jax_model, toa_data, params)
        result = fitter.fit_toas(maxiter=1)

        assert result.chi2 < chi2_pre

    @pytest.mark.slow
    def test_multiple_iterations_converge(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, _noise = build_timing_model(pint_model)

        fitter = WLSFitter(jax_model, toa_data, params)
        chi2_one = fitter.fit_toas(maxiter=1).chi2
        result = fitter.fit_toas(maxiter=5)

        # Converged: extra iterations never worsen the fit and settle to a
        # finite chi2; the published model fits its own TOAs at reduced chi2 ~1.
        assert jnp.isfinite(result.chi2)
        assert result.chi2 <= chi2_one + 1e-6 * chi2_one
        assert result.reduced_chi2 < 2.0

    @pytest.mark.slow
    def test_design_matrix_shape(self, ngc6440e):
        pint_model, toas = ngc6440e
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, _noise = build_timing_model(pint_model)

        M = compute_design_matrix(jax_model, toa_data, params)
        assert M.shape == (toa_data.n_toas, params.n_free + 1)
        assert not jnp.any(jnp.isnan(M))


# ---------------------------------------------------------------------------
# Tests: NGC6440E astrometry (comparison against PINT)
# ---------------------------------------------------------------------------


class TestNGC6440EAstrometry:
    """Compare JaxPINT astrometry delay and fitted positions against PINT."""

    @pytest.mark.slow
    def test_geometric_delay_matches_pint(self, ngc6440e):
        """Roemer delay from JaxPINT matches PINT's solar_system_geometric_delay."""
        pint_model, toas = ngc6440e

        # PINT delay
        pint_delay = pint_model.components[
            "AstrometryEquatorial"
        ].solar_system_geometric_delay(toas)

        # JaxPINT delay
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, _noise = build_timing_model(pint_model)

        from jaxpint.delay.astrometry import AstrometryEquatorial

        for comp in jax_model.delay_components:
            if isinstance(comp, AstrometryEquatorial):
                jax_delay = comp(toa_data, params, jnp.zeros(toa_data.n_toas))
                break

        np.testing.assert_allclose(
            np.array(jax_delay), pint_delay.to(u.s).value, rtol=1e-10
        )

    @pytest.mark.slow
    def test_fit_converges_with_astrometry(self, ngc6440e):
        """JaxPINT fit with RAJ/DECJ free converges to a good chi2."""
        pint_model, toas = ngc6440e

        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, _noise = build_timing_model(pint_model)
        fitter = WLSFitter(jax_model, toa_data, params)
        result = fitter.fit_toas(maxiter=5)

        # With missing components (SolarWindDispersion)
        # the reduced chi2 won't be ~1, but the fit should still converge.
        assert result.reduced_chi2 < 1000
        assert "RAJ" in result.params.free_names()
        assert "DECJ" in result.params.free_names()
        assert result.parameter_uncertainties is not None
        assert jnp.all(result.parameter_uncertainties > 0)


# ---------------------------------------------------------------------------
# Convergence detection and the maxiter default
#
# PINT's plain WLSFitter/GLSFitter run exactly `maxiter` Gauss-Newton steps with
# no convergence test and default maxiter=1; only its Downhill fitters iterate
# to convergence. JaxPINT's fitters early-exit once the step is negligible
# against the parameter uncertainty, so the default cap is the downhill one (10)
# -- an already-converged fit costs one step and stops, so the higher cap only
# ever charges cold starts.
# ---------------------------------------------------------------------------

_DATA = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"


def _ngc6440e_fitter():
    import jaxpint.par as jpar
    from jaxpint import build_model, native
    from jaxpint.fitters import WLSFitter as JaxWLSFitter

    parsed = jpar.get_model(str(_DATA / "NGC6440E.par"))
    toa_data = native.get_TOAs(str(_DATA / "NGC6440E.tim"), parsed)
    tm, nm = build_model(parsed, toa_data)
    return JaxWLSFitter(tm, toa_data, parsed.params, noise_model=nm), parsed


def test_default_maxiter_is_the_downhill_default():
    """The default is 10 (PINT's Downhill value), not 1 (its plain-WLS value)."""
    from jaxpint.fitters._base import _DEFAULT_MAXITER

    assert _DEFAULT_MAXITER == 10


def test_fit_converges_and_reports_it():
    fitter, _parsed = _ngc6440e_fitter()
    res = fitter.fit_toas()
    assert bool(res.converged)
    assert float(res.step_sigma) < 1e-3
    assert float(res.reduced_chi2) == pytest.approx(1.0638, abs=1e-3)


def test_converged_flag_is_false_on_a_cold_start():
    """The flag must be able to say False, or it certifies nothing.

    A converged fit reporting True proves little on its own -- NGC6440E is
    close enough to its solution that even a single step lands within
    tolerance. Perturbing F0 by 1e-6 Hz is far enough out that one step
    cannot recover, so this pins the discriminating case.
    """
    fitter, parsed = _ngc6440e_fitter()
    names = list(parsed.params.names)
    values = np.asarray(parsed.params.values).copy()
    values[names.index("F0")] += 1e-6
    cold = parsed.params.with_values(jnp.asarray(values))

    one = fitter.fit_toas(maxiter=1, params=cold)
    assert not bool(one.converged)
    assert float(one.step_sigma) > 1.0


def test_convergence_does_not_imply_correctness():
    """A converged fit can still be the wrong solution -- documented, so pinned.

    With nearest-pulse phase tracking, a cold start can settle into a
    cycle-slipped stationary point (the cure -- absolute pulse-number
    tracking -- is demonstrated on this exact start in
    tests/test_pulse_numbers.py). It converges cleanly and reports a wildly
    bad chi2. If this ever starts recovering the true solution, the warning on
    ``BaseFitResult.converged`` is stale and should be removed.
    """
    fitter, parsed = _ngc6440e_fitter()
    names = list(parsed.params.names)
    values = np.asarray(parsed.params.values).copy()
    values[names.index("F0")] += 1e-6
    cold = parsed.params.with_values(jnp.asarray(values))

    res = fitter.fit_toas(maxiter=10, params=cold)
    assert bool(res.converged)  # stopped moving...
    assert float(res.reduced_chi2) > 100.0  # ...somewhere wrong


def test_early_exit_matches_a_longer_run():
    """Raising maxiter past convergence changes nothing -- the loop exits early.

    This is what makes the higher default free: if the loop ran the full count
    regardless, the default change would multiply every fit's cost by 10.
    """
    fitter, _parsed = _ngc6440e_fitter()
    short = fitter.fit_toas(maxiter=10)
    long = fitter.fit_toas(maxiter=200)
    assert float(short.chi2) == pytest.approx(float(long.chi2), rel=1e-12)
    np.testing.assert_allclose(
        np.asarray(short.params.values), np.asarray(long.params.values), rtol=1e-12
    )
