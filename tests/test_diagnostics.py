"""Tests for post-fit diagnostics (whitened residuals, normality tests, ftest).

PINT-free except the two slow parity checks at the bottom (function-level
importorskip). The whitening tests validate against dense linear algebra, not
against another JaxPINT code path. 
"""

from __future__ import annotations


import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.fitters import (
    ftest,
    ftest_results,
    normality_tests,
    whiten_residuals,
)
from jaxpint.noise import EcorrNoise, NoiseModel, ScaleToaError
from tests.helpers import make_params, make_toa_data

# ---------------------------------------------------------------------------
# Shared small noise setup: EFAC/EQUAD white noise + 3-epoch ECORR
# ---------------------------------------------------------------------------


def _setup(n_toas=24, n_epochs=3):
    U_np = np.zeros((n_toas, n_epochs))
    per = n_toas // n_epochs
    for i in range(n_epochs):
        U_np[i * per : (i + 1) * per, i] = 1.0
    mask = np.ones(n_toas, dtype=bool)
    toa_data = make_toa_data(
        n_toas=n_toas, error=1e-6, flag_masks={"EFAC1": mask, "EQUAD1": mask}
    )
    params = make_params(
        ("EFAC1", "EQUAD1", "ECORR1"),
        [1.2, 0.5e-6, 1.5e-6],
        units=("", "s", "s"),
    )
    white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    ecorr = EcorrNoise(
        ecorr_names=("ECORR1",),
        quantization_matrix=jnp.array(U_np),
        ecorr_epoch_slices=((0, n_epochs),),
    )
    nm = NoiseModel(white_noise=white, correlated=(ecorr,))
    return toa_data, params, nm


def _dense_whiten(r, toa_data, params, nm):
    """Reference: whitening via explicit dense inverses, no Woodbury."""
    sigma = np.asarray(nm.scaled_sigma(toa_data, params))
    Ndiag, U, Phi = (np.asarray(x) for x in nm.covariance(toa_data, params))
    C = np.diag(Ndiag) + U @ np.diag(Phi) @ U.T
    b = np.diag(Phi) @ U.T @ np.linalg.solve(C, r)
    return (r - U @ b) / sigma


class TestWhitenResiduals:
    def test_matches_dense_linear_algebra(self):
        toa_data, params, nm = _setup()
        r = np.asarray(jax.random.normal(jax.random.PRNGKey(0), (24,))) * 2e-6
        w = np.asarray(whiten_residuals(jnp.asarray(r), toa_data, params, nm))
        npt.assert_allclose(w, _dense_whiten(r, toa_data, params, nm), rtol=1e-10)

    def test_supplied_realizations_short_circuit(self):
        """Passing the conditional mean explicitly gives the identical answer.

        This is the GLS-result path (noise_realizations from the fit); it must
        agree with the self-computed path at the same params, or the two
        entry points would quietly whiten different things.
        """
        toa_data, params, nm = _setup()
        r = np.asarray(jax.random.normal(jax.random.PRNGKey(1), (24,))) * 2e-6
        Ndiag, U, Phi = (np.asarray(x) for x in nm.covariance(toa_data, params))
        C = np.diag(Ndiag) + U @ np.diag(Phi) @ U.T
        b = np.diag(Phi) @ U.T @ np.linalg.solve(C, r)
        w_auto = whiten_residuals(jnp.asarray(r), toa_data, params, nm)
        w_given = whiten_residuals(
            jnp.asarray(r), toa_data, params, nm, noise_realizations=jnp.asarray(b)
        )
        npt.assert_allclose(np.asarray(w_given), np.asarray(w_auto), rtol=1e-10)

    def test_white_only_reduces_to_scaled_sigma(self):
        toa_data, params, _ = _setup()
        white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
        nm = NoiseModel(white_noise=white, correlated=())
        r = jnp.full(24, 3e-6)
        w = whiten_residuals(r, toa_data, params, nm)
        sigma = nm.scaled_sigma(toa_data, params)
        npt.assert_allclose(np.asarray(w), np.asarray(r / sigma), rtol=1e-12)

    def test_jit_compatible(self):
        toa_data, params, nm = _setup()
        r = jnp.ones(24) * 1e-6
        w_eager = whiten_residuals(r, toa_data, params, nm)
        # NoiseModel is an eqx.Module, i.e. a pytree -- it traces, no
        # static_argnums (and it is not hashable, so static would fail).
        w_jit = jax.jit(whiten_residuals)(r, toa_data, params, nm)
        npt.assert_allclose(np.asarray(w_jit), np.asarray(w_eager), rtol=1e-12)


class TestNormalityTests:
    def test_calibration_true_null_passes(self):
        """Standard-normal draws must pass both tests -- and be seen to pass.

        Fixed key: this is a regression pin, not a statistical experiment.
        """
        w = jax.random.normal(jax.random.PRNGKey(42), (2000,))
        rep = normality_tests(w)
        assert rep.ks_p > 0.05
        assert rep.ad_stat < rep.ad_critical[0.05]

    def test_detection_unwhitened_fails(self):
        """The discriminating case: correlated, unscaled residuals must FAIL.

        Without this, the calibration test above proves nothing -- a
        normality_tests that always returned p=0.5 would pass it.
        """
        key = jax.random.PRNGKey(7)
        # Strongly correlated + wrong scale: a random walk of unit steps.
        w = jnp.cumsum(jax.random.normal(key, (2000,)))
        rep = normality_tests(w)
        assert rep.ks_p < 1e-6
        assert rep.ad_stat > rep.ad_critical[0.01]

    def test_scale_error_detected(self):
        """sigma mis-scaled by 2x: marginal-looking residuals, decisively caught.

        This is the failure mode whitening exists to expose (an EFAC that
        should have been fit); KS against N(0,1) is fully specified, so a
        scale error is power, not nuisance.
        """
        w = 2.0 * jax.random.normal(jax.random.PRNGKey(3), (2000,))
        rep = normality_tests(w)
        assert rep.ks_p < 1e-6
        assert rep.ad_stat > rep.ad_critical[0.01]

    def test_case0_critical_values_are_stephens(self):
        """The AD null here is fully-specified N(0,1) (case 0), NOT scipy's
        estimated-parameters case; pin the table so nobody 'fixes' it to
        scipy.stats.anderson's values (case 3, a different null)."""
        rep = normality_tests(jax.random.normal(jax.random.PRNGKey(0), (100,)))
        assert rep.ad_critical[0.05] == 2.492
        assert rep.ad_critical[0.01] == 3.857

    def test_too_few_residuals_raises(self):
        with pytest.raises(ValueError, match=">= 8"):
            normality_tests(jnp.ones(4))


class TestFtest:
    def test_known_value_against_scipy(self):
        """F and p reproduce the direct scipy computation."""
        from scipy.stats import f as fdist

        res = ftest(120.0, 100, 100.0, 98)
        f_expected = ((120.0 - 100.0) / 2) / (100.0 / 98)
        assert res.f_stat == pytest.approx(f_expected, rel=1e-12)
        assert res.p == pytest.approx(fdist.sf(f_expected, 2, 98), rel=1e-12)

    def test_equal_dof_warns_nan(self):
        with pytest.warns(UserWarning, match="equal degrees"):
            res = ftest(120.0, 100, 100.0, 100)
        assert np.isnan(res.p) and np.isnan(res.f_stat)

    def test_no_improvement_is_p_one(self):
        with pytest.warns(UserWarning, match="did not improve"):
            res = ftest(100.0, 100, 100.0, 98)
        assert res.p == 1.0

    def test_swapped_arguments_raise(self):
        """dof_simple < dof_complex is an argument-order mistake, not a result."""
        with pytest.raises(ValueError, match="Swap the arguments"):
            ftest(100.0, 98, 120.0, 100)

    def test_ftest_results_reads_fit_results(self):
        class _R:
            def __init__(self, chi2, dof):
                self.chi2, self.dof = chi2, dof

        res = ftest_results(_R(120.0, 100), _R(100.0, 98))
        assert res.p == ftest(120.0, 100, 100.0, 98).p

    @pytest.mark.slow
    def test_matches_pint_ftest(self):
        """Numeric parity with pint.utils.FTest over a value grid."""
        pytest.importorskip("pint")
        from pint.utils import FTest

        for chi2_1, dof_1, chi2_2, dof_2 in [
            (120.0, 100, 100.0, 98),
            (59.6, 56, 55.1, 55),
            (2000.0, 500, 1500.0, 490),
        ]:
            ours = ftest(chi2_1, dof_1, chi2_2, dof_2).p
            theirs = float(FTest(chi2_1, dof_1, chi2_2, dof_2))
            assert ours == pytest.approx(theirs, rel=1e-12), (chi2_1, dof_1)


# ---------------------------------------------------------------------------
# End-to-end: whitening a real GLS fit, and an ftest on nested real fits
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_whitened_gls_fit_is_standard_normal():
    """Simulate noise from a real model, fit GLS, whiten via the new API.

    The statistical assertions mirror what test_noise_generation checks with
    its hand-rolled whitening; here the API under test does the work.
    """
    import pathlib

    import jaxpint.par as jpar
    from jaxpint.fitters import GLSFitter
    from jaxpint.simulation import make_fake_toas_uniform

    par = jpar.get_model(
        str(
            pathlib.Path(__file__).resolve().parent
            / "data"
            / "pint_inputs"
            / "B1855+09_NANOGrav_9yv1.gls.par"
        )
    )
    # Realize the model, then add noise per the par's own noise description.
    from jaxpint import build_model

    toa_data = make_fake_toas_uniform(
        53400.0, 56500.0, 400, par, obs="ao", freq_mhz=1400.0, error_us=1.0
    )
    model, nm = build_model(par, toa_data)
    from jaxpint.simulation import make_fake_toas

    comps = ([nm.white_noise] if nm.white_noise is not None else []) + list(
        nm.correlated
    )
    toa_data = make_fake_toas(
        model, toa_data, par.params, jax.random.PRNGKey(11), noise_components=comps
    )

    fit = GLSFitter(model, toa_data, par.params, noise_model=nm).fit_toas(maxiter=3)
    w = whiten_residuals(
        fit.residuals,
        toa_data,
        fit.params,
        nm,
        noise_realizations=fit.noise_realizations,
    )
    rep = normality_tests(w)
    assert abs(float(jnp.std(w)) - 1.0) < 0.15
    assert rep.ks_p > 0.01
    assert rep.ad_stat < rep.ad_critical[0.01]


@pytest.mark.slow
def test_ftest_on_nested_real_fits():
    """Freezing F1 out of NGC6440E: the F-test must scream.

    F1 is decisively detected in this dataset, so chi2(without F1) >> chi2
    (with), and p should be ~0. The reverse comparison (adding a parameter the
    data does not need) is covered by the p ~ U(0,1) behaviour asserted in
    unit tests; here the point is the wiring of two real fits end to end.
    """
    import pathlib

    import jaxpint.par as jpar
    from jaxpint import WLSFitter, build_model, native
    from jaxpint.types import ParameterVector

    d = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
    parsed = jpar.get_model(str(d / "NGC6440E.par"))
    toa_data = native.get_TOAs(str(d / "NGC6440E.tim"), parsed)
    model, nm = build_model(parsed, toa_data)

    full = WLSFitter(model, toa_data, parsed.params, noise_model=nm).fit_toas()

    p0 = parsed.params
    i_f1 = list(p0.names).index("F1")
    frozen_mask = tuple(f or (k == i_f1) for k, f in enumerate(p0.frozen_mask))

    def _fit_with_f1_frozen_at(f1_value):
        pv = ParameterVector(
            values=p0.values.at[i_f1].set(f1_value),
            frozen_mask=frozen_mask,
            names=p0.names,
            units=p0.units,
            epoch_int_values=p0.epoch_int_values,
        )
        return WLSFitter(model, toa_data, pv, noise_model=nm).fit_toas()

    f1 = float(np.asarray(p0.values)[i_f1])
    # NGC6440E.par carries no uncertainty column, so take sigma(F1) from the
    # full fit itself -- the more principled scale for "how far off is
    # detectable" anyway.
    free_names = [n for n, fr in zip(p0.names, p0.frozen_mask) if not fr]
    unc = float(np.asarray(full.parameter_uncertainties)[free_names.index("F1")])
    assert np.isfinite(unc) and unc > 0

    # Null direction: freezing F1 at its already-converged value costs the fit
    # nothing, so freeing it is NOT warranted -- p must be large. (The first
    # version of this test asserted p < 1e-10 here and failed with p = 0.82:
    # the test was wrong, not the code. A par file records a converged
    # solution, so its own F1 is the one value freezing is free at.)
    at_converged = _fit_with_f1_frozen_at(f1)
    res_null = ftest_results(at_converged, full)
    assert at_converged.dof == full.dof + 1
    assert res_null.p > 0.05

    # Detection direction: freeze F1 15 sigma off its solution; the richer
    # model now buys ~15^2 in chi2 for one dof and the F-test must scream.
    off = _fit_with_f1_frozen_at(f1 + 15.0 * unc)
    res_det = ftest_results(off, full)
    assert float(off.chi2) > float(full.chi2) + 100.0
    assert res_det.p < 1e-6
