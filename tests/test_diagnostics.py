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


# ---------------------------------------------------------------------------
# Wideband whitening
# ---------------------------------------------------------------------------


def _wb_setup(n_toas=24, n_epochs=3):
    """Wideband toy: the narrowband _setup plus DM values/errors."""
    toa_data, params, nm = _setup(n_toas, n_epochs)
    import equinox as eqx

    toa_data = eqx.tree_at(
        lambda t: (t.dm_values, t.dm_errors),
        toa_data,
        (jnp.full(n_toas, 15.0), jnp.full(n_toas, 1e-4)),
        is_leaf=lambda x: x is None,
    )
    return toa_data, params, nm


class TestWidebandWhitening:
    def test_matches_dense_linear_algebra(self):
        """Validate the stacked system against explicit dense inverses."""
        from jaxpint.fitters import whiten_wideband_residuals
        from jaxpint.fitters.wideband import stack_wideband_noise

        toa_data, params, nm = _wb_setup()
        key = jax.random.PRNGKey(5)
        rt = np.asarray(jax.random.normal(key, (24,))) * 2e-6
        rd = np.asarray(jax.random.normal(jax.random.PRNGKey(6), (24,))) * 2e-4

        wt, wd = whiten_wideband_residuals(
            jnp.asarray(rt), jnp.asarray(rd), toa_data, params, nm
        )

        sigma_toa, Ndiag, U, Phi, sigma_dm = (
            np.asarray(x) for x in stack_wideband_noise(nm, toa_data, params)
        )
        r = np.concatenate([rt, rd])
        C = np.diag(Ndiag) + U @ np.diag(Phi) @ U.T
        b = np.diag(Phi) @ U.T @ np.linalg.solve(C, r)
        w_ref = (r - U @ b) / np.concatenate([sigma_toa, sigma_dm])
        npt.assert_allclose(np.asarray(wt), w_ref[:24], rtol=1e-10)
        npt.assert_allclose(np.asarray(wd), w_ref[24:], rtol=1e-10)

    def test_dm_block_is_white_only(self):
        """DM whitening is division by scaled_dm_sigma -- pinned deliberately.

        The stacked U has zero DM-block rows (wideband_covariance models DM
        as white-only). If this test ever fails because DM rows gained GP
        support, that is the tracked noise-model gap closing: update this to
        the dense reference, don't delete it.
        """
        from jaxpint.fitters import whiten_wideband_residuals

        toa_data, params, nm = _wb_setup()
        rt = jnp.zeros(24)
        rd = jnp.asarray(np.linspace(-3e-4, 3e-4, 24))
        _, wd = whiten_wideband_residuals(rt, rd, toa_data, params, nm)
        sigma_dm = nm.scaled_dm_sigma(toa_data, params)
        npt.assert_allclose(np.asarray(wd), np.asarray(rd / sigma_dm), rtol=1e-12)

    def test_none_noise_model_uses_raw_errors(self):
        """noise_model=None mirrors the fitter: raw TOA/DM errors, no basis."""
        from jaxpint.fitters import whiten_wideband_residuals

        toa_data, params, _ = _wb_setup()
        rt = jnp.full(24, 2e-6)
        rd = jnp.full(24, 2e-4)
        wt, wd = whiten_wideband_residuals(rt, rd, toa_data, params, None)
        npt.assert_allclose(np.asarray(wt), np.asarray(rt / toa_data.error), rtol=1e-12)
        npt.assert_allclose(
            np.asarray(wd), np.asarray(rd / toa_data.dm_errors), rtol=1e-12
        )

    @pytest.mark.slow
    def test_wideband_fit_whitens_to_standard_normal(self):
        """End-to-end on the vendored J1614 wideband pair, native path.

        Both blocks must come out ~N(0,1); the fit's own noise_realizations
        feed the whitening, exactly as a user would do it.
        """
        import pathlib

        import jaxpint.par as jpar
        from jaxpint import build_model, native
        from jaxpint.fitters import (
            WidebandGLSFitter,
            whiten_wideband_residuals,
        )

        d = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
        parsed = jpar.get_model(str(d / "J1614-2230_NANOGrav_12yv3.wb.gls.par"))
        toa_data = native.get_TOAs(str(d / "J1614-2230_NANOGrav_12yv3.wb.tim"), parsed)
        model, nm = build_model(parsed, toa_data)
        fit = WidebandGLSFitter(model, toa_data, parsed.params, noise_model=nm).fit_toas(
            maxiter=3
        )
        wt, wd = whiten_wideband_residuals(
            fit.time_residuals,
            fit.dm_residuals,
            toa_data,
            fit.params,
            nm,
            noise_realizations=fit.noise_realizations,
        )
        # Real data, real (already-published) noise model: the whitened blocks
        # should be near-standard-normal. Bounds are loose-ish because this is
        # observed data, not a simulation drawn from the model itself.
        assert abs(float(jnp.std(wt)) - 1.0) < 0.25
        assert abs(float(jnp.std(wd)) - 1.0) < 0.25
        rep_t = normality_tests(wt)
        assert rep_t.ks_p > 1e-4

    @pytest.mark.slow
    def test_parity_vs_pint_wideband_whitening(self):
        """Element-wise vs PINT's calc_wideband_whitened_resids, same params.

        Both sides evaluate at the PAR values (no fit on either side), so the
        comparison isolates the whitening machinery from fit-state
        differences. J1614's noise is DMX + white + ECORR -- no DM GP -- so
        the documented DM-block divergence cannot enter here.
        """
        pytest.importorskip("pint")
        import pint.models
        import pint.residuals
        import pint.toa
        from pint.config import examplefile

        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitters import whiten_wideband_residuals
        from jaxpint.fitters.wideband import compute_wideband_residuals

        par = examplefile("J1614-2230_NANOGrav_12yv3.wb.gls.par")
        tim = examplefile("J1614-2230_NANOGrav_12yv3.wb.tim")
        m = pint.models.get_model(par)
        toas = pint.toa.get_TOAs(tim, model=m)

        pres = pint.residuals.WidebandTOAResiduals(toas, m)
        w_pint = pres.calc_wideband_whitened_resids()

        toa_data = pint_toas_to_jax(toas, model=m)
        params = pint_model_to_params(m).params
        model, nm = build_timing_model(m, toas)
        r = compute_wideband_residuals(model, toa_data, params)
        n = toa_data.n_toas
        # PINT's wideband residuals are mean-subtracted on the time block;
        # match that before whitening (see the narrowband demeaning note).
        rt = r[:n] - jnp.mean(r[:n])
        rd = r[n:]
        wt, wd = whiten_wideband_residuals(rt, rd, toa_data, params, nm)
        ours = np.concatenate([np.asarray(wt), np.asarray(wd)])
        # Measured: max 2.0e-02, median 5.8e-04 (dimensionless). The max is
        # the known residual-level implementation difference (~1e-8 s, per the
        # cross-implementation tolerances) expressed in whitened units
        # (~0.5 us sigma -> 2e-2), not a whitening-machinery error -- the toy
        # tests above pin the machinery against dense algebra at 1e-10.
        diff = np.max(np.abs(ours - np.asarray(w_pint)))
        assert diff < 0.05, f"max whitened-residual diff {diff:.3e} (dimensionless)"


# ---------------------------------------------------------------------------
# Residuals options + epoch averaging
# ---------------------------------------------------------------------------


class TestComputeResiduals:
    @pytest.fixture(scope="class")
    def ngc(self):
        import pathlib

        import jaxpint.par as jpar
        from jaxpint import build_model, native

        d = pathlib.Path(__file__).resolve().parent / "data" / "pint_inputs"
        parsed = jpar.get_model(str(d / "NGC6440E.par"))
        td = native.get_TOAs(str(d / "NGC6440E.tim"), parsed)
        tm, nm = build_model(parsed, td)
        return parsed, td, tm

    def test_subtract_mean_false_is_the_raw_primitive(self, ngc):
        from jaxpint.fitters import compute_residuals, compute_time_residuals

        parsed, td, tm = ngc
        raw = compute_time_residuals(tm, td, parsed.params)
        opt = compute_residuals(tm, td, parsed.params, subtract_mean=False)
        npt.assert_array_equal(np.asarray(opt), np.asarray(raw))

    def test_weighted_mean_is_removed(self, ngc):
        from jaxpint.fitters import compute_residuals

        parsed, td, tm = ngc
        r = np.asarray(compute_residuals(tm, td, parsed.params))
        w = 1.0 / np.asarray(td.error) ** 2
        assert abs(np.sum(w * r) / np.sum(w)) < 1e-15  # weighted mean gone
        r_uw = np.asarray(
            compute_residuals(tm, td, parsed.params, use_weighted_mean=False)
        )
        assert abs(r_uw.mean()) < 1e-15  # plain mean gone
        # and the two demeanings genuinely differ (errors are heterogeneous)
        assert np.abs(r - r_uw).max() > 0


class TestEcorrAverage:
    def test_toy_matches_hand_computation(self):
        """3 epochs of 8 TOAs (the shared _setup): every output checked
        against the formula computed longhand."""
        from jaxpint.fitters import ecorr_average

        toa_data, params, nm = _setup()
        rng = np.random.default_rng(0)
        r = rng.normal(0, 2e-6, 24)

        avg = ecorr_average(r, toa_data, params, nm)
        err = np.asarray(nm.scaled_sigma(toa_data, params))
        w = 1.0 / err**2
        ecorr2 = 1.5e-6**2
        for e in range(3):
            sl = slice(8 * e, 8 * (e + 1))
            npt.assert_allclose(
                avg.time_resids[e], np.sum(w[sl] * r[sl]) / np.sum(w[sl]), rtol=1e-12
            )
            npt.assert_allclose(
                avg.errors[e], np.sqrt(1.0 / np.sum(w[sl]) + ecorr2), rtol=1e-12
            )
            npt.assert_array_equal(avg.indices[e], np.arange(24)[sl])

    def test_raw_error_mode_drops_ecorr_term(self):
        from jaxpint.fitters import ecorr_average

        toa_data, params, nm = _setup()
        r = np.zeros(24)
        avg = ecorr_average(r, toa_data, params, nm, use_noise_model=False)
        w = 1.0 / np.asarray(toa_data.error) ** 2
        npt.assert_allclose(
            avg.errors[0], np.sqrt(1.0 / np.sum(w[:8])), rtol=1e-12
        )

    def test_requires_ecorr(self):
        from jaxpint.fitters import ecorr_average
        from jaxpint.noise import NoiseModel, ScaleToaError

        toa_data, params, _ = _setup()
        nm = NoiseModel(
            white_noise=ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",)),
            correlated=(),
        )
        with pytest.raises(ValueError, match="EcorrNoise"):
            ecorr_average(np.zeros(24), toa_data, params, nm)

    @pytest.mark.slow
    def test_parity_vs_pint_ecorr_average(self):
        """Element-wise vs PINT's Residuals.ecorr_average on B1855 real data."""
        pytest.importorskip("pint")
        import pint.models
        import pint.residuals
        import pint.toa
        from pint.config import examplefile

        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitters import compute_residuals, ecorr_average

        m = pint.models.get_model(examplefile("B1855+09_NANOGrav_9yv1.gls.par"))
        toas = pint.toa.get_TOAs(examplefile("B1855+09_NANOGrav_9yv1.tim"), model=m)
        pres = pint.residuals.Residuals(toas, m)
        pavg = pres.ecorr_average()

        td = pint_toas_to_jax(toas, model=m)
        params = pint_model_to_params(m).params
        tm, nm = build_timing_model(m, toas)
        # PINT demeans with scaled uncertainties; match, or the removed
        # constant differs by ~1 us and every averaged residual shifts.
        r = compute_residuals(
            tm, td, params, errors=nm.scaled_sigma(td, params)
        )
        avg = ecorr_average(np.asarray(r), td, params, nm)

        assert avg.time_resids.shape == pavg["time_resids"].shape
        # Averaged MJDs and errors: pure bookkeeping, must agree tightly.
        npt.assert_allclose(
            avg.mjds, pavg["mjds"].value, rtol=0, atol=1e-7
        )
        npt.assert_allclose(
            avg.errors, pavg["errors"].to("s").value, rtol=1e-6
        )
        # Averaged residuals: carries the known ~1e-8 s residual-level
        # implementation difference, so a matched tolerance.
        npt.assert_allclose(
            avg.time_resids, pavg["time_resids"].to("s").value, atol=5e-8
        )
