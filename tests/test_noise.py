"""Tests for the noise model (EFAC/EQUAD white noise scaling).

Covers unit tests with synthetic data and integration tests comparing
against PINT's ScaleToaError.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.noise import ScaleToaError

import astropy.units as u

from tests.helpers import make_toa_data, make_noise_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_toa_data(n_toas, errors, flag_masks):
    return make_toa_data(n_toas, tdb_int=0.0, tdb_frac=0.0,
                         error=errors, flag_masks=flag_masks,
                         planet_positions=None)


# ---------------------------------------------------------------------------
# Unit tests: ScaleToaError
# ---------------------------------------------------------------------------


class TestScaleToaErrorUnit:
    """Unit tests for EFAC/EQUAD scaling with synthetic data."""

    def test_efac_only(self):
        """EFAC multiplies errors on masked TOAs, leaves others unchanged."""
        n = 6
        errors = np.full(n, 1e-6)  # 1 µs
        mask = np.array([True, True, True, False, False, False])
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})
        params = make_noise_params(["EFAC1"], [1.5])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())
        sigma = noise.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(sigma[:3], 1.5e-6)
        np.testing.assert_allclose(sigma[3:], 1e-6)

    def test_equad_only(self):
        """EQUAD is added in quadrature to masked TOAs."""
        n = 4
        errors = np.full(n, 3e-6)  # 3 µs
        equad = 4e-6  # 4 µs → sqrt(9+16) = 5 µs
        mask = np.array([True, True, False, False])
        toa_data = _make_toa_data(n, errors, {"EQUAD1": mask})
        params = make_noise_params(["EQUAD1"], [equad])

        noise = ScaleToaError(efac_names=(), equad_names=("EQUAD1",))
        sigma = noise.scaled_sigma(toa_data, params)

        expected_masked = np.sqrt(3e-6**2 + 4e-6**2)
        np.testing.assert_allclose(sigma[:2], expected_masked)
        np.testing.assert_allclose(sigma[2:], 3e-6)

    def test_efac_and_equad(self):
        """EFAC × √(σ² + EQUAD²) matches PINT convention."""
        n = 4
        errors = np.full(n, 3e-6)
        efac = 1.2
        equad = 4e-6
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask, "EQUAD1": mask})
        params = make_noise_params(["EFAC1", "EQUAD1"], [efac, equad])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
        sigma = noise.scaled_sigma(toa_data, params)

        expected = efac * np.sqrt(3e-6**2 + equad**2)
        np.testing.assert_allclose(sigma, expected, rtol=1e-14)

    def test_multiple_masks_disjoint(self):
        """Two EFAC/EQUAD pairs on disjoint TOA subsets."""
        n = 6
        errors = np.full(n, 1e-6)
        mask_a = np.array([True, True, True, False, False, False])
        mask_b = np.array([False, False, False, True, True, True])
        toa_data = _make_toa_data(
            n,
            errors,
            {"EFAC1": mask_a, "EFAC2": mask_b, "EQUAD1": mask_a, "EQUAD2": mask_b},
        )
        params = make_noise_params(
            ["EFAC1", "EFAC2", "EQUAD1", "EQUAD2"],
            [1.5, 2.0, 0.5e-6, 1.0e-6],
        )

        noise = ScaleToaError(
            efac_names=("EFAC1", "EFAC2"),
            equad_names=("EQUAD1", "EQUAD2"),
        )
        sigma = noise.scaled_sigma(toa_data, params)

        expected_a = 1.5 * np.sqrt(1e-6**2 + 0.5e-6**2)
        expected_b = 2.0 * np.sqrt(1e-6**2 + 1.0e-6**2)
        np.testing.assert_allclose(sigma[:3], expected_a, rtol=1e-14)
        np.testing.assert_allclose(sigma[3:], expected_b, rtol=1e-14)

    def test_jit_compatible(self):
        """scaled_sigma works under jax.jit."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})
        params = make_noise_params(["EFAC1"], [1.3])

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        sigma_eager = noise.scaled_sigma(toa_data, params)
        sigma_jit = jax.jit(noise.scaled_sigma)(toa_data, params)

        np.testing.assert_allclose(sigma_jit, sigma_eager, rtol=1e-15)

    def test_differentiable_efac(self):
        """Can differentiate chi2 through scaled_sigma w.r.t. EFAC."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EFAC1": mask})

        noise = ScaleToaError(efac_names=("EFAC1",), equad_names=())

        def chi2_fn(efac_val):
            p = make_noise_params(["EFAC1"], [efac_val])
            sigma = noise.scaled_sigma(toa_data, p)
            residuals = jnp.ones(n) * 1e-6  # dummy residuals
            return jnp.sum((residuals / sigma) ** 2)

        grad = jax.grad(chi2_fn)(1.0)
        assert jnp.isfinite(grad)
        # Increasing EFAC increases sigma, decreases chi2 → grad < 0
        assert grad < 0

    def test_differentiable_equad(self):
        """Can differentiate chi2 through scaled_sigma w.r.t. EQUAD."""
        n = 4
        errors = np.full(n, 1e-6)
        mask = np.ones(n, dtype=bool)
        toa_data = _make_toa_data(n, errors, {"EQUAD1": mask})

        noise = ScaleToaError(efac_names=(), equad_names=("EQUAD1",))

        def chi2_fn(equad_val):
            p = make_noise_params(["EQUAD1"], [equad_val])
            sigma = noise.scaled_sigma(toa_data, p)
            residuals = jnp.ones(n) * 1e-6
            return jnp.sum((residuals / sigma) ** 2)

        grad = jax.grad(chi2_fn)(0.5e-6)
        assert jnp.isfinite(grad)
        assert grad < 0

    def test_no_noise_returns_raw_errors(self):
        """Empty EFAC/EQUAD lists return raw TOA errors unchanged."""
        n = 4
        errors = np.array([1e-6, 2e-6, 3e-6, 4e-6])
        toa_data = _make_toa_data(n, errors, {})
        params = make_noise_params([], [])

        noise = ScaleToaError(efac_names=(), equad_names=())
        sigma = noise.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(sigma, errors)


# ---------------------------------------------------------------------------
# Integration tests against PINT
# ---------------------------------------------------------------------------


class TestScaleToaErrorVsPINT:
    """Compare JaxPINT noise scaling against PINT on real/synthetic data."""

    @pytest.fixture(scope="class")
    def synthetic_with_noise(self):
        """Create a synthetic pulsar with EFAC/EQUAD noise parameters."""
        import io
        import pint.models as models
        import pint.toa as toa
        from pint.simulation import make_fake_toas_uniform

        np.random.seed(42)

        par = """\
PSR           J0000+0000
RAJ           00:00:00   1
DECJ          00:00:00   1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
EFAC -f L-wide 1.3
EQUAD -f L-wide 0.8
EFAC -f S-wide 1.1
EQUAD -f S-wide 0.5
TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""
        m = models.get_model(io.StringIO(par))
        # Create fake TOAs with two "backends"
        t1 = make_fake_toas_uniform(
            54000, 56000, 50, model=m, obs="gbt", freq=1400.0,
            error=1.0 * u.us, add_noise=True,
        )
        t2 = make_fake_toas_uniform(
            54000, 56000, 50, model=m, obs="gbt", freq=2000.0,
            error=1.5 * u.us, add_noise=True,
        )
        # Manually set flags for the backends
        for i in range(t1.ntoas):
            t1.table["flags"][i]["f"] = "L-wide"
        for i in range(t2.ntoas):
            t2.table["flags"][i]["f"] = "S-wide"
        t = toa.merge_TOAs([t1, t2])
        return m, t

    @pytest.mark.slow
    def test_scaled_sigma_matches_pint(self, synthetic_with_noise):
        """JaxPINT scaled sigma matches PINT's scaled_toa_uncertainty."""
        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )

        pint_model, toas = synthetic_with_noise

        # PINT reference
        pint_sigma = pint_model.scaled_toa_uncertainty(toas).to(u.s).value

        # JaxPINT
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        _tm, noise_model = build_timing_model(pint_model)

        assert noise_model is not None
        jax_sigma = noise_model.scaled_sigma(toa_data, params)

        np.testing.assert_allclose(
            np.array(jax_sigma), pint_sigma, rtol=1e-12,
            err_msg="JaxPINT scaled sigma does not match PINT",
        )


    @pytest.mark.slow
    def test_wls_chi2_matches_pint(self, synthetic_with_noise):
        """WLS fit with EFAC/EQUAD: JaxPINT chi2 matches PINT."""
        import copy

        from pint.fitter import WLSFitter as PINTWLSFitter

        from jaxpint.bridge import (
            build_timing_model,
            pint_model_to_params,
            pint_toas_to_jax,
        )
        from jaxpint.fitters import WLSFitter

        pint_model, toas = synthetic_with_noise

        # --- PINT WLS fit ---
        m_pint = copy.deepcopy(pint_model)
        pint_fitter = PINTWLSFitter(toas, m_pint)
        pint_fitter.fit_toas(maxiter=1)
        pint_chi2 = pint_fitter.resids.chi2

        # --- JaxPINT WLS fit ---
        toa_data = pint_toas_to_jax(toas, model=pint_model)
        params = pint_model_to_params(pint_model).params
        jax_model, noise_model = build_timing_model(pint_model)

        assert noise_model is not None
        fitter = WLSFitter(jax_model, toa_data, params, noise_model=noise_model)
        jax_result = fitter.fit_toas(maxiter=1)
        jax_chi2 = jax_result.chi2

        # rtol=0.01: JaxPINT's int/frac Horner uses a different (more precise)
        # numerical path than PINT's longdouble taylor_horner, producing
        # small residual differences that accumulate into chi2.
        np.testing.assert_allclose(
            jax_chi2, pint_chi2, rtol=0.01,
            err_msg=f"JaxPINT chi2 ({jax_chi2}) != PINT chi2 ({pint_chi2})",
        )



# ---------------------------------------------------------------------------
# Multi-backend parity vs PINT (many backends, ECORR, TempoNest TNEQ)
# ---------------------------------------------------------------------------


_PAR_HEAD = """PSR           J0000+0000
RAJ           00:00:00   1
DECJ          00:00:00   1
PEPOCH        55000
F0            100        1
F1            -1e-15     1
DM            15         1
"""
_PAR_TAIL = """TZRMJD        55000
TZRFRQ        1400
TZRSITE       @
EPHEM         DE421
CLOCK         TT(BIPM2019)
UNITS         TDB
"""


def _backend_toas(pint_model, backends, *, n_per=12, start=54000, end=56000):
    """Build TOAs carrying a ``-f`` flag per backend, merged into one TOAs object.

    ``make_fake_toas_uniform`` stamps flags uniformly across all its TOAs, so a
    heterogeneous ``-f`` column requires one call per backend plus a merge.
    """
    import pint.toa as toa
    from pint.simulation import make_fake_toas_uniform

    chunks = []
    for i, be in enumerate(backends):
        t = make_fake_toas_uniform(
            start, end, n_per, model=pint_model, obs="gbt",
            freq=1000.0 + 100.0 * i, error=(1.0 + 0.1 * i) * u.us,
        )
        for k in range(t.ntoas):
            t.table["flags"][k]["f"] = be
        chunks.append(t)
    return toa.merge_TOAs(chunks)


def _jax_scaled_sigma(pint_model, toas):
    from jaxpint.bridge import (
        build_timing_model, pint_model_to_params, pint_toas_to_jax,
    )

    toa_data = pint_toas_to_jax(toas, model=pint_model)
    params = pint_model_to_params(pint_model).params
    _tm, noise_model = build_timing_model(pint_model)
    assert noise_model is not None
    return np.array(noise_model.scaled_sigma(toa_data, params))


@pytest.mark.slow
def test_twelve_backends_scaled_sigma_matches_pint():
    """>=10 masked parameters: guards the lexicographic-ordering invariant.

    ``names_with_prefix`` sorts as EFAC1, EFAC10, EFAC11, EFAC2, ... With 12
    backends the sort genuinely interleaves, so this fails loudly if anything
    ever starts depending on positional (rather than name-keyed) ordering.
    """
    import io
    import pint.models as models

    backends = [f"be{i:02d}" for i in range(12)]
    lines = "".join(
        f"EFAC -f {be} {1.0 + 0.05 * i}\nEQUAD -f {be} {0.1 + 0.02 * i}\n"
        for i, be in enumerate(backends)
    )
    m = models.get_model(io.StringIO(_PAR_HEAD + lines + _PAR_TAIL))
    assert len([p for p in m.params if p.startswith("EFAC")]) >= 12

    toas = _backend_toas(m, backends, n_per=6)
    np.testing.assert_allclose(
        _jax_scaled_sigma(m, toas),
        m.scaled_toa_uncertainty(toas).to(u.s).value,
        rtol=1e-12,
        err_msg="12-backend scaled sigma diverges from PINT",
    )


@pytest.mark.slow
def test_tneq_multi_backend_scaled_sigma_matches_pint():
    """TempoNest TNEQ (log10 s) must behave exactly like the equivalent EQUAD.

    Regression: TNEQ was previously undeclared, so the line was dropped by the
    parser and the EQUAD silently vanished from the noise model.
    """
    import io
    import pint.models as models

    backends = ["L-wide", "S-wide"]
    lines = ("EFAC -f L-wide 1.3\nTNEQ -f L-wide -6.5\n"
             "EFAC -f S-wide 1.1\nTNEQ -f S-wide -7.0\n")
    m = models.get_model(io.StringIO(_PAR_HEAD + lines + _PAR_TAIL))

    toas = _backend_toas(m, backends, n_per=20)
    np.testing.assert_allclose(
        _jax_scaled_sigma(m, toas),
        m.scaled_toa_uncertainty(toas).to(u.s).value,
        rtol=1e-12,
        err_msg="TNEQ-derived EQUAD diverges from PINT",
    )


@pytest.mark.slow
def test_multi_backend_ecorr_covariance_matches_pint():
    """Per-backend ECORR: the low-rank basis must reproduce PINT's ECORR covariance.

    ECORR is a basis term, not a masked diagonal -- each parameter owns a
    disjoint block of quantization columns -- so a multi-backend split changes
    the basis structure, which nothing previously covered.
    """
    import io
    import pint.models as models
    import pint.toa as toa
    from pint.simulation import make_fake_toas_fromMJDs

    backends = ["beA", "beB"]
    lines = ("EFAC -f beA 1.0\nECORR -f beA 0.5\n"
             "EFAC -f beB 1.0\nECORR -f beB 0.3\n")
    m = models.get_model(io.StringIO(_PAR_HEAD + lines + _PAR_TAIL))

    # Epoch-clustered MJDs: 3 TOAs ~0.3 s apart per epoch, so each epoch falls
    # inside one dt=1 s quantization bucket and survives the nmin=2 cut *within
    # each backend*. Uniform TOAs would quantize to an empty ECORR basis.
    day = 1.0 / 86400.0
    chunks = []
    for i, be in enumerate(backends):
        mjds = np.concatenate([
            [54000.0 + 30.0 * e + i * 0.5 + k * 0.3 * day for k in range(3)]
            for e in range(20)
        ])
        t = make_fake_toas_fromMJDs(mjds, model=m, obs="gbt",
                                    freq=1000.0 + 100.0 * i, error=1.0 * u.us)
        for k in range(t.ntoas):
            t.table["flags"][k]["f"] = be
        chunks.append(t)
    toas = toa.merge_TOAs(chunks)

    from jaxpint.bridge import (
        build_timing_model, pint_model_to_params, pint_toas_to_jax,
    )

    toa_data = pint_toas_to_jax(toas, model=m)
    params = pint_model_to_params(m).params
    # ECORR is a *basis* term: EcorrNoise.build needs the TOAs to quantize
    # epochs, so the model must be built with them (without, it is silently
    # omitted -- "EcorrNoise found but no toa_data provided").
    _tm, noise_model = build_timing_model(m, toas)
    assert noise_model is not None
    assert any(type(c).__name__ == "EcorrNoise" for c in noise_model.correlated), (
        "EcorrNoise missing from the built noise model"
    )

    _ndiag, U, phi = noise_model.covariance(toa_data, params)
    U = np.asarray(U)
    jax_cov = U @ np.diag(np.asarray(phi)) @ U.T

    # ecorr_cov_matrix returns a bare ndarray already in s^2 (its weights are
    # ``ec.quantity.to(u.s).value ** 2``), so no unit conversion is needed.
    ecorr = m.components["EcorrNoise"]
    pint_cov = np.asarray(ecorr.ecorr_cov_matrix(toas))

    assert U.shape[1] > 0, "multi-backend ECORR produced an empty basis"
    np.testing.assert_allclose(
        jax_cov, pint_cov, rtol=1e-10, atol=1e-25,
        err_msg="multi-backend ECORR covariance diverges from PINT",
    )
