"""Sherman–Morrison kernel ECORR: math pins and basis-form equivalence.

The kernel and basis forms encode the *same* covariance, so every
likelihood quantity must agree between a model with ECORR as basis
columns (`NoiseModel.correlated`) and the same model with ECORR as a
whitening kernel (`NoiseModel.ecorr_kernel`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.likelihood import single_pulsar_logL, precompute_single_pulsar_factor
from jaxpint.noise import EcorrKernelNoise, EcorrNoise, NoiseModel, PLRedNoise
from jaxpint.utils import NO_EPOCH, build_fourier_basis, build_quantization_index
from tests.helpers import make_params, make_simple_pulsar

_DAY = 86400.0


# ---------------------------------------------------------------------------
# Fixtures: one pulsar, ECORR expressed both ways
# ---------------------------------------------------------------------------


def _setup(n=60, with_orphan_toa=True, seed=0):
    """Spindown pulsar + red noise + two-backend ECORR, basis and kernel."""
    toa_data, timing_model, base_nm, _ = make_simple_pulsar(n, f0=200.0, f1=-1e-15)
    t = np.asarray(toa_data.tdb_seconds)

    F, freqs, df = build_fourier_basis(t, 3, t.max() - t.min())
    red = PLRedNoise(
        fourier_basis=jnp.asarray(F),
        freqs=jnp.asarray(freqs),
        freq_bin_widths=jnp.asarray(df),
        tnredamp_name="TNREDAMP",
        tnredgam_name="TNREDGAM",
    )

    masks = {
        "ECORR1": np.arange(n) % 2 == 0,
        "ECORR2": np.arange(n) % 2 == 1,
    }
    if with_orphan_toa:
        # Give the last TOA no epoch: exclude it from both masks so the
        # NO_EPOCH path is always exercised.
        masks["ECORR1"][-1] = False
        masks["ECORR2"][-1] = False
    idx, n_epochs, eslices = build_quantization_index(t, masks, dt=_DAY / 20)
    assert n_epochs > 2
    if with_orphan_toa:
        assert idx[-1] == NO_EPOCH

    ecorr_basis = EcorrNoise(
        ecorr_names=("ECORR1", "ECORR2"),
        epoch_index=idx,
        n_epochs=n_epochs,
        ecorr_epoch_slices=(eslices["ECORR1"], eslices["ECORR2"]),
    )
    ecorr_kernel = EcorrKernelNoise.from_basis(ecorr_basis)

    nm_basis = NoiseModel(
        white_noise=base_nm.white_noise, correlated=(red, ecorr_basis)
    )
    nm_kernel = NoiseModel(
        white_noise=base_nm.white_noise,
        correlated=(red,),
        ecorr_kernel=ecorr_kernel,
    )

    params = make_params(
        names=(
            "F0", "F1", "PEPOCH", "EFAC1", "EQUAD1",
            "TNREDAMP", "TNREDGAM", "ECORR1", "ECORR2",
        ),
        values=(200.0, -1e-15, 0.0, 1.1, 3e-7, -13.5, 3.2, 5e-7, 3e-7),
        frozen_mask=(False, False, True, True, True, False, False, False, False),
        epoch_int_values={"PEPOCH": 59000.0},
    )
    return toa_data, timing_model, nm_basis, nm_kernel, params, ecorr_kernel


# ---------------------------------------------------------------------------
# Whitener math against a dense reference
# ---------------------------------------------------------------------------


class TestWhitenerMath:
    def _dense_N(self, Ndiag, jitter2, idx, n_epochs):
        N = np.diag(np.asarray(Ndiag))
        for e in range(n_epochs):
            u = (np.asarray(idx) == e).astype(float)
            N += float(jitter2[e]) * np.outer(u, u)
        return N

    def test_whitener_and_logdet_vs_dense(self):
        rng = np.random.default_rng(7)
        n, n_epochs = 25, 4
        idx = rng.integers(0, n_epochs, n).astype(np.int32)
        idx[[3, 17]] = NO_EPOCH
        Ndiag = jnp.asarray(rng.uniform(0.5, 2.0, n))

        kern = EcorrKernelNoise(
            ecorr_names=("ECORR1",),
            epoch_index=idx,
            n_epochs=n_epochs,
            ecorr_epoch_slices=((0, n_epochs),),
        )
        params = make_params(("ECORR1",), (1.0,), units=("s",))
        # One param -> uniform j2 here; heterogeneous j2 gets its dense pin
        # in test_fit_cinv_vs_dense (two-parameter setup).
        ops = kern.ops(Ndiag, params)
        N = self._dense_N(Ndiag, np.full(n_epochs, 1.0), idx, n_epochs)
        W = np.asarray(ops.whiten(jnp.eye(n)))
        # W N W^T = I also implies N^{-1} = W^T W; no separate assertion.
        npt.assert_allclose(W @ N @ W.T, np.eye(n), atol=1e-11)
        sign, logdet = np.linalg.slogdet(N)
        assert sign > 0
        npt.assert_allclose(float(ops.extra_logdet), logdet, rtol=1e-12)


# ---------------------------------------------------------------------------
# Basis <-> kernel equivalence through the likelihood stack
# ---------------------------------------------------------------------------


class TestBasisKernelEquivalence:
    def test_single_pulsar_logL(self):
        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        for use_qr in (False, True):
            lb = single_pulsar_logL(toa_data, tm, nm_b, params, use_qr=use_qr)
            lk = single_pulsar_logL(toa_data, tm, nm_k, params, use_qr=use_qr)
            npt.assert_allclose(float(lk), float(lb), rtol=1e-10)

    def test_gradients_match(self):
        toa_data, tm, nm_b, nm_k, params, _ = _setup()

        def logl(nm):
            def f(v):
                return single_pulsar_logL(
                    toa_data, tm, nm, params.with_free_values(v)
                )
            return jax.grad(f)(params.free_values())

        gb, gk = logl(nm_b), logl(nm_k)
        assert bool(jnp.all(jnp.isfinite(gk)))
        npt.assert_allclose(np.asarray(gk), np.asarray(gb), rtol=1e-7, atol=1e-12)

    def test_jit_and_equivalence_under_jit(self):
        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        f = jax.jit(lambda nm, v: single_pulsar_logL(
            toa_data, tm, nm, params.with_free_values(v)
        ))
        npt.assert_allclose(
            float(f(nm_k, params.free_values())),
            float(f(nm_b, params.free_values())),
            rtol=1e-10,
        )

    def test_pta_inner_tier(self):
        from jaxpint.pta.likelihood import _per_pulsar_intermediates

        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        rng = np.random.default_rng(1)
        F_corr = jnp.asarray(rng.standard_normal((int(toa_data.n_toas), 6)))
        out_b = _per_pulsar_intermediates(toa_data, tm, nm_b, params, F_corr)
        out_k = _per_pulsar_intermediates(toa_data, tm, nm_k, params, F_corr)
        for b, k, tag in zip(out_b, out_k, ("rCr", "logdet", "proj", "overlap")):
            npt.assert_allclose(
                np.asarray(k), np.asarray(b), rtol=1e-8, atol=1e-10,
                err_msg=f"inner-tier {tag} mismatch",
            )

    def test_conditional_red_block_marginal(self):
        from jaxpint.pta.conditional import (
            conditional_covariance,
            conditional_single_pulsar,
        )

        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        cond_b = conditional_single_pulsar(toa_data, tm, nm_b, params)
        cond_k = conditional_single_pulsar(toa_data, tm, nm_k, params)
        n_red = 6  # 2 * 3 frequencies; red block is first in `correlated`
        npt.assert_allclose(
            np.asarray(cond_k.mean), np.asarray(cond_b.mean)[:n_red], rtol=1e-8
        )
        cov_b = np.asarray(conditional_covariance(cond_b))
        cov_k = np.asarray(conditional_covariance(cond_k))
        npt.assert_allclose(
            cov_k, cov_b[:n_red, :n_red], rtol=1e-7, atol=1e-20
        )

    def test_generate_bitwise_identical_to_basis(self):
        toa_data, _, nm_b, _, params, kern = _setup()
        basis_ecorr = nm_b.correlated[1]
        key = jax.random.PRNGKey(11)
        d_basis = basis_ecorr.generate(toa_data, params, key)
        d_kernel = kern.generate(toa_data, params, key)
        npt.assert_array_equal(np.asarray(d_kernel), np.asarray(d_basis))
        # Within-epoch draws are constant; NO_EPOCH TOA draws zero.
        idx = np.asarray(kern.epoch_index)
        assert float(np.asarray(d_kernel)[idx == NO_EPOCH].sum()) == 0.0


# ---------------------------------------------------------------------------
# Guards on unsupported paths
# ---------------------------------------------------------------------------


class TestKernelGuards:
    def test_misplaced_in_correlated_raises(self):
        # The actual user mistake: kernel placed in `correlated`, error
        # surfacing through the likelihood (via NoiseModel.covariance).
        toa_data, tm, _, nm_k, params, kern = _setup()
        nm_bad = NoiseModel(white_noise=nm_k.white_noise, correlated=(kern,))
        with pytest.raises(TypeError, match="ecorr_kernel"):
            single_pulsar_logL(toa_data, tm, nm_bad, params)

    # (The former precompute-factor and whiten_residuals guards became
    # features in K2 — their positive tests live in TestK2FitterIntegration.)


class TestKernelHierarchy:
    """The kernel is deliberately NOT a basis GP (composition over LSP-violating
    inheritance): shared logic lives in epoch_jitter2, not a shared parent."""

    def test_not_a_basis_gp(self):
        from jaxpint.noise._basis_gp import _BasisGPNoise

        assert not issubclass(EcorrKernelNoise, EcorrNoise)
        assert not issubclass(EcorrKernelNoise, _BasisGPNoise)

    def test_jitter_assembly_shared(self):
        # Tripwire, not an independent check: both forms currently call the
        # same free function (epoch_jitter2), so this can only fail if the
        # implementations are ever forked.
        toa_data, _, nm_b, _, params, kern = _setup()
        basis_ecorr = nm_b.correlated[1]
        npt.assert_array_equal(
            np.asarray(kern.ecorr_weights(params)),
            np.asarray(basis_ecorr.ecorr_weights(params)),
        )

    def test_ecorr_average_accepts_kernel_slot(self):
        from jaxpint.fitters.diagnostics import ecorr_average

        toa_data, _, nm_b, nm_k, params, _ = _setup()
        r = jnp.asarray(
            np.random.default_rng(3).standard_normal(int(toa_data.n_toas)) * 1e-6
        )
        avg_b = ecorr_average(r, toa_data, params, nm_b)
        avg_k = ecorr_average(r, toa_data, params, nm_k)
        npt.assert_allclose(
            np.asarray(avg_k.mjds), np.asarray(avg_b.mjds), rtol=1e-12
        )
        npt.assert_allclose(
            np.asarray(avg_k.time_resids), np.asarray(avg_b.time_resids),
            rtol=1e-12,
        )
        npt.assert_allclose(
            np.asarray(avg_k.errors), np.asarray(avg_b.errors), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# fitter / factor / whitening integration
# ---------------------------------------------------------------------------


class TestFitterIntegration:
    def test_factor_path_matches_direct(self):
        from jaxpint.likelihood import single_pulsar_logL_with_factor

        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        direct_k = float(single_pulsar_logL(toa_data, tm, nm_k, params))
        factor = precompute_single_pulsar_factor(toa_data, nm_k, params)
        via_factor = float(
            single_pulsar_logL_with_factor(toa_data, tm, factor, params)
        )
        npt.assert_allclose(via_factor, direct_k, rtol=1e-12)
        # And both agree with the basis form.
        direct_b = float(single_pulsar_logL(toa_data, tm, nm_b, params))
        npt.assert_allclose(via_factor, direct_b, rtol=1e-10)

    def test_gls_fit_equivalence(self):
        from jaxpint.fitters import GLSFitter

        toa_data, tm, nm_b, nm_k, params, _ = _setup()
        # GLS fits timing parameters; noise parameters stay frozen (their
        # residual Jacobian is exactly zero, so leaving them free yields
        # numerically undefined updates in any GLS path — pre-existing).
        fit_params = make_params(
            names=params.names,
            values=tuple(float(v) for v in np.asarray(params.values)),
            frozen_mask=tuple(n not in ("F0", "F1") for n in params.names),
            epoch_int_values={"PEPOCH": 59000.0},
        )
        res_b = GLSFitter(tm, toa_data, fit_params, noise_model=nm_b).fit_toas(maxiter=3)
        res_k = GLSFitter(tm, toa_data, fit_params, noise_model=nm_k).fit_toas(maxiter=3)
        npt.assert_allclose(
            np.asarray(res_k.params.free_values()),
            np.asarray(res_b.params.free_values()),
            rtol=1e-8,
        )
        npt.assert_allclose(
            np.asarray(res_k.covariance_matrix),
            np.asarray(res_b.covariance_matrix),
            rtol=1e-6,
            atol=1e-30,
        )
        npt.assert_allclose(float(res_k.chi2), float(res_b.chi2), rtol=1e-8)
        npt.assert_allclose(
            np.asarray(res_k.residuals), np.asarray(res_b.residuals),
            rtol=1e-6, atol=1e-12,
        )

    def test_fit_cinv_vs_dense(self):
        from jaxpint.fitters import GLSFitter

        toa_data, tm, nm_b, nm_k, params, kern = _setup(n=40)
        fitter = GLSFitter(tm, toa_data, params, noise_model=nm_k)
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.standard_normal(int(toa_data.n_toas)))

        # Dense reference: C = N_full + U_red Phi U_red^T.
        Ndiag = np.asarray(nm_k.scaled_sigma(toa_data, params)) ** 2
        idx = np.asarray(kern.epoch_index)
        j2 = np.asarray(kern.ecorr_weights(params))
        C = np.diag(Ndiag)
        for e in range(kern.n_epochs):
            u = (idx == e).astype(float)
            C += j2[e] * np.outer(u, u)
        _, U_red, Phi_red = nm_k.correlated[0].covariance(toa_data, params)
        C += np.asarray(U_red) @ np.diag(np.asarray(Phi_red)) @ np.asarray(U_red).T

        npt.assert_allclose(
            np.asarray(fitter._fit_cinv(params, x)),
            np.linalg.solve(C, np.asarray(x)),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_whiten_residuals_kernel_only(self):
        from jaxpint.fitters.diagnostics import whiten_residuals

        toa_data, tm, _, nm_k, params, kern = _setup(n=40)
        nm_kernel_only = NoiseModel(
            white_noise=nm_k.white_noise, correlated=(), ecorr_kernel=kern
        )
        rng = np.random.default_rng(9)
        r = jnp.asarray(rng.standard_normal(int(toa_data.n_toas)) * 1e-6)
        w = whiten_residuals(r, toa_data, params, nm_kernel_only)
        # No GP block: whitened residuals are W r, so sum(w^2) = r^T N_full^-1 r.
        Ndiag = np.asarray(nm_kernel_only.scaled_sigma(toa_data, params)) ** 2
        idx = np.asarray(kern.epoch_index)
        j2 = np.asarray(kern.ecorr_weights(params))
        N = np.diag(Ndiag)
        for e in range(kern.n_epochs):
            u = (idx == e).astype(float)
            N += j2[e] * np.outer(u, u)
        expected = np.asarray(r) @ np.linalg.solve(N, np.asarray(r))
        npt.assert_allclose(float(jnp.sum(w**2)), expected, rtol=1e-10)

    def test_whiten_residuals_full_model_finite(self):
        from jaxpint.fitters.diagnostics import whiten_residuals

        # Smoke only, deliberately: with a GP block present the kernel path
        # whitens *more* than the basis form (epoch decorrelation on top of
        # conditional-mean subtraction), so there is no basis-form equality
        # to pin, and a dense reference would just restate the
        # implementation's own definition.
        toa_data, _, _, nm_k, params, _ = _setup()
        r = jnp.asarray(np.random.default_rng(2).standard_normal(
            int(toa_data.n_toas)) * 1e-6)
        w = whiten_residuals(r, toa_data, params, nm_k)
        assert bool(jnp.all(jnp.isfinite(w)))
