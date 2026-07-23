"""Phase 0 of the optimal-statistic port: the per-pulsar GW block producer.

Guards that :func:`per_pulsar_gw_blocks` reproduces exactly the same per-pulsar
``(kv, km)`` blocks the correlated ``pta_logL`` inner tier computes via
:func:`_per_pulsar_intermediates` These are
the ingredients the optimal statistic (:mod:`jaxpint.frequentist.optimal`)
will contract into pairwise cross-correlations.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.fitters import compute_time_residuals
from jaxpint.noise import NoiseModel
from jaxpint.noise.white import ScaleToaError
from jaxpint.noise.red_noise import PLRedNoise
from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import (
    GWBlocks,
    PTAConfig,
    per_pulsar_gw_blocks,
    _per_pulsar_intermediates,
)
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
from jaxpint._psd import powerlaw_psd
from jaxpint.pta.signals.orf import hd_orf

from tests.helpers import make_simple_pulsar, make_params


jax.config.update("jax_enable_x64", True)

T_SPAN = 365.25 * 86400.0
LOG10_A = -14.0
GAMMA = 4.33


@functools.lru_cache(maxsize=None)  # immutable returns; shared with test_optimal_nmos
def _two_pulsar_gwb_config(n_components=5):
    """A 2-pulsar PTA with a single HD-correlated GWB injector."""
    rng = np.random.default_rng(7)
    positions = rng.normal(size=(2, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    tds, tms, nms, pps = [], [], [], []
    for i in range(2):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=20 + 5 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=7 + i
        )
        tds.append(td)
        tms.append(tm)
        nms.append(nm)
        pps.append(pp)

    inj = HDCorrelatedGWBInjector(
        pulsar_positions=positions,
        n_components=n_components,
        T_span=T_SPAN,
        initial_values={"log10_A": LOG10_A, "gamma": GAMMA},
    )
    gp = inj.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=tuple(tms),
        noise_models=tuple(nms),
        signal_injectors=(),
        correlated_injectors=(inj,),
    )
    return gp, tuple(pps), config, inj


@functools.lru_cache(maxsize=None)  # immutable returns; shared with test_optimal_nmos
def _two_pulsar_red_noise_config(n_components=5, n_red=3):
    """2-pulsar HD-GWB config WITH per-pulsar intrinsic red noise.

    The red noise gives each pulsar a non-trivial low-rank covariance
    ``C_p = diag(N) + U diag(Φ) Uᵀ`` (``U`` has columns), so a dense
    brute-force reference genuinely cross-checks the Woodbury solve inside the
    block producer    """
    rng = np.random.default_rng(11)
    positions = rng.normal(size=(2, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    tds, tms, nms, pps = [], [], [], []
    for i in range(2):
        td, tm, _, pp = make_simple_pulsar(
            n_toas=20 + 5 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=11 + i
        )
        toas_s = td.tdb_int.astype(jnp.float64) * 86400.0 + td.tdb_frac * 86400.0
        freqs_rn = jnp.arange(1, n_red + 1) / T_SPAN
        phase = 2.0 * jnp.pi * toas_s[:, None] * freqs_rn[None, :]
        F_rn = jnp.column_stack([jnp.sin(phase), jnp.cos(phase)])
        df_rn = jnp.full(n_red, 1.0 / T_SPAN)
        nm = NoiseModel(
            white_noise=ScaleToaError(
                efac_names=("EFAC1",), equad_names=("EQUAD1",)
            ),
            correlated=(
                PLRedNoise(
                    fourier_basis=F_rn,
                    freqs=freqs_rn,
                    freq_bin_widths=df_rn,
                    tnredamp_name="TNREDAMP",
                    tnredgam_name="TNREDGAM",
                ),
            ),
        )
        pp = make_params(
            names=pp.names + ("TNREDAMP", "TNREDGAM", "TNREDC"),
            values=list(np.array(pp.values)) + [-13.0, 3.0, n_red],
            frozen_mask=pp.frozen_mask + (True, True, True),
            epoch_int_values=pp.epoch_int_values,
        )
        tds.append(td)
        tms.append(tm)
        nms.append(nm)
        pps.append(pp)

    inj = HDCorrelatedGWBInjector(
        pulsar_positions=positions,
        n_components=n_components,
        T_span=T_SPAN,
        initial_values={"log10_A": LOG10_A, "gamma": GAMMA},
    )
    gp = inj.register_params(GlobalParams.empty())
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=tuple(tms),
        noise_models=tuple(nms),
        signal_injectors=(),
        correlated_injectors=(inj,),
    )
    return gp, tuple(pps), config, inj, positions


def test_blocks_match_per_pulsar_intermediates():
    """kv/km per pulsar match a direct _per_pulsar_intermediates call, exactly.
    Catches drift between per_pulsar_gw_blocks and _per_pulsar_intermediates"""
    gp, pps, config, inj = _two_pulsar_gwb_config()
    blocks = per_pulsar_gw_blocks(gp, pps, config)

    assert isinstance(blocks, GWBlocks)
    for p in range(config.n_pulsars):
        F_p = inj.get_fourier_basis(config.toa_data_list[p])
        _, _, kv_p, km_p = _per_pulsar_intermediates(
            config.toa_data_list[p],
            config.timing_models[p],
            config.noise_models[p],
            pps[p],
            F_p,
        )
        # Same code path -> bitwise-identical, no tolerance. kv is passed
        # through verbatim; km is symmetrized by the producer, so symmetrize
        # the reference the same way (identical op on identical values).
        np.testing.assert_array_equal(
            np.asarray(blocks.basis_proj_residual[p]), np.asarray(kv_p)
        )
        km_sym = 0.5 * (km_p + km_p.T)
        np.testing.assert_array_equal(
            np.asarray(blocks.basis_overlap[p]), np.asarray(km_sym)
        )


def test_psd_and_orf_independently_recomputed():
    """Φ and Γ match INDEPENDENT recomputations, not the injector's own methods.

    Avoids the tautology of comparing ``blocks.psd`` to ``inj.get_psd``
    """
    n_components = 5
    gp, pps, config, _, positions = _two_pulsar_red_noise_config(
        n_components=n_components
    )
    blocks = per_pulsar_gw_blocks(gp, pps, config)

    # independent power-law assembly (freq grid, df, sin/cos repeat).
    freqs = np.arange(1, n_components + 1) / T_SPAN
    df = 1.0 / T_SPAN
    psd = np.asarray(powerlaw_psd(jnp.asarray(freqs), LOG10_A, GAMMA)) * df
    phi_ref = np.repeat(psd, 2)  # sin and cos of a frequency share its PSD
    np.testing.assert_allclose(np.asarray(blocks.psd), phi_ref, rtol=1e-12)

    # Hellings-Downs evaluated directly on the positions.
    n_psr = config.n_pulsars
    gamma_ref = np.array(
        [
            [float(hd_orf(positions[a], positions[b])) for b in range(n_psr)]
            for a in range(n_psr)
        ]
    )
    np.testing.assert_allclose(
        np.asarray(blocks.orf_matrix), gamma_ref, rtol=1e-12
    )


def test_block_shapes():
    n_components = 4
    gp, pps, config, _ = _two_pulsar_gwb_config(n_components=n_components)
    blocks = per_pulsar_gw_blocks(gp, pps, config)
    n_psr = config.n_pulsars
    n_basis = 2 * n_components  # sin + cos per frequency
    assert blocks.basis_proj_residual.shape == (n_psr, n_basis)
    assert blocks.basis_overlap.shape == (n_psr, n_basis, n_basis)
    assert blocks.psd.shape == (n_basis,)
    assert blocks.orf_matrix.shape == (n_psr, n_psr)


def test_km_symmetric_psd_positive():
    """km_p is *exactly* symmetric and numerically PSD;

    ``km`` is still severely ill-conditioned (the timing/noise
    ``C`` spans many decades), so its smallest eigenvalues sit at the noise
    floor and can go slightly negative
    We therefore assert PSD *relative to the top eigenvalue*.
    """
    gp, pps, config, _ = _two_pulsar_gwb_config()
    blocks = per_pulsar_gw_blocks(gp, pps, config)
    for p in range(config.n_pulsars):
        km = np.asarray(blocks.basis_overlap[p])
        np.testing.assert_array_equal(km, km.T)  # exactly symmetric, no tol
        eig = np.linalg.eigvalsh(km)
        assert eig.min() > -1e-9 * eig.max()  # numerically PSD
    assert bool(jnp.all(blocks.psd > 0.0))


def test_requires_single_correlated_injector():
    """Zero (or multiple) correlated injectors is a hard error."""
    gp, pps, config, _ = _two_pulsar_gwb_config()
    no_corr = PTAConfig(
        toa_data_list=config.toa_data_list,
        timing_models=config.timing_models,
        noise_models=config.noise_models,
        signal_injectors=(),
        correlated_injectors=(),
    )
    with pytest.raises(ValueError, match="exactly one correlated injector"):
        per_pulsar_gw_blocks(gp, pps, no_corr)


def test_jit_traceable_over_params():
    """Traceable under jit over (global, pulsar) params — NMOS readiness."""
    gp, pps, config, _ = _two_pulsar_gwb_config()

    @jax.jit
    def f(gp_, pps_):
        return per_pulsar_gw_blocks(gp_, pps_, config)

    jitted = f(gp, pps)
    ref = per_pulsar_gw_blocks(gp, pps, config)
    # jit vs eager agree only to XLA float-reassociation precision, not bitwise.
    np.testing.assert_allclose(
        np.asarray(jitted.basis_overlap), np.asarray(ref.basis_overlap),
        rtol=1e-9, atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(jitted.basis_proj_residual),
        np.asarray(ref.basis_proj_residual), rtol=1e-9, atol=0.0,
    )


def test_blocks_match_dense_reference():
    """Independent value check: kv/km match a DENSE brute-force solve.

    The producer computes ``kv = FᵀC⁻¹r`` and ``km = FᵀC⁻¹F`` via the Woodbury
    identity (``woodbury_solve``).  Here we form the full per-pulsar covariance
    ``C_p = diag(N) + U diag(Φ) Uᵀ`` densely and solve with ``np.linalg.solve``
    """
    gp, pps, config, inj, _ = _two_pulsar_red_noise_config()
    blocks = per_pulsar_gw_blocks(gp, pps, config)

    for p in range(config.n_pulsars):
        td = config.toa_data_list[p]
        r = np.asarray(compute_time_residuals(config.timing_models[p], td, pps[p]))
        Ndiag, U, Phi = config.noise_models[p].covariance(td, pps[p])
        Ndiag, U, Phi = np.asarray(Ndiag), np.asarray(U), np.asarray(Phi)
        assert U.shape[1] > 0, "fixture must exercise the Woodbury low-rank path"
        C = np.diag(Ndiag) + U @ np.diag(Phi) @ U.T  # dense per-pulsar kernel
        F = np.asarray(inj.get_fourier_basis(td))

        kv_dense = F.T @ np.linalg.solve(C, r)
        km_dense = F.T @ np.linalg.solve(C, F)
        km_dense = 0.5 * (km_dense + km_dense.T)

        # Woodbury vs a dense solve of a well-conditioned C agree to ~1e-15;
        # 1e-10 (norm-wise, relative Frobenius) leaves margin for BLAS/platform
        # variation while still catching any real value bug.
        kv_actual = np.asarray(blocks.basis_proj_residual[p])
        kv_rel = np.linalg.norm(kv_actual - kv_dense) / np.linalg.norm(kv_dense)
        assert kv_rel < 1e-10, f"kv dense-vs-Woodbury relative error {kv_rel:.2e}"

        km_actual = np.asarray(blocks.basis_overlap[p])
        km_rel = np.linalg.norm(km_actual - km_dense) / np.linalg.norm(km_dense)
        assert km_rel < 1e-10, f"km dense-vs-Woodbury relative error {km_rel:.2e}"
