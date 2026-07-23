"""optimal-statistic port: the core ``os`` contraction.

Optimal statistic: Anholm et al. (2009, arXiv:0809.0701); time-domain estimator
Chamberlin et al. (2015, arXiv:1410.8256); noise-marginalized OS (NMOS)
Vigeland et al. (2018, arXiv:1805.12188).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import PTAConfig, per_pulsar_gw_blocks
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
from jaxpint.frequentist.optimal import OptimalStatistic, optimal_statistic

from tests.helpers import make_simple_pulsar


jax.config.update("jax_enable_x64", True)

T_SPAN = 365.25 * 86400.0
LOG10_A = -14.0
GAMMA = 4.33


@functools.lru_cache(maxsize=None)
def _blocks(n_pulsars=4, n_components=5):
    """Per-pulsar GW blocks for an ``n_pulsars`` HD-correlated array.
    """
    rng = np.random.default_rng(3)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    tds, tms, nms, pps = [], [], [], []
    for i in range(n_pulsars):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=20 + 4 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=3 + i
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
    return per_pulsar_gw_blocks(gp, tuple(pps), config)


def _os_reference(blocks, log10_A):
    """From-scratch NumPy pulsar-pair loop — the independent reference.

    Implements the same ρ/σ/Â²/SNR formula with an explicit ``for a < b`` loop
    and ``np.trace(D_a @ D_b)`` (no triu indexing, no swapaxes).
    """
    kv = np.asarray(blocks.basis_proj_residual)
    km = np.asarray(blocks.basis_overlap)
    orf = np.asarray(blocks.orf_matrix)
    sqrt_phi = np.sqrt(np.asarray(blocks.psd))
    n_psr = kv.shape[0]
    gwnorm = 10.0 ** (2.0 * log10_A)

    num = 0.0
    den = 0.0
    for a in range(n_psr):
        for b in range(a + 1, n_psr):
            uk_a = sqrt_phi * kv[a]
            uk_b = sqrt_phi * kv[b]
            ts = uk_a @ uk_b
            D_a = sqrt_phi[:, None] * km[a] * sqrt_phi[None, :]
            D_b = sqrt_phi[:, None] * km[b] * sqrt_phi[None, :]
            bs = np.trace(D_a @ D_b)
            rho = gwnorm * ts / bs
            sigma = gwnorm / np.sqrt(bs)
            num += rho * orf[a, b] / sigma**2
            den += orf[a, b] ** 2 / sigma**2
    a_squared = num / den
    a_squared_sigma = 1.0 / np.sqrt(den)
    return a_squared, a_squared_sigma, a_squared / a_squared_sigma


def test_matches_independent_pair_loop():
    """Vectorized contraction == from-scratch NumPy pair loop."""
    blocks = _blocks()
    os = optimal_statistic(blocks, LOG10_A)
    a2_ref, a2s_ref, snr_ref = _os_reference(blocks, LOG10_A)

    assert isinstance(os, OptimalStatistic)
    np.testing.assert_allclose(float(os.a_squared), a2_ref, rtol=1e-10)
    np.testing.assert_allclose(float(os.a_squared_sigma), a2s_ref, rtol=1e-10)
    np.testing.assert_allclose(float(os.snr), snr_ref, rtol=1e-10)


def test_permutation_invariant():
    """Â²/σ/SNR are invariant to relabelling the pulsars.

    Permute the per-pulsar blocks (rows of kv/km) and the ORF matrix (rows and
    columns) by the same permutation; the unordered-pair sum must be unchanged.
    """
    blocks = _blocks(n_pulsars=4)
    ref = optimal_statistic(blocks, LOG10_A)

    perm = np.array([2, 0, 3, 1])
    permuted = blocks._replace(
        basis_proj_residual=blocks.basis_proj_residual[perm],
        basis_overlap=blocks.basis_overlap[perm],
        orf_matrix=blocks.orf_matrix[np.ix_(perm, perm)],
    )
    out = optimal_statistic(permuted, LOG10_A)

    np.testing.assert_allclose(float(out.a_squared), float(ref.a_squared), rtol=1e-12)
    np.testing.assert_allclose(
        float(out.a_squared_sigma), float(ref.a_squared_sigma), rtol=1e-12
    )
    np.testing.assert_allclose(float(out.snr), float(ref.snr), rtol=1e-12)


def test_snr_scale_invariant():
    """SNR is invariant to the gwnorm convention; Â² and σ_Â² scale by it.

    Changing ``log10_A`` by Δ (with the same blocks) rescales ``gwnorm`` by
    ``10^(2Δ)``; Â² and σ_Â² both scale by that factor, so their ratio (SNR)
    is unchanged.  A bug applying ``gwnorm`` to only ρ or only σ would break
    this.
    """
    blocks = _blocks()
    base = optimal_statistic(blocks, LOG10_A)
    shifted = optimal_statistic(blocks, LOG10_A + 0.5)
    factor = 10.0 ** (2.0 * 0.5)

    np.testing.assert_allclose(float(shifted.snr), float(base.snr), rtol=1e-10)
    np.testing.assert_allclose(
        float(shifted.a_squared), float(base.a_squared) * factor, rtol=1e-10
    )
    np.testing.assert_allclose(
        float(shifted.a_squared_sigma),
        float(base.a_squared_sigma) * factor,
        rtol=1e-10,
    )


def test_jit_and_vmap():
    """jit-traceable, and vmappable over log10_A (NMOS readiness)."""
    blocks = _blocks()
    ref = optimal_statistic(blocks, LOG10_A)

    jitted = jax.jit(optimal_statistic)(blocks, LOG10_A)
    np.testing.assert_allclose(float(jitted.snr), float(ref.snr), rtol=1e-9)

    log10_As = jnp.array([-15.0, -14.5, -14.0])
    batched = jax.vmap(lambda a: optimal_statistic(blocks, a))(log10_As)
    assert batched.snr.shape == (3,)
    # SNR is amplitude-invariant -> the whole batch equals the scalar SNR.
    np.testing.assert_allclose(
        np.asarray(batched.snr), float(ref.snr), rtol=1e-9
    )
