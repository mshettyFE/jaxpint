r"""Spectral models (power law / broken power law / free spectrum).

The central claim under test: swapping the PSD model does **not** disturb
the Woodbury machinery, because every spectrum here fills the same diagonal
``\Phi`` — a free spectrum is more hyperparameters, not more covariance
structure.  The equivalence tests therefore run the *full* likelihood
pipeline (CURN fast path and correlated two-tier path) with a free spectrum
pinned to a power law, and demand equality with the power-law result to
float precision.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.noise import FreeSpectrumNoise, NoiseModel, PLRedNoise
from jaxpint.pta.likelihood import PTAConfig, pta_logL
from jaxpint.pta.signals import (
    BrokenPowerLawSpectrum,
    CURNInjector,
    FreeSpectrum,
    HDCorrelatedGWBInjector,
    PowerLawSpectrum,
    powerlaw_psd,
)
from jaxpint.types import GlobalParams

from tests.helpers import make_fourier_basis, make_simple_pulsar

jax.config.update("jax_enable_x64", True)

T_SPAN = 1e8
N_COMP = 5
LOG10_A = -14.0
GAMMA = 4.33

FREQS = jnp.arange(1, N_COMP + 1) / T_SPAN
DF = 1.0 / T_SPAN

# Frozen golden literals — data, not code.  The tests read these; nothing
# recomputes them at test time, so a convention drift edited into the
# library *and* the test helpers in tandem still fails here.
# Helps pin down the normalization of the power spectrum
# (see generate_golden_values).
GOLDEN = {
    # S(f)·Δf at A=1e-14, γ=4.33, f=Δf=1e-8 Hz (bin 1 of the fixture).
    "powerlaw_w0": 3.914810390581516e-14,
}


def _matched_log10_rho(log10_A=LOG10_A, gamma=GAMMA):
    """Per-bin log10_rho that reproduces the power law: \rho_k^2 = S(f_k)·Df."""
    psd = powerlaw_psd(FREQS, log10_A, gamma) * DF
    return 0.5 * jnp.log10(psd)


# ---------------------------------------------------------------------------
# Spectrum-level unit tests
# ---------------------------------------------------------------------------


def _value_of(defaults):
    return lambda name: jnp.asarray(defaults[name])


def _powerlaw_weights_reference(freqs, log10_A, gamma, df):
    """Independent transcription of Arzoumanian et al. (2016) Eq. 1.
    """
    fyr = 1.0 / (365.25 * 86400.0)
    freqs = np.asarray(freqs, dtype=np.float64)
    psd = (
        (10.0**log10_A) ** 2
        / (12.0 * np.pi**2)
        * fyr ** (gamma - 3.0)
        * freqs ** (-gamma)
    )
    return np.repeat(psd * df, 2)


def generate_golden_values() -> dict[str, float]:
    """Recompute the values frozen in ``GOLDEN``, printed paste-ready.

    Deliberately NOT called by any test — the tests read the frozen
    ``GOLDEN`` literals, so a convention drift edited into the library
    *and* into :func:`_powerlaw_weights_reference` in tandem still trips
    the golden assertion.  When the spectral convention is *meant* to
    change, run ``python tests/test_spectral_models.py``, paste the
    printed dict over ``GOLDEN``, and say so in the commit.
    """
    goldens = {
        # S(f)·Δf at A=1e-14, γ=4.33, f=Δf=1e-8 Hz (bin 1 of the fixture).
        "powerlaw_w0": float(
            _powerlaw_weights_reference(np.array([1e-8]), -14.0, 4.33, 1e-8)[0]
        ),
    }
    print("GOLDEN = {")
    for name, value in goldens.items():
        print(f'    "{name}": {value:.15e},')
    print("}")
    return goldens


def test_powerlaw_spectrum_matches_independent_reference():
    spec = PowerLawSpectrum(log10_A=LOG10_A, gamma=GAMMA)
    w = np.asarray(spec.psd_weights(FREQS, DF, _value_of(spec.param_defaults())))
    ref = _powerlaw_weights_reference(FREQS, LOG10_A, GAMMA, DF)
    npt.assert_allclose(w, ref, rtol=1e-14)
    # Frozen literal (see GOLDEN / generate_golden_values): pins the absolute
    # normalization — a wrong f_yr, 12π², or df convention shifts it.
    npt.assert_allclose(w[0], GOLDEN["powerlaw_w0"], rtol=1e-12)


def test_broken_powerlaw_reduces_to_powerlaw():
    """With the bend far above the sampled band, the bend factor is 1."""
    spec = BrokenPowerLawSpectrum(log10_A=LOG10_A, gamma=GAMMA, log10_fb=-2.0)
    w = spec.psd_weights(FREQS, DF, _value_of(spec.param_defaults()))
    ref = _powerlaw_weights_reference(FREQS, LOG10_A, GAMMA, DF)
    npt.assert_allclose(np.asarray(w), ref, rtol=1e-12)


def test_broken_powerlaw_flattens_above_bend():
    """Well above f_b the bend factor ≈ (f/f_b)^γ, i.e. the PSD flattens."""
    log10_fb = float(jnp.log10(FREQS[0])) - 1.0  # bend below the whole band
    spec = BrokenPowerLawSpectrum(log10_A=LOG10_A, gamma=GAMMA, log10_fb=log10_fb)
    w = np.asarray(spec.psd_weights(FREQS, DF, _value_of(spec.param_defaults())))[0::2]
    pl = _powerlaw_weights_reference(FREQS, LOG10_A, GAMMA, DF)[0::2]
    ratio = w / pl
    expected = np.asarray((FREQS / 10.0**log10_fb) ** GAMMA)
    npt.assert_allclose(ratio, expected, rtol=1e-6)


def test_free_spectrum_matches_matched_powerlaw():
    rho = _matched_log10_rho()
    spec = FreeSpectrum(N_COMP, log10_rho=[float(r) for r in rho])
    assert spec.param_names == tuple(f"log10_rho_{k}" for k in range(N_COMP))
    w = spec.psd_weights(FREQS, DF, _value_of(spec.param_defaults()))
    ref = jnp.repeat(powerlaw_psd(FREQS, LOG10_A, GAMMA) * DF, 2)
    npt.assert_allclose(np.asarray(w), np.asarray(ref), rtol=1e-14)


def test_free_spectrum_validation():
    with pytest.raises(ValueError, match="expected 5"):
        FreeSpectrum(5, log10_rho=[-8.0, -8.0])
    with pytest.raises(ValueError, match="n_components"):
        CURNInjector(n_components=4, T_span=T_SPAN, spectrum=FreeSpectrum(5))


# ---------------------------------------------------------------------------
# Full-pipeline equivalence: the Woodbury path is spectrum-agnostic
# ---------------------------------------------------------------------------


def _two_pulsar_setup():
    tds, tms, nms, pps = [], [], [], []
    for i in range(2):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=25 + 5 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=41 + i
        )
        tds.append(td)
        tms.append(tm)
        nms.append(nm)
        pps.append(pp)
    return tuple(tds), tuple(tms), tuple(nms), tuple(pps)


def _config_and_params(injector, correlated: bool):
    tds, tms, nms, pps = _two_pulsar_setup()
    config = PTAConfig(
        toa_data_list=tds,
        timing_models=tms,
        noise_models=nms,
        signal_injectors=() if correlated else (injector,),
        correlated_injectors=(injector,) if correlated else (),
    )
    gp = injector.register_params(GlobalParams.empty())
    return gp, pps, config


def _positions():
    rng = np.random.default_rng(3)
    pos = rng.normal(size=(2, 3))
    return jnp.asarray(pos / np.linalg.norm(pos, axis=1, keepdims=True))


def _injector_pair(correlated: bool):
    """(power-law injector, matched free-spectrum injector) of the same kind."""
    rho = [float(r) for r in _matched_log10_rho()]
    if correlated:
        pos = _positions()
        pl = HDCorrelatedGWBInjector(
            pos, N_COMP, T_SPAN,
            initial_values={"log10_A": LOG10_A, "gamma": GAMMA},
        )
        fs = HDCorrelatedGWBInjector(
            pos, N_COMP, T_SPAN, spectrum=FreeSpectrum(N_COMP, log10_rho=rho)
        )
    else:
        pl = CURNInjector(
            N_COMP, T_SPAN,
            initial_values={"log10_A": LOG10_A, "gamma": GAMMA},
        )
        fs = CURNInjector(
            N_COMP, T_SPAN, spectrum=FreeSpectrum(N_COMP, log10_rho=rho)
        )
    return pl, fs


@pytest.mark.parametrize("correlated", [False, True], ids=["curn", "hd"])
def test_free_spectrum_logL_matches_matched_powerlaw(correlated):
    """pta_logL(free spectrum @ matched ρ) == pta_logL(power law).

    Runs the real solve: the CURN fast path (per-pulsar Woodbury) or the
    correlated two-tier path (outer Cholesky over the joint \Phi).  Equality
    to ~1e-12 shows the free spectrum flows through the identical
    machinery — nothing about the decomposition breaks.
    """
    pl, fs = _injector_pair(correlated)
    gp_pl, pps, config_pl = _config_and_params(pl, correlated)
    gp_fs, _, config_fs = _config_and_params(fs, correlated)

    logL_pl = float(pta_logL(gp_pl, pps, config_pl))
    logL_fs = float(pta_logL(gp_fs, pps, config_fs))
    assert np.isfinite(logL_pl)
    npt.assert_allclose(logL_fs, logL_pl, rtol=1e-12)


@pytest.mark.parametrize("correlated", [False, True], ids=["curn", "hd"])
def test_free_spectrum_bins_are_independent(correlated):
    """Perturbing a single ρ bin moves logL; the bins are live parameters."""
    _, fs = _injector_pair(correlated)
    gp, pps, config = _config_and_params(fs, correlated)
    base = float(pta_logL(gp, pps, config))

    idx = gp.param_index("gwb_log10_rho_2")
    gp_bumped = gp.with_values(gp.values.at[idx].add(0.5))
    bumped = float(pta_logL(gp_bumped, pps, config))
    assert bumped != base


@pytest.mark.parametrize("correlated", [False, True], ids=["curn", "hd"])
def test_free_spectrum_logL_matches_dense_reference(correlated):
    """Absolute anchor: pta_logL == a dense numpy Gaussian logL, at a ragged ρ.

    The equivalence tests above compare two library paths to each other;
    this one compares against an independent brute-force reference — the
    full covariance is formed densely (``C_ab = δ_ab N_a + Γ_ab F_a Φ F_bᵀ``,
    with ``Φ = 10^(2ρ)`` written inline) and evaluated with
    ``np.linalg.solve``/``slogdet``.  The spectrum is deliberately ragged
    (random per-bin ρ, representable by no power law), so this checks the
    free-spectrum likelihood itself, not just power-law agreement.
    """
    from jaxpint.fitters import compute_time_residuals
    from jaxpint.pta.signals import fourier_basis, hd_orf

    rng = np.random.default_rng(77)
    rho = rng.uniform(-7.5, -6.0, N_COMP)
    phi = np.repeat(10.0 ** (2.0 * rho), 2)

    if correlated:
        pos = _positions()
        inj = HDCorrelatedGWBInjector(
            pos, N_COMP, T_SPAN,
            spectrum=FreeSpectrum(N_COMP, log10_rho=[float(r) for r in rho]),
        )
    else:
        inj = CURNInjector(
            N_COMP, T_SPAN,
            spectrum=FreeSpectrum(N_COMP, log10_rho=[float(r) for r in rho]),
        )
    gp, pps, config = _config_and_params(inj, correlated)
    logL = float(pta_logL(gp, pps, config))

    # Dense reference, assembled with numpy only.
    rs, Ns, Fs = [], [], []
    for td, tm, nm, pp in zip(
        config.toa_data_list, config.timing_models, config.noise_models, pps
    ):
        rs.append(np.asarray(compute_time_residuals(tm, td, pp)))
        Ndiag, U_n, _ = nm.covariance(td, pp)
        assert U_n.shape[1] == 0  # white-noise-only fixture
        Ns.append(np.asarray(Ndiag))
        F, _ = fourier_basis(td.tdb_seconds, N_COMP, T_SPAN)
        Fs.append(np.asarray(F))

    if correlated:
        Gamma = np.array(
            [[float(hd_orf(pos[a], pos[b])) for b in range(2)] for a in range(2)]
        )
    else:
        Gamma = np.eye(2)

    sizes = [len(r) for r in rs]
    n_tot = sum(sizes)
    C = np.zeros((n_tot, n_tot))
    offs = np.concatenate([[0], np.cumsum(sizes)])
    for a in range(2):
        sa = slice(offs[a], offs[a + 1])
        C[sa, sa] += np.diag(Ns[a])
        for b in range(2):
            sb = slice(offs[b], offs[b + 1])
            C[sa, sb] += Gamma[a, b] * (Fs[a] * phi[None, :]) @ Fs[b].T
    r = np.concatenate(rs)
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    logL_dense = -0.5 * (r @ np.linalg.solve(C, r) + logdet + n_tot * np.log(2 * np.pi))

    npt.assert_allclose(logL, logL_dense, rtol=1e-10)


def test_free_spectrum_gradient_is_finite_and_nonzero():
    """jax.grad through the correlated path w.r.t. every ρ bin (NUTS-ready)."""
    _, fs = _injector_pair(correlated=True)
    gp, pps, config = _config_and_params(fs, correlated=True)

    grad = jax.grad(lambda v: pta_logL(gp.with_values(v), pps, config))(gp.values)
    grad = np.asarray(grad)
    assert grad.shape == (N_COMP,)
    assert np.isfinite(grad).all()
    assert (grad != 0.0).all()


# ---------------------------------------------------------------------------
# Per-pulsar FreeSpectrumNoise component
# ---------------------------------------------------------------------------


def test_free_spectrum_noise_matches_matched_plrednoise():
    """Covariance triple (Ndiag, U, Φ) identical to PLRedNoise at matched ρ."""
    n_toas = 30
    F, freqs, df, _ = make_fourier_basis(n_toas, N_COMP, T_SPAN)
    td, _, _, pp = make_simple_pulsar(n_toas=n_toas, f0=200.0, f1=-1e-15, seed=9)

    plred = PLRedNoise(
        fourier_basis=F, freqs=freqs, freq_bin_widths=df,
        tnredamp_name="TNREDAMP", tnredgam_name="TNREDGAM",
    )
    rho_names = tuple(f"TNFREERHO_{k}" for k in range(N_COMP))
    fsn = FreeSpectrumNoise(
        fourier_basis=F, freqs=freqs, freq_bin_widths=df, rho_names=rho_names
    )

    from tests.helpers import make_params

    rho = _matched_log10_rho()
    params = make_params(
        ("TNREDAMP", "TNREDGAM", *rho_names),
        (LOG10_A, GAMMA, *[float(r) for r in rho]),
        frozen_mask=(True,) * (2 + N_COMP),
    )

    n1, u1, p1 = plred.covariance(td, params)
    n2, u2, p2 = fsn.covariance(td, params)
    npt.assert_array_equal(np.asarray(n1), np.asarray(n2))
    npt.assert_array_equal(np.asarray(u1), np.asarray(u2))
    npt.assert_allclose(np.asarray(p2), np.asarray(p1), rtol=1e-14)

    # And through NoiseModel's stacked Woodbury interface.
    from jaxpint.noise import ScaleToaError

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    nm_pl = NoiseModel(white_noise=white, correlated=(plred,))
    nm_fs = NoiseModel(white_noise=white, correlated=(fsn,))
    params_full = make_params(
        ("EFAC1", "EQUAD1", "TNREDAMP", "TNREDGAM", *rho_names),
        (1.0, 0.0, LOG10_A, GAMMA, *[float(r) for r in rho]),
        frozen_mask=(True,) * (4 + N_COMP),
    )
    Nd1, U1, Ph1 = nm_pl.covariance(td, params_full)
    Nd2, U2, Ph2 = nm_fs.covariance(td, params_full)
    npt.assert_array_equal(np.asarray(Nd1), np.asarray(Nd2))
    npt.assert_array_equal(np.asarray(U1), np.asarray(U2))
    npt.assert_allclose(np.asarray(Ph2), np.asarray(Ph1), rtol=1e-14)


def test_free_spectrum_noise_validates_names():
    F, freqs, df, _ = make_fourier_basis(20, N_COMP, T_SPAN)
    with pytest.raises(ValueError, match="one per frequency bin"):
        FreeSpectrumNoise(
            fourier_basis=F, freqs=freqs, freq_bin_widths=df,
            rho_names=("RHO_0",),
        )


# ---------------------------------------------------------------------------
# Combined IRN + CRN: discovery's make*_crn configurations, by composition
# ---------------------------------------------------------------------------

N_IRN = 8  # per-pulsar intrinsic bins; the CRN keeps N_COMP (< N_IRN) bins,
# so the first N_COMP frequencies are shared — discovery's truncated-CRN case.


def _interleaved_basis(toas_seconds, n_freq):
    """Interleaved sin/cos basis on the pulsar's own times (numpy)."""
    freqs = np.arange(1, n_freq + 1) / T_SPAN
    phase = 2.0 * np.pi * np.asarray(toas_seconds)[:, None] * freqs[None, :]
    F = np.stack([np.sin(phase), np.cos(phase)], axis=-1).reshape(-1, 2 * n_freq)
    return F, freqs


def _combined_config(irn: str, correlated: bool):
    """Two-pulsar PTA: per-pulsar IRN (8 bins) + free-spectrum CRN (5 bins).

    Per-pulsar IRN parameters are deliberately asymmetric so cross-pulsar
    wiring mistakes cannot cancel.  Returns the dense-reference ingredients
    alongside the config: ``irn_dense`` holds each pulsar's ``(F, w)`` with
    the weights computed by the test's own inline formulas, independent of
    the library components.
    """
    from jaxpint.noise import ScaleToaError
    from tests.helpers import make_params

    rng = np.random.default_rng(101)
    crn_rho = rng.uniform(-7.5, -6.0, N_COMP)
    crn_phi = np.repeat(10.0 ** (2.0 * crn_rho), 2)

    white = ScaleToaError(efac_names=("EFAC1",), equad_names=("EQUAD1",))
    tds, tms, nms, pps, irn_dense = [], [], [], [], []
    for i in range(2):
        td, tm, _nm, pp = make_simple_pulsar(
            n_toas=25 + 5 * i, f0=200.0 + 10.0 * i, f1=-1e-15, seed=41 + i
        )
        F_irn, freqs_irn = _interleaved_basis(td.tdb_seconds, N_IRN)
        df_irn = np.full(N_IRN, 1.0 / T_SPAN)
        if irn == "powerlaw":
            log10_A_i, gamma_i = -13.2 - 0.3 * i, 3.0 + 0.8 * i
            comp = PLRedNoise(
                fourier_basis=jnp.asarray(F_irn),
                freqs=jnp.asarray(freqs_irn),
                freq_bin_widths=jnp.asarray(df_irn),
                tnredamp_name="TNREDAMP",
                tnredgam_name="TNREDGAM",
            )
            extra_names: tuple[str, ...] = ("TNREDAMP", "TNREDGAM")
            extra_vals: tuple[float, ...] = (log10_A_i, gamma_i)
            w_irn = _powerlaw_weights_reference(
                freqs_irn, log10_A_i, gamma_i, 1.0 / T_SPAN
            )
        else:
            rho_i = rng.uniform(-7.3, -6.2, N_IRN)
            rho_names = tuple(f"TNFREERHO_{k}" for k in range(N_IRN))
            comp = FreeSpectrumNoise(
                fourier_basis=jnp.asarray(F_irn),
                freqs=jnp.asarray(freqs_irn),
                freq_bin_widths=jnp.asarray(df_irn),
                rho_names=rho_names,
            )
            extra_names = rho_names
            extra_vals = tuple(float(r) for r in rho_i)
            w_irn = np.repeat(10.0 ** (2.0 * rho_i), 2)

        pp = make_params(
            names=pp.names + extra_names,
            values=list(np.asarray(pp.values)) + list(extra_vals),
            frozen_mask=pp.frozen_mask + (True,) * len(extra_names),
            epoch_int_values=pp.epoch_int_values,
        )
        tds.append(td)
        tms.append(tm)
        nms.append(NoiseModel(white_noise=white, correlated=(comp,)))
        pps.append(pp)
        irn_dense.append((F_irn, w_irn))

    pos = _positions() if correlated else None
    spectrum = FreeSpectrum(N_COMP, log10_rho=[float(r) for r in crn_rho])
    if correlated:
        inj = HDCorrelatedGWBInjector(pos, N_COMP, T_SPAN, spectrum=spectrum)
    else:
        inj = CURNInjector(N_COMP, T_SPAN, spectrum=spectrum)
    config = PTAConfig(
        toa_data_list=tuple(tds),
        timing_models=tuple(tms),
        noise_models=tuple(nms),
        signal_injectors=() if correlated else (inj,),
        correlated_injectors=(inj,) if correlated else (),
    )
    gp = inj.register_params(GlobalParams.empty())
    return gp, tuple(pps), config, irn_dense, crn_phi, pos


@pytest.mark.parametrize("correlated", [False, True], ids=["curn", "hd"])
@pytest.mark.parametrize("irn", ["powerlaw", "freespec"])
def test_combined_irn_crn_matches_dense_reference(irn, correlated):
    """IRN + free-spectrum CRN against a dense numpy Gaussian logL.

    Certifies that JaxPINT's separate-low-rank-blocks composition IS the
    equivalent of discovery's fused IRN+CRN spectra (``makepowerlaw_crn``
    / ``makefreespectrum_crn``): nested frequencies (CRN bins coincide
    with the first IRN bins), truncated CRN (5 of 8 bins), asymmetric
    per-pulsar IRN, both spectra live — one absolute reference.
    """
    from jaxpint.fitters import compute_time_residuals
    from jaxpint.pta.signals import fourier_basis, hd_orf

    gp, pps, config, irn_dense, crn_phi, pos = _combined_config(irn, correlated)
    logL = float(pta_logL(gp, pps, config))

    rs, Ns, Fs_crn = [], [], []
    for td, tm, nm, pp in zip(
        config.toa_data_list, config.timing_models, config.noise_models, pps
    ):
        rs.append(np.asarray(compute_time_residuals(tm, td, pp)))
        Ndiag, _U, _P = nm.covariance(td, pp)  # white diagonal; IRN added below
        Ns.append(np.asarray(Ndiag))
        F, _ = fourier_basis(td.tdb_seconds, N_COMP, T_SPAN)
        Fs_crn.append(np.asarray(F))

    if correlated:
        Gamma = np.array(
            [[float(hd_orf(pos[a], pos[b])) for b in range(2)] for a in range(2)]
        )
    else:
        Gamma = np.eye(2)

    sizes = [len(r) for r in rs]
    n_tot = sum(sizes)
    offs = np.concatenate([[0], np.cumsum(sizes)])
    C = np.zeros((n_tot, n_tot))
    for a in range(2):
        sa = slice(offs[a], offs[a + 1])
        F_irn, w_irn = irn_dense[a]
        C[sa, sa] += np.diag(Ns[a]) + (F_irn * w_irn[None, :]) @ F_irn.T
        for b in range(2):
            sb = slice(offs[b], offs[b + 1])
            C[sa, sb] += Gamma[a, b] * (Fs_crn[a] * crn_phi[None, :]) @ Fs_crn[b].T
    r = np.concatenate(rs)
    sign, logdet = np.linalg.slogdet(C)
    assert sign > 0
    logL_dense = -0.5 * (
        r @ np.linalg.solve(C, r) + logdet + n_tot * np.log(2 * np.pi)
    )

    npt.assert_allclose(logL, logL_dense, rtol=1e-10)


def test_combined_free_spectrum_gradients():
    """Both parameter routes of the joint free-spec model are sampler-ready.

    Gradients w.r.t. the global CRN ρ bins AND one pulsar's per-pulsar IRN
    ρ bins — the full free-spectrum posterior NUTS would see.
    """
    gp, pps, config, _, _, _ = _combined_config("freespec", correlated=True)

    g_gp = np.asarray(
        jax.grad(lambda v: pta_logL(gp.with_values(v), pps, config))(gp.values)
    )
    assert g_gp.shape == (N_COMP,)
    assert np.isfinite(g_gp).all()
    assert (g_gp != 0.0).all()

    def with_p0(vals):
        return pta_logL(gp, (pps[0].with_values(vals),) + pps[1:], config)

    g0 = np.asarray(jax.grad(with_p0)(pps[0].values))
    rho_idx = [pps[0].names.index(f"TNFREERHO_{k}") for k in range(N_IRN)]
    assert np.isfinite(g0).all()
    assert (g0[rho_idx] != 0.0).all()


# ---------------------------------------------------------------------------
# Prior assembly for free-spectrum parameters
# ---------------------------------------------------------------------------


def test_free_spectrum_priors_match_registered_names():
    """free_spectrum_priors covers exactly what the injector registers."""
    from jaxpint.bayes.samplers import free_spectrum_priors

    spectrum = FreeSpectrum(3)
    inj = CURNInjector(3, T_SPAN, spectrum=spectrum)
    gp = inj.register_params(GlobalParams.empty())
    spec = free_spectrum_priors(spectrum)
    assert spec.owned_names() == set(gp.names)
    d = spec.flat["gwb_log10_rho_0"]
    assert float(d.low) == -9.0 and float(d.high) == -4.0
    # Bare bin count + custom prefix form.
    spec2 = free_spectrum_priors(3, prefix="crn_")
    assert "crn_log10_rho_2" in spec2.owned_names()


if __name__ == "__main__":
    generate_golden_values()
