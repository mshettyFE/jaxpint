"""Pure-math parity between JaxPINT and enterprise building blocks.

Fast smoke module: one enterprise pulsar, no PTA objects.  Each test compares
a single ingredient (times, residuals, Fourier basis, PSD weights, ECORR
quantization, ORF) so that any failure in the heavier logL modules can be
localized here first.

Convention divergences are pinned as explicit sentinel tests rather than
hidden behind loose tolerances:

- BOTH stacks evaluate GP bases at *barycentered* TOAs, and the time arrays
  are identical ;
- enterprise's HD ORF auto-term is 1.0 (pulsar term included, the NANOGrav
  convention), JaxPINT's ``hd_orf`` self-correlation is 0.5
  (``test_hd_orf_diagonal_convention``).
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt


def test_toas_and_errors_match(white_bundle):
    """TOA uncertainties agree exactly; TOA epochs differ only by sort order."""
    b = white_bundle
    npt.assert_allclose(
        b.psr.toaerrs,
        np.asarray(b.toa_data.error),
        rtol=1e-14,
        err_msg="TOA uncertainties (seconds) disagree between stacks",
    )
    assert len(b.psr.toas) == b.toa_data.n_toas


def test_gp_basis_frame_matches_enterprise(white_bundle):
    """Sentinel: both stacks' GP-basis times are barycentric — and identical.

    ``PintPulsar._toas = model.get_barycentric_toas(toas)``, and JaxPINT's
    ``TOAData.basis_seconds`` (the coordinate every noise/GWB Fourier basis
    and ECORR quantization is evaluated at, ``basis_coord='barycentric'`` for
    bridge-loaded data) is the same quantity — measured agreement is exact
    (0 s).      """
    b = white_bundle
    basis = np.asarray(b.toa_data.basis_seconds)
    dt = np.abs(b.psr.toas - basis)
    assert dt.max() < 1e-6, (
        f"GP-basis frames diverged (max {dt.max():.3g} s); a time convention "
        "changed in one stack — re-derive the GP logL tolerances"
    )
    # And plain TDB is still ~a Roemer delay away: basis_seconds is genuinely
    # a different (barycentric) coordinate, not an alias of tdb_seconds — so
    # the frame agreement above is a real statement, not a trivial one.
    dt_tdb = np.abs(b.psr.toas - np.asarray(b.toa_data.tdb_seconds))
    assert 1.0 < dt_tdb.max() < 600.0


def test_residuals_match(white_bundle):
    """Raw residual parity (calibrates end-to-end logL tolerances).

    PHOFF in the par file makes PINT (hence enterprise) skip implicit
    weighted-mean subtraction, so both stacks compute raw phase/F0 residuals.
    Observed agreement ~6e-10 s on ~1e-6 s residuals.
    """
    from jaxpint.fitters import compute_time_residuals

    b = white_bundle
    r_jax = np.asarray(compute_time_residuals(b.timing_model, b.toa_data, b.params))
    npt.assert_allclose(
        r_jax,
        b.psr.residuals,
        atol=5e-9,
        rtol=0,
        err_msg="raw time residuals disagree between stacks",
    )


def test_fourier_design_matrix(white_bundle):
    """Fourier bases are identical at identical times (interleaved sin/cos).

    Evaluating JaxPINT's ``build_fourier_basis`` at enterprise's own
    (barycentered) toas with enterprise's Tspan isolates pure math from the
    time-convention gap; observed agreement is bit-exact.
    """
    from enterprise.signals.utils import createfourierdesignmatrix_red

    from jaxpint.utils import build_fourier_basis

    b = white_bundle
    tspan = float(b.psr.toas.max() - b.psr.toas.min())
    F_ent, freqs_ent = createfourierdesignmatrix_red(b.psr.toas, nmodes=10, Tspan=tspan)
    F_jax, freqs_jax, _ = build_fourier_basis(b.psr.toas, 10, tspan)
    npt.assert_allclose(
        np.repeat(freqs_jax, 2),
        freqs_ent,
        rtol=1e-14,
        err_msg="Fourier frequency grids disagree",
    )
    npt.assert_allclose(
        np.asarray(F_jax),
        F_ent,
        atol=1e-12,
        rtol=0,
        err_msg="Fourier design matrices disagree (column ordering?)",
    )


def test_powerlaw_prior_weights(white_bundle):
    """powerlaw PSD weights: S(f)*df per (sin,cos) pair, identical formulas."""
    from enterprise.signals.utils import createfourierdesignmatrix_red, powerlaw

    from jaxpint.spectra import powerlaw_psd
    from jaxpint.utils import build_fourier_basis

    b = white_bundle
    tspan = float(b.psr.toas.max() - b.psr.toas.min())
    _, freqs_ent = createfourierdesignmatrix_red(b.psr.toas, nmodes=10, Tspan=tspan)
    _, freqs_jax, widths_jax = build_fourier_basis(b.psr.toas, 10, tspan)
    for log10_A, gamma in [(-14.0, 4.33), (-13.0, 2.0), (-15.5, 6.5)]:
        phi_ent = powerlaw(freqs_ent, log10_A=log10_A, gamma=gamma)
        phi_jax = np.repeat(
            np.asarray(powerlaw_psd(freqs_jax, log10_A, gamma) * widths_jax), 2
        )
        npt.assert_allclose(
            phi_jax,
            phi_ent,
            rtol=1e-12,
            err_msg=f"powerlaw weights disagree at (log10_A={log10_A}, gamma={gamma})",
        )


def test_quantization_matrix(white_bundle):
    """ECORR epoch quantization: identical groupings (dt=1 s, nmin=2).

    Compared as sets of TOA-index groups so column order is irrelevant.
    Enterprise quantizes barycentered toas and JaxPINT TDB toas, but the
    0.3 s intra-epoch clustering is far inside dt=1 s for both.
    """
    from enterprise.signals.utils import create_quantization_matrix

    from jaxpint.noise.ecorr import EcorrNoise

    b = white_bundle
    ecorr = next(c for c in b.noise_model.correlated if isinstance(c, EcorrNoise))
    U_jax = np.asarray(ecorr.quantization_matrix)
    U_ent, _ = create_quantization_matrix(b.psr.toas, dt=1, nmin=2)
    assert U_jax.shape[1] > 0, "JaxPINT ECORR basis is empty — fixture regression"
    assert U_ent.shape == U_jax.shape

    def groups(U):
        return sorted(tuple(np.nonzero(U[:, j])[0]) for j in range(U.shape[1]))

    assert groups(U_ent) == groups(U_jax), "ECORR epoch groupings disagree"


def test_hd_orf_offdiagonal():
    """Hellings-Downs curve agrees for distinct pulsar pairs."""
    import jax.numpy as jnp
    from enterprise.signals.utils import hd_orf as ent_hd_orf

    from jaxpint.pta.signals.orf import hd_orf

    rng = np.random.default_rng(7)
    for _ in range(10):
        p1, p2 = rng.normal(size=(2, 3))
        p1 /= np.linalg.norm(p1)
        p2 /= np.linalg.norm(p2)
        npt.assert_allclose(
            float(hd_orf(jnp.asarray(p1), jnp.asarray(p2))),
            float(ent_hd_orf(p1, p2)),
            rtol=1e-10,
            err_msg="HD ORF disagrees for an off-diagonal pulsar pair",
        )


def test_hd_orf_diagonal_convention():
    """Sentinel: HD auto-correlation is 1.0 in enterprise, 0.5 in JaxPINT.

    enterprise's ``utils.hd_orf`` special-cases ``pos1 == pos2`` to 1.0 (the
    NANOGrav convention: the auto-term includes the pulsar term), while
    JaxPINT's ``hd_orf`` clips the log and returns the 0.5 self-correlation
    limit — and ``HDCorrelatedGWBInjector`` puts that on the ORF diagonal,
    halving the GWB auto-power relative to enterprise.  This is a genuine
    convention gap awaiting adjudication; ``test_ent_hd_gwb.py`` shows the
    likelihood machinery matches once the diagonal is overridden to 1.0.
    If JaxPINT is changed to the enterprise convention, update this test and
    flip the sentinel in test_ent_hd_gwb.py.
    """
    import jax.numpy as jnp
    from enterprise.signals.utils import hd_orf as ent_hd_orf

    from jaxpint.pta.signals.orf import hd_orf

    pos = np.array([0.3, -0.4, np.sqrt(1 - 0.25)])
    assert float(ent_hd_orf(pos, pos)) == 1.0
    npt.assert_allclose(float(hd_orf(jnp.asarray(pos), jnp.asarray(pos))), 0.5, rtol=1e-12)


def test_orf_matrix_of_injector_offdiagonal():
    """Injector ORF matrix off-diagonals match enterprise pair-by-pair.

    (Random unit vectors rather than the pta_bundles fixture: building three
    enterprise pulsars is slow and belongs to the --runslow modules only.)
    """
    import jax.numpy as jnp
    from enterprise.signals.utils import hd_orf as ent_hd_orf

    from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector

    rng = np.random.default_rng(11)
    positions = rng.normal(size=(3, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    inj = HDCorrelatedGWBInjector(
        pulsar_positions=jnp.asarray(positions), n_components=4, T_span=1e8
    )
    Gamma = np.asarray(inj.get_orf_matrix())
    for a in range(3):
        for bb in range(3):
            if a == bb:
                continue
            npt.assert_allclose(
                Gamma[a, bb],
                float(ent_hd_orf(positions[a], positions[bb])),
                rtol=1e-10,
                err_msg=f"ORF matrix entry ({a},{bb}) disagrees",
            )


def test_module_requires_both_stacks():
    """Guard: this module must import both pint and enterprise at test level
    so the root conftest's source-regex tags it requires_pint AND
    requires_enterprise (imports below are the literal match targets)."""
    import enterprise  # noqa: F401
    import pint  # noqa: F401
