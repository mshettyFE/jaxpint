"""White-noise (EFAC/EQUAD/ECORR) logL cross-validation vs enterprise.

Two layers per feature:

- *kernel-level*: enterprise's own residual vector is fed into a dense-numpy
  Gaussian logL built from JaxPINT's covariance ingredients — this isolates
  noise-convention parity from residual parity and carries the tight
  tolerances;
- *end-to-end*: JaxPINT's ``single_pulsar_logL`` (its own residuals + solver)
  vs the enterprise PTA, tolerance limited by the ~1e-9 s residual parity
  measured in test_ent_building_blocks.py.

The absolute white-noise test doubles as the pin for the normalization
constant
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from tests.enterprise_checks._ent_helpers import (
    WHITE_ECORR_US,
    WHITE_EFAC,
    WHITE_EQUAD_US,
    dense_logL,
)

# Single-sourced with conftest's white_bundle par via _ent_helpers.WHITE_*.
EFAC = WHITE_EFAC
EQUAD_S = WHITE_EQUAD_US * 1e-6
ECORR_S = WHITE_ECORR_US * 1e-6


@pytest.fixture(scope="module")
def ent_signals():
    """Enterprise signal classes for the white_bundle's par values."""
    from enterprise.signals import gp_signals, parameter, white_signals

    mn = white_signals.MeasurementNoise(
        efac=parameter.Constant(EFAC),
        log10_t2equad=parameter.Constant(np.log10(EQUAD_S)),
    )
    ec_kernel = white_signals.EcorrKernelNoise(
        log10_ecorr=parameter.Constant(np.log10(ECORR_S))
    )
    ec_basis = gp_signals.EcorrBasisModel(
        log10_ecorr=parameter.Constant(np.log10(ECORR_S))
    )
    return mn, ec_kernel, ec_basis


def test_efac_equad_scaled_variance(white_bundle):
    """JaxPINT Ndiag follows the tempo2/T2EQUAD convention EFAC^2*(err^2+EQUAD^2).

    Enterprise's MeasurementNoise(log10_t2equad=...) implements the same
    formula; its side is pinned against JaxPINT's Ndiag by the exact logL
    identity in test_white_logL_absolute_pins_normalization.
    """
    b = white_bundle
    Ndiag = np.asarray(b.noise_model.scaled_sigma(b.toa_data, b.params)) ** 2
    expected = EFAC**2 * (np.asarray(b.toa_data.error) ** 2 + EQUAD_S**2)
    npt.assert_allclose(Ndiag, expected, rtol=1e-12)


def test_white_logL_absolute_pins_normalization(white_bundle, ent_signals):
    """Enterprise white-only PTA logL == dense logL from JaxPINT's Ndiag.

    Uses enterprise's own residuals on both sides, so any nonzero difference
    is a white-noise covariance or normalization-constant mismatch.  Observed
    agreement: exact to float64 (diff 0.0 on |logL| ~ 1.6e3).
    """
    from enterprise.signals import signal_base

    b = white_bundle
    mn, _, _ = ent_signals
    pta = signal_base.PTA([signal_base.SignalCollection([mn])(b.psr)])
    logL_ent = pta.get_lnlikelihood({})

    Ndiag = np.asarray(b.noise_model.scaled_sigma(b.toa_data, b.params)) ** 2
    logL_manual = dense_logL(b.psr.residuals, Ndiag, np.zeros((len(Ndiag), 0)), np.zeros(0))
    npt.assert_allclose(
        logL_ent,
        logL_manual,
        rtol=1e-12,
        err_msg="white-noise covariance or logL normalization mismatch",
    )


def test_ecorr_kernel_vs_basis_vs_jax(white_bundle, ent_signals):
    """ECORR: enterprise Sherman-Morrison, enterprise basis GP, and JaxPINT's
    (Ndiag, U, Phi) all describe the same covariance.

    Kernel-level with enterprise residuals; observed agreement ~3e-12.
    """
    from enterprise.signals import signal_base

    b = white_bundle
    mn, ec_kernel, ec_basis = ent_signals
    logL_sm = signal_base.PTA([(mn + ec_kernel)(b.psr)]).get_lnlikelihood({})
    logL_basis = signal_base.PTA([(mn + ec_basis)(b.psr)]).get_lnlikelihood({})

    Ndiag, U, Phi = b.noise_model.covariance(b.toa_data, b.params)
    logL_jax = dense_logL(b.psr.residuals, Ndiag, U, Phi)

    npt.assert_allclose(logL_sm, logL_jax, rtol=1e-10,
                        err_msg="EcorrKernelNoise vs JaxPINT ECORR mismatch")
    npt.assert_allclose(logL_basis, logL_jax, rtol=1e-10,
                        err_msg="EcorrBasisModel vs JaxPINT ECORR mismatch")


def test_white_logL_end_to_end(white_bundle, ent_signals):
    """Full JaxPINT single_pulsar_logL vs enterprise PTA (same normalization).

    Tolerance is set by residual parity (~6e-10 s max, building-blocks
    module), which propagates to ~1e-3 absolute in logL for these SNRs.
    """
    from enterprise.signals import signal_base

    from jaxpint.likelihood import single_pulsar_logL

    b = white_bundle
    mn, ec_kernel, _ = ent_signals
    logL_ent = signal_base.PTA([(mn + ec_kernel)(b.psr)]).get_lnlikelihood({})
    logL_jax = float(
        single_pulsar_logL(b.toa_data, b.timing_model, b.noise_model, b.params)
    )
    npt.assert_allclose(
        logL_jax,
        logL_ent,
        atol=5e-3,
        rtol=0,
        err_msg="end-to-end white-noise logL mismatch beyond residual-parity budget",
    )
