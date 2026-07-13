"""GWB Fourier basis <-> PSD frequency alignment.

Regression guard for a real bug: ``fourier_basis`` must order its columns so
that ``get_psd`` / ``gwb_covariance``'s ``jnp.repeat(psd, 2)`` assigns each
frequency's PSD to *that* frequency's (sin, cos) pair.  A blocked
``[sin... | cos...]`` layout paired with the interleaved ``repeat`` PSD
misaligns them, mis-weighting the GW spectrum.

Deliberately *not* self-consistent: they assemble the GW
covariance from first principles, per frequency, so they fail under a
basis/PSD misalignment.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jaxpint.pta.signals.gwb import fourier_basis, powerlaw_psd


jax.config.update("jax_enable_x64", True)


def test_fourier_basis_is_interleaved():
    """Columns are [sin f1, cos f1, sin f2, cos f2, ...]."""
    T = 365.25 * 86400.0
    n = 4
    tt = jnp.linspace(0.0, T, 37)
    F, freqs = fourier_basis(tt, n, T)
    F = np.asarray(F)
    tt_n, fr = np.asarray(tt), np.asarray(freqs)
    for k in range(n):
        # atol: sin/cos cross zero, where a pure-rtol check would blow up.
        np.testing.assert_allclose(
            F[:, 2 * k], np.sin(2 * np.pi * fr[k] * tt_n), atol=1e-12
        )
        np.testing.assert_allclose(
            F[:, 2 * k + 1], np.cos(2 * np.pi * fr[k] * tt_n), atol=1e-12
        )


def test_gwb_covariance_frequency_alignment():
    """Each frequency's PSD weights ITS OWN sin & cos column.

    Build ``C = F diag(Φ) Fᵀ`` from the library (Φ = ``repeat(psd, 2)``) and
    compare to an independently assembled ``C = Σ_k psd_k (s_k s_kᵀ + c_k c_kᵀ)``.
    A blocked-vs-interleaved mismatch permutes which ``psd`` multiplies which
    frequency and breaks this equality.
    """
    T = 365.25 * 86400.0
    n = 4
    log10_A, gamma = -14.0, 4.33
    tt = jnp.linspace(0.0, T, 50)

    F, freqs = fourier_basis(tt, n, T)
    df = 1.0 / T
    psd = powerlaw_psd(freqs, log10_A, gamma) * df
    Phi = jnp.repeat(psd, 2)  # the ordering the library uses
    C_lib = np.asarray(F @ jnp.diag(Phi) @ F.T)

    tt_n, fr, ps = np.asarray(tt), np.asarray(freqs), np.asarray(psd)
    C_ref = np.zeros((tt_n.size, tt_n.size))
    for k in range(n):
        s = np.sin(2 * np.pi * fr[k] * tt_n)
        c = np.cos(2 * np.pi * fr[k] * tt_n)
        C_ref += ps[k] * (np.outer(s, s) + np.outer(c, c))

    np.testing.assert_allclose(C_lib, C_ref, rtol=1e-10, atol=0.0)
