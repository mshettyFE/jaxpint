"""Free-spectrum (per-frequency) Fourier noise component.

The free-spectrum sibling of :class:`~jaxpint.noise.PLRedNoise`: instead of
two power-law hyperparameters, each frequency bin carries its own amplitude
parameter ``log10_ρ_k`` (per-bin RMS in seconds, discovery's
``freespectrum`` convention — ``Δf`` is absorbed into ρ).

The covariance is still ``C = F · diag(w) · Fᵀ`` with **diagonal** weights
``w_k = 10^(2·log10_ρ_k)`` — a free spectrum changes how many
hyperparameters fill the diagonal, not the Woodbury structure, so it runs
through the identical :class:`~jaxpint.noise.NoiseModel` solve path.

This component is constructed programmatically and declares no ``PARAMS``
for the par registry: there is no standard par-file syntax for per-bin
*power* hyperparameters (PINT/tempo2/TempoNest par files only carry
power-law keywords like ``TNREDAMP``/``TNREDGAM``; the per-bin ``WAVE_N``
/ ``WXSIN_``/``WXCOS_`` families are deterministic Fourier *coefficients*,
i.e. the flat-prior limit of this GP, not per-bin variances).  The
ecosystem convention for free-spectrum hyperparameters instead lives in
enterprise/discovery noise dictionaries and chain files, named
``{psr}_red_noise_log10_rho_{k}`` (enterprise) or ``..._log10_rho(k)``
(discovery).  Supply parameter names at construction; following the
``log10_rho`` naming keeps future noisedict interop straightforward.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint._psd import expand_sin_cos, free_spectrum_psd
from jaxpint.noise._fourier_gp import _FourierGPNoise
from jaxpint.types import ParameterVector


class FreeSpectrumNoise(_FourierGPNoise):
    """Free-spectrum noise on a fixed alternating sin/cos Fourier basis.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Pre-computed Fourier design matrix with alternating sin/cos
        columns (as for :class:`~jaxpint.noise.PLRedNoise`).
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf per bin.  Kept for interface symmetry with the power-law
        components; **not** used in the weights (ρ absorbs Δf).
    rho_names : tuple of str
        Per-frequency parameter names, in frequency order (one per bin,
        e.g. ``("TNFREERHO_0001", …)``), each holding ``log10_ρ_k``.
    """

    rho_names: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        n_freq = len(self.freqs)
        if len(self.rho_names) != n_freq:
            raise ValueError(
                f"rho_names has {len(self.rho_names)} entries, expected "
                f"{n_freq} (one per frequency bin)"
            )

    def psd_weights(self, params: ParameterVector) -> Float[Array, " n_basis"]:
        """Per-bin weights ``10^(2·log10_ρ_k)``, repeated for the sin/cos pair."""
        log10_rho = params.param_values(self.rho_names)
        return expand_sin_cos(free_spectrum_psd(log10_rho))

    def static_basis(self) -> Float[Array, "n_toas n_basis"]:
        # Fixed basis -> advertise it so NoiseModel can pre-stack it once.
        return self.fourier_basis
