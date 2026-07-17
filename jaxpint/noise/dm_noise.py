"""Power-law DM noise model for JaxPINT.

Achromatic-in-form red noise applied to dispersion: the Fourier design
matrix is pre-scaled by ``(1400 / f_obs)²`` by the bridge, so DM's
inverse-frequency-squared dependence is baked into the basis and the
component is otherwise identical to :class:`~jaxpint.noise.PLRedNoise`.
Shared machinery lives in
:class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise


class PLDMNoise(_PowerLawFourierNoise):
    """Power-law DM noise via a frequency-scaled Fourier basis.

    The Fourier design matrix is pre-scaled by ``(1400 / f_obs)²`` by the bridge,
    so it is fixed (static-basis component). The PSD weights depend on the
    amplitude (``TNDMAMP``) and spectral index (``TNDMGAM``) parameters.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Pre-computed Fourier design matrix already scaled by
        ``(1400 / f_obs)²`` per TOA.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin (used to weight the PSD).
    tndmamp_name : str
        Parameter name for the log10 amplitude.
    tndmgam_name : str
        Parameter name for the spectral index.
    """

    PARAMS = (
        ParamDecl("TNDMAMP"),
        ParamDecl("TNDMGAM"),
        ParamDecl("TNDMC", kind="int"),
        ParamDecl("TNDMTSPAN"),
    )

    tndmamp_name: str = eqx.field(static=True)
    tndmgam_name: str = eqx.field(static=True)

    @property
    def _amp_name(self) -> str:
        return self.tndmamp_name

    @property
    def _gam_name(self) -> str:
        return self.tndmgam_name

    def static_basis(self) -> Float[Array, "n_toas n_basis"]:
        # Fixed basis (DM scaling pre-baked) -> pre-stackable by NoiseModel.
        return self.fourier_basis
