"""Power-law red noise model for JaxPINT.

Implements achromatic red noise with a power-law power spectral density
using an alternating Fourier basis (sin/cos pairs), matching PINT's
``PLRedNoise`` component.

The noise covariance is decomposed as::

    C_rn = F · diag(w) · Fᵀ

where *F* is a Fourier design matrix (pre-computed by the bridge) and
*w* are the power-law PSD weights computed from the amplitude and
spectral index parameters. The shared machinery lives in
:class:`~jaxpint.noise._power_law._PowerLawFourierNoise`.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._power_law import _PowerLawFourierNoise


class PLRedNoise(_PowerLawFourierNoise):
    """Power-law red noise via an alternating Fourier basis.

    The Fourier design matrix *F* is pre-computed by the bridge from TOA times.
    The PSD weights depend on the amplitude (``TNREDAMP``) and spectral index
    (``TNREDGAM``) parameters and are computed dynamically (differentiable). The
    basis is fixed, so this is a static-basis component.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Pre-computed Fourier design matrix with alternating sin/cos
        columns: ``[sin(2πf₁t), cos(2πf₁t), sin(2πf₂t), ...]``.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin (used to weight the PSD).
    tnredamp_name : str
        Parameter name for the log10 amplitude.
    tnredgam_name : str
        Parameter name for the spectral index.
    """

    PARAMS = (
        ParamDecl("TNREDAMP"),
        ParamDecl("TNREDGAM"),
        ParamDecl("RNAMP"),
        ParamDecl("RNIDX"),
        ParamDecl("TNREDC", kind="int"),
        ParamDecl("TNREDTSPAN"),
    )

    tnredamp_name: str = eqx.field(static=True)
    tnredgam_name: str = eqx.field(static=True)

    @property
    def _amp_name(self) -> str:
        return self.tnredamp_name

    @property
    def _gam_name(self) -> str:
        return self.tnredgam_name

    def static_basis(self) -> Float[Array, "n_toas n_basis"]:
        # Fixed basis -> advertise it so NoiseModel can pre-stack it once.
        return self.fourier_basis
