"""Power-law chromatic noise model for JaxPINT.

Like :class:`~jaxpint.noise.PLRedNoise`, but the (unscaled) Fourier basis is
multiplied at evaluation time by ``(f_ref / f_obs)^α``, where ``α`` is the
chromatic index (``TNCHROMIDX``). Because the scaling depends on a (possibly
fitted) parameter, this is a *dynamic*-basis component (it does not pre-stack)
and the scaling is computed at runtime to stay JAX-differentiable. Shared
machinery lives in :class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise
from jaxpint.types import TOAData, ParameterVector


class PLChromNoise(_PowerLawFourierNoise):
    """Power-law chromatic noise with arbitrary chromatic index.

    At each ``covariance`` / ``generate`` call the basis is multiplied
    by ``(f_ref / f_obs)^α`` (``α`` from ``TNCHROMIDX``), keeping the scaling
    differentiable through ``α``.

    Parameters
    ----------
    fourier_basis : (n_toas, 2 * n_freqs)
        Raw (unscaled) Fourier design matrix with alternating sin/cos columns.
    freqs : (n_freqs,)
        Frequency array in Hz.
    freq_bin_widths : (n_freqs,)
        Δf for each frequency bin.
    tnchromamp_name : str
        Parameter name for the log10 amplitude.
    tnchromgam_name : str
        Parameter name for the spectral index.
    tnchromidx_name : str
        Parameter name for the chromatic index (α).
    fref : float
        Reference radio frequency in MHz (default 1400.0).
    """

    PARAMS = (
        ParamDecl("TNCHROMAMP"),
        ParamDecl("TNCHROMGAM"),
        ParamDecl("TNCHROMIDX"),
        ParamDecl("TNCHROMC", kind="int"),
        ParamDecl("TNCHROMTSPAN"),
    )

    tnchromamp_name: str = eqx.field(static=True)
    tnchromgam_name: str = eqx.field(static=True)
    tnchromidx_name: str = eqx.field(static=True)
    fref: float = eqx.field(static=True, default=1400.0)

    @property
    def _amp_name(self) -> str:
        return self.tnchromamp_name

    @property
    def _gam_name(self) -> str:
        return self.tnchromgam_name

    def _basis(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, "n_toas n_basis"]:
        """Fourier basis scaled by ``(f_ref / f_obs)^α`` per TOA."""
        alpha = params.param_value(self.tnchromidx_name)
        D = (self.fref / toa_data.freq) ** alpha  # (n_toas,)
        return self._fourier_basis_jax * D[:, None]
