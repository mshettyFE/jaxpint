"""Power-law DM noise model for JaxPINT.

Achromatic-in-form red noise applied to dispersion: the Fourier design
matrix is pre-scaled by ``(1400 / f_obs)²`` by the bridge, so DM's
inverse-frequency-squared dependence is baked into the basis and the
component is otherwise identical to :class:`~jaxpint.noise.PLRedNoise`.
Shared machinery lives in
:class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.PL_DM_NOISE, pint_names=("PLDMNoise",))
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

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PLDMNoise]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        import numpy as np
        import jax.numpy as jnp
        from jaxpint._build_context import basis_seconds, span_seconds
        from jaxpint.utils import build_fourier_basis

        par = ctx.par
        toa_data = ctx.toa_data
        if toa_data is None:
            return None
        basis_s = basis_seconds(toa_data)
        n_freqs = par.int_params.get("TNDMC", 30)
        T = span_seconds(par, basis_s, "TNDMTSPAN")

        F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

        bary_freqs_mhz = np.asarray(toa_data.freq)
        D = (1400.0 / bary_freqs_mhz) ** 2
        F_dm = F * D[:, None]

        return cls(
            fourier_basis=jnp.asarray(F_dm),
            freqs=jnp.asarray(freqs),
            freq_bin_widths=jnp.asarray(freq_bin_widths),
            tndmamp_name="TNDMAMP",
            tndmgam_name="TNDMGAM",
        )

    @property
    def _amp_name(self) -> str:
        return self.tndmamp_name

    @property
    def _gam_name(self) -> str:
        return self.tndmgam_name

    def static_basis(self) -> Float[Array, "n_toas n_basis"]:
        # Fixed basis (DM scaling pre-baked) -> pre-stackable by NoiseModel.
        return self.fourier_basis
