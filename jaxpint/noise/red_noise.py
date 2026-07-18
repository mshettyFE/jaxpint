"""Power-law red noise model for JaxPINT.

Implements achromatic red noise with a power-law power spectral density
using an alternating Fourier basis (sin/cos pairs), matching PINT's
``PLRedNoise`` component.

The noise covariance is decomposed as::

    C_rn = F · diag(w) · Fᵀ

where *F* is a Fourier design matrix (pre-computed by the bridge) and
*w* are the power-law PSD weights computed from the amplitude and
spectral index parameters. The shared machinery lives in
:class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.PL_RED_NOISE, pint_names=("PLRedNoise",))
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

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PLRedNoise]":
        """Construct from a parsed model (co-located with the physics it builds).

        Builds the Fourier design matrix from the pulsar's basis times; ``None``
        when no TOA data is available (the basis can't be built).
        """
        from jaxpint._build_context import basis_seconds, span_seconds
        from jaxpint.utils import build_fourier_basis

        toa_data = ctx.toa_data
        if toa_data is None:
            return None
        basis_s = basis_seconds(toa_data)
        n_freqs = ctx.par.int_params.get("TNREDC", 30)
        T = span_seconds(ctx.par, basis_s, "TNREDTSPAN")

        F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)
        return cls(
            fourier_basis=jnp.asarray(F),
            freqs=jnp.asarray(freqs),
            freq_bin_widths=jnp.asarray(freq_bin_widths),
            tnredamp_name="TNREDAMP",
            tnredgam_name="TNREDGAM",
        )
