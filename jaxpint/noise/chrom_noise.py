"""Power-law chromatic noise model for JaxPINT.

Like :class:`~jaxpint.noise.PLRedNoise`, but the (unscaled) Fourier basis is
multiplied at evaluation time by ``(f_ref / f_obs)^α``, where ``α`` is the
chromatic index (``TNCHROMIDX``). Because the scaling depends on a (possibly
fitted) parameter, this is a *dynamic*-basis component (it does not pre-stack)
and the scaling is computed at runtime to stay JAX-differentiable. Shared
machinery lives in :class:`~jaxpint.noise._fourier_gp._PowerLawFourierNoise`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl
from jaxpint.noise._fourier_gp import _PowerLawFourierNoise
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.PL_CHROM_NOISE, pint_names=("PLChromNoise",))
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

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PLChromNoise]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        import jax.numpy as jnp
        from jaxpint._build_context import basis_seconds, span_seconds
        from jaxpint.utils import build_fourier_basis

        par = ctx.par
        toa_data = ctx.toa_data
        if toa_data is None:
            return None
        basis_s = basis_seconds(toa_data)
        n_freqs = par.int_params.get("TNCHROMC", 30)
        T = span_seconds(par, basis_s, "TNCHROMTSPAN")

        F, freqs, freq_bin_widths = build_fourier_basis(basis_s, n_freqs, T)

        return cls(
            fourier_basis=jnp.asarray(F),
            freqs=jnp.asarray(freqs),
            freq_bin_widths=jnp.asarray(freq_bin_widths),
            tnchromamp_name="TNCHROMAMP",
            tnchromgam_name="TNCHROMGAM",
            tnchromidx_name="TNCHROMIDX",
            fref=1400.0,
        )

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
