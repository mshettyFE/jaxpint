"""Fourier-basis timing noise delay component (WaveX).

The delay is modelled as a sum of sinusoids at specified frequencies:

    delay(t) = Σ_i (WXSIN_i * sin(2π * WXFREQ_i * (t - WXEPOCH - delay))
                  + WXCOS_i * cos(2π * WXFREQ_i * (t - WXEPOCH - delay)))

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import SECS_PER_DAY
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import fourier_sum

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.WAVE_X, pint_names=("WaveX",))
class WaveX(DelayComponent):
    """Fourier-basis timing noise (WaveX).

    Parameters
    ----------
    n_components : int
        Number of Fourier components.
    wxepoch_name : str
        Name of the reference epoch parameter.
    wxfreq_names : tuple[str, ...]
        Names of the frequency parameters (1/day).
    wxsin_names : tuple[str, ...]
        Names of the sine amplitude parameters (seconds).
    wxcos_names : tuple[str, ...]
        Names of the cosine amplitude parameters (seconds).

    Raises
    ------
    ValueError
        If ``n_components`` is less than 1.
    ValueError
        If the length of ``wxfreq_names``, ``wxsin_names``, or
        ``wxcos_names`` does not match ``n_components``.
    """

    PARAMS = (
        ParamDecl("WXFREQ_0001", prefix="WXFREQ_"),
        ParamDecl("WXSIN_0001", prefix="WXSIN_", frozen_default=False),
        ParamDecl("WXCOS_0001", prefix="WXCOS_", frozen_default=False),
        ParamDecl("WXEPOCH", kind="mjd"),
    )

    n_components: int = eqx.field(static=True)
    wxfreq_names: tuple[str, ...] = eqx.field(static=True)
    wxsin_names: tuple[str, ...] = eqx.field(static=True)
    wxcos_names: tuple[str, ...] = eqx.field(static=True)
    wxepoch_name: str = eqx.field(static=True, default="WXEPOCH")

    def __check_init__(self):
        self.check_name_tuples(
            "n_components",
            "wxfreq_names",
            "wxsin_names",
            "wxcos_names",
            label="component",
        )

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[WaveX]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        from jaxpint._build_context import epoch_or_pepoch

        wx_indices = ctx.par.params.prefix_indices("WXFREQ_")
        if not wx_indices:
            return None
        wxepoch_name = epoch_or_pepoch(ctx.par, "WXEPOCH")
        return cls(
            n_components=len(wx_indices),
            wxepoch_name=wxepoch_name,
            wxfreq_names=tuple(f"WXFREQ_{i:04d}" for i in wx_indices),
            wxsin_names=tuple(f"WXSIN_{i:04d}" for i in wx_indices),
            wxcos_names=tuple(f"WXCOS_{i:04d}" for i in wx_indices),
        )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute WaveX delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data.
        params : ParameterVector
            Timing-model parameters.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        array, shape (n_toas,)
            WaveX delay in seconds.
        """
        epoch = params.epoch_dual(self.wxepoch_name)
        dt_days = (toa_data.tdb - epoch).total - delay / SECS_PER_DAY

        freqs = params.param_values(self.wxfreq_names)
        sins = params.param_values(self.wxsin_names)
        coses = params.param_values(self.wxcos_names)

        return fourier_sum(dt_days, freqs, sins, coses)
