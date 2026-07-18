"""Piecewise dispersion delay component (DMX).

The dispersion measure is modelled as piecewise-constant within user-defined MJD bins:

    DM(t) = Σ DMX_i   for each bin i where DMXR1_i <= t <= DMXR2_i

and the delay for each TOA is:

    delay = DM(t) * K_DM / freq^2

where freq is in MHz and K_DM = 1 / 2.41e-4 (MHz^2 s cm^3 / pc).

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent, ParamDecl
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(component=Component.DISPERSION_DMX, pint_names=("DispersionDMX",))
class DispersionDMX(DispersionDelayComponent):
    """Piecewise-constant DM dispersion delay (DMX model).

    Parameters
    ----------
    n_bins : int
        Number of DMX bins.
    dmx_names : tuple[str, ...]
        Names of DMX value parameters, e.g. ``("DMX_0001", "DMX_0002")``.
    dmxr1_names : tuple[str, ...]
        Names of bin-start MJD epoch parameters, e.g. ``("DMXR1_0001", ...)``.
    dmxr2_names : tuple[str, ...]
        Names of bin-end MJD epoch parameters, e.g. ``("DMXR2_0001", ...)``.

    Raises
    ------
    ValueError
        If ``n_bins`` is less than 1.
    ValueError
        If the length of ``dmx_names``, ``dmxr1_names``, or ``dmxr2_names``
        does not match ``n_bins``.
    """

    PARAMS = (
        ParamDecl("DMX_0001", prefix="DMX_", frozen_default=False),
        ParamDecl("DMXR1_0001", kind="mjd", prefix="DMXR1_"),
        ParamDecl("DMXR2_0001", kind="mjd", prefix="DMXR2_"),
    )

    n_bins: int = eqx.field(static=True)
    dmx_names: tuple[str, ...] = eqx.field(static=True)
    dmxr1_names: tuple[str, ...] = eqx.field(static=True)
    dmxr2_names: tuple[str, ...] = eqx.field(static=True)

    def __check_init__(self):
        self.check_name_tuples(
            "n_bins", "dmx_names", "dmxr1_names", "dmxr2_names", label="bin"
        )

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[DispersionDMX]":
        """Construct from a parsed model (co-located with the physics it builds).

        One bin per ``DMX_NNNN`` index present; ``None`` when there are none.
        """
        idx = ctx.par.params.prefix_indices("DMX_")
        if not idx:
            return None
        return cls(
            n_bins=len(idx),
            dmx_names=tuple(f"DMX_{i:04d}" for i in idx),
            dmxr1_names=tuple(f"DMXR1_{i:04d}" for i in idx),
            dmxr2_names=tuple(f"DMXR2_{i:04d}" for i in idx),
        )

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute piecewise-constant DM from DMX bins.

        Each TOA receives the DMX value of the bin it falls within.
        TOAs outside all bins receive zero DM contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (MJD times for bin assignment).
        params : ParameterVector
            Timing-model parameters containing DMX, DMXR1, DMXR2 values.
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this method).

        Returns
        -------
        array, shape (n_toas,)
            Piecewise DM in pc cm^-3 at each TOA.
        """
        toa_mjd = toa_data.mjd.total

        dm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            r1 = params.epoch_dual(self.dmxr1_names[i]).total
            r2 = params.epoch_dual(self.dmxr2_names[i]).total

            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)
            dmx_val = params.param_value(self.dmx_names[i])
            dm = dm + jnp.where(in_bin, dmx_val, 0.0)

        return dm
