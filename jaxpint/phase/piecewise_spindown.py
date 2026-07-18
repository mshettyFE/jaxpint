"""Piecewise spindown phase component.

The phase is modelled as a Taylor expansion within user-defined time bins:

    phase(t) = Σ_n [ PWPH_n + PWF0_n*dt + PWF1_n*dt^2/2! + PWF2_n*dt^3/3! ]
               for t in [PWSTART_n, PWSTOP_n)  where dt = t - PWEP_n

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ParamDecl, PhaseComponent
from jaxpint.constants import SECS_PER_DAY
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import taylor_horner

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(
    component=Component.PIECEWISE_SPINDOWN, pint_names=("PiecewiseSpindown",)
)
class PiecewiseSpindown(PhaseComponent):
    """Piecewise Taylor-expansion spindown model.

    Parameters
    ----------
    n_pieces : int
        Number of piecewise segments.
    pwstart_names : tuple[str, ...]
        Names of segment start epoch parameters (MJD).
    pwstop_names : tuple[str, ...]
        Names of segment stop epoch parameters (MJD).
    pwep_names : tuple[str, ...]
        Names of segment reference epoch parameters (MJD).
    pwph_names : tuple[str, ...]
        Names of segment phase offset parameters (dimensionless cycles).
    pwf0_names : tuple[str, ...]
        Names of segment frequency parameters (Hz).
    pwf1_names : tuple[str, ...]
        Names of segment frequency derivative parameters (Hz/s).
    pwf2_names : tuple[str, ...]
        Names of segment second derivative parameters (Hz/s^2).

    Raises
    ------
    ValueError
        If ``n_pieces`` is less than 1.
    ValueError
        If the length of any segment parameter name tuple does not match
        ``n_pieces``.
    """

    PARAMS = (
        ParamDecl("PWEP_1", kind="mjd", prefix="PWEP_"),
        ParamDecl("PWSTART_1", kind="mjd", prefix="PWSTART_"),
        ParamDecl("PWSTOP_1", kind="mjd", prefix="PWSTOP_"),
        ParamDecl("PWPH_1", prefix="PWPH_"),
        ParamDecl("PWF0_1", prefix="PWF0_"),
        ParamDecl("PWF1_1", prefix="PWF1_"),
        ParamDecl("PWF2_1", prefix="PWF2_"),
    )

    n_pieces: int = eqx.field(static=True)
    pwstart_names: tuple[str, ...] = eqx.field(static=True)
    pwstop_names: tuple[str, ...] = eqx.field(static=True)
    pwep_names: tuple[str, ...] = eqx.field(static=True)
    pwph_names: tuple[str, ...] = eqx.field(static=True)
    pwf0_names: tuple[str, ...] = eqx.field(static=True)
    pwf1_names: tuple[str, ...] = eqx.field(static=True)
    pwf2_names: tuple[str, ...] = eqx.field(static=True)

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[PiecewiseSpindown]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        pw_indices = ctx.par.params.prefix_indices("PWEP_")
        if not pw_indices:
            return None
        return cls(
            n_pieces=len(pw_indices),
            pwstart_names=tuple(f"PWSTART_{i}" for i in pw_indices),
            pwstop_names=tuple(f"PWSTOP_{i}" for i in pw_indices),
            pwep_names=tuple(f"PWEP_{i}" for i in pw_indices),
            pwph_names=tuple(f"PWPH_{i}" for i in pw_indices),
            pwf0_names=tuple(f"PWF0_{i}" for i in pw_indices),
            pwf1_names=tuple(f"PWF1_{i}" for i in pw_indices),
            pwf2_names=tuple(f"PWF2_{i}" for i in pw_indices),
        )

    def __check_init__(self):
        self.check_name_tuples(
            "n_pieces",
            "pwstart_names",
            "pwstop_names",
            "pwep_names",
            "pwph_names",
            "pwf0_names",
            "pwf1_names",
            "pwf2_names",
            label="piece",
        )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> DualFloat:
        """Compute piecewise spindown phase contribution.

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
        DualFloat
            Phase contribution in cycles (int + frac split).
        """
        toa_tdb = toa_data.tdb.total
        phase = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_pieces):
            start = params.epoch_dual(self.pwstart_names[i]).total
            stop = params.epoch_dual(self.pwstop_names[i]).total
            affected = (toa_tdb >= start) & (toa_tdb < stop)

            # Time since segment epoch (DualFloat precision)
            ep = params.epoch_dual(self.pwep_names[i])
            dt = (toa_data.tdb - ep).total * SECS_PER_DAY - delay

            pwph = params.param_value(self.pwph_names[i])
            pwf0 = params.param_value(self.pwf0_names[i])
            pwf1 = params.param_value(self.pwf1_names[i])
            pwf2 = params.param_value(self.pwf2_names[i])
            coeffs = jnp.array([pwph, pwf0, pwf1, pwf2])

            piece_phase = taylor_horner(dt, coeffs)
            phase = phase + jnp.where(affected, piece_phase, 0.0)

        return DualFloat.from_cycles(jnp.zeros(toa_data.n_toas), phase)
