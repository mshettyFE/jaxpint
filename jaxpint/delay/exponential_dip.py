"""Exponential dip delay component.

Models chromatic exponential dip events (e.g. profile changes) with a
smooth logistic transition:

    delay_i = -A * (f/fref)^gamma * norm * expfac

where norm ensures the extremum equals A at the peak, and expfac
combines an exponential decay with a smooth logistic onset.

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.par._component_registry import register_component
from jaxpint.par.registry import Component
from jaxpint.types import TOAData, ParameterVector

if TYPE_CHECKING:
    from jaxpint._build_context import BuildContext


@register_component(
    component=Component.EXPONENTIAL_DIP, pint_names=("SimpleExponentialDip",)
)
class ExponentialDip(DelayComponent):
    """Exponential dip delay model.

    Parameters
    ----------
    n_dips : int
        Number of dip events.
    expdipeps_name : str
        Name of the transition timescale parameter (days).
    expdipfref_name : str
        Name of the reference frequency parameter (MHz).
    expdipep_names : tuple[str, ...]
        Names of event epoch parameters (MJD).
    expdipamp_names : tuple[str, ...]
        Names of event amplitude parameters (seconds).
    expdipidx_names : tuple[str, ...]
        Names of chromatic index parameters (dimensionless).
    expdiptau_names : tuple[str, ...]
        Names of decay timescale parameters (days).

    Raises
    ------
    ValueError
        If ``n_dips`` is less than 1.
    ValueError
        If the length of ``expdipep_names``, ``expdipamp_names``,
        ``expdipidx_names``, or ``expdiptau_names`` does not match
        ``n_dips``.
    """

    PARAMS = (
        ParamDecl(
            "EXPDIPEP_1",
            kind="mjd",
            prefix="EXPDIPEP_",
            aliases=("EXPEP_1",),
            prefix_aliases=("EXPEP_",),
        ),
        ParamDecl(
            "EXPDIPAMP_1",
            unit="s",
            prefix="EXPDIPAMP_",
            aliases=("EXPPH_1",),
            prefix_aliases=("EXPPH_",),
        ),
        ParamDecl(
            "EXPDIPIDX_1",
            prefix="EXPDIPIDX_",
            aliases=("EXPINDEX_1",),
            prefix_aliases=("EXPINDEX_",),
        ),
        ParamDecl(
            "EXPDIPTAU_1",
            prefix="EXPDIPTAU_",
            aliases=("EXPTAU_1",),
            prefix_aliases=("EXPTAU_",),
        ),
        ParamDecl("EXPDIPEPS"),
        ParamDecl("EXPDIPFREF"),
    )

    n_dips: int = eqx.field(static=True)
    expdipep_names: tuple[str, ...] = eqx.field(static=True)
    expdipamp_names: tuple[str, ...] = eqx.field(static=True)
    expdipidx_names: tuple[str, ...] = eqx.field(static=True)
    expdiptau_names: tuple[str, ...] = eqx.field(static=True)
    expdipeps_name: str = eqx.field(static=True, default="EXPDIPEPS")
    expdipfref_name: str = eqx.field(static=True, default="EXPDIPFREF")

    @classmethod
    def build(cls, ctx: "BuildContext") -> "Optional[ExponentialDip]":
        """Construct from a parsed model (co-located with the physics it builds)."""
        dip_indices = ctx.par.params.prefix_indices("EXPDIPEPOCH_")
        if not dip_indices:
            dip_indices = ctx.par.params.prefix_indices("EXPDIPEP_")
        if not dip_indices:
            return None
        return cls(
            n_dips=len(dip_indices),
            expdipeps_name="EXPDIPEPS",
            expdipfref_name="EXPDIPFREF",
            expdipep_names=tuple(f"EXPDIPEP_{i}" for i in dip_indices),
            expdipamp_names=tuple(f"EXPDIPAMP_{i}" for i in dip_indices),
            expdipidx_names=tuple(f"EXPDIPIDX_{i}" for i in dip_indices),
            expdiptau_names=tuple(f"EXPDIPTAU_{i}" for i in dip_indices),
        )

    def __check_init__(self):
        self.check_name_tuples(
            "n_dips",
            "expdipep_names",
            "expdipamp_names",
            "expdipidx_names",
            "expdiptau_names",
            label="dip event",
        )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute exponential dip delay contribution.

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
            Exponential dip delay in seconds.
        """
        eps = params.param_value(self.expdipeps_name)  # days
        fref = params.param_value(self.expdipfref_name)  # MHz
        ffac = toa_data.freq / fref

        toa_tdb = toa_data.tdb.total  # MJD (days)

        total = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_dips):
            T = params.epoch_dual(self.expdipep_names[i]).total
            dt = toa_tdb - T  # days

            A = params.param_value(self.expdipamp_names[i])  # seconds
            gamma = params.param_value(self.expdipidx_names[i])  # dimensionless
            tau = params.param_value(self.expdiptau_names[i])  # days

            # Normalization so extremum = A
            norm = (tau / eps) ** (eps / tau) * (tau / (tau - eps)) ** (
                (tau - eps) / tau
            )

            # Exponential factor with smooth logistic transition.
            expfac_pos = jnp.exp(-dt / tau) / (1.0 + jnp.exp(-dt / eps))
            expfac_neg = jnp.exp(dt * (tau - eps) / (tau * eps)) / (
                1.0 + jnp.exp(dt / eps)
            )
            expfac = jnp.where(dt >= 0.0, expfac_pos, expfac_neg)

            total = total + (-A * ffac**gamma * norm * expfac)

        return total
