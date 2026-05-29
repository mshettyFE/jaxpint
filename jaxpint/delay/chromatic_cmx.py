"""Piecewise chromatic measure delay component (ChromaticCMX).

Ports PINT's ``ChromaticCMX`` class as a pure Equinox module.  The chromatic
measure is modelled as piecewise-constant within user-defined MJD bins:

    CM(t) = Σ CMX_i   for each bin i where CMXR1_i <= t <= CMXR2_i

and the delay for each TOA is:

    delay = CM(t) * K_DM * freq^(-alpha)

where freq is in MHz and alpha = TNCHROMIDX.

All derivatives are handled by ``jax.jacobian`` through ``__call__``.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import DelayComponent, ParamDecl
from jaxpint.constants import DMCONST
from jaxpint.dual_float import DualFloat
from jaxpint.types import TOAData, ParameterVector


class ChromaticCMX(DelayComponent):
    """Piecewise-constant chromatic measure delay (CMX model).

    Parameters
    ----------
    n_bins : int
        Number of CMX bins.
    cmx_names : tuple[str, ...]
        Names of CMX value parameters, e.g. ``("CMX_0001", "CMX_0002")``.
    cmxr1_names : tuple[str, ...]
        Names of bin-start MJD epoch parameters.
    cmxr2_names : tuple[str, ...]
        Names of bin-end MJD epoch parameters.
    tnchromidx_name : str
        Name of the chromatic index parameter (default ``"TNCHROMIDX"``).

    Raises
    ------
    ValueError
        If ``n_bins`` is less than 1.
    ValueError
        If the length of ``cmx_names``, ``cmxr1_names``, or ``cmxr2_names``
        does not match ``n_bins``.
    """

    PARAMS = (
        ParamDecl("CMX_0001", prefix="CMX_", frozen_default=False),
        ParamDecl("CMXR1_0001", kind="mjd", prefix="CMXR1_"),
        ParamDecl("CMXR2_0001", kind="mjd", prefix="CMXR2_"),
        ParamDecl("TNCHROMIDX"),
    )

    n_bins: int = eqx.field(static=True)
    cmx_names: tuple[str, ...] = eqx.field(static=True)
    cmxr1_names: tuple[str, ...] = eqx.field(static=True)
    cmxr2_names: tuple[str, ...] = eqx.field(static=True)
    tnchromidx_name: str = eqx.field(static=True, default="TNCHROMIDX")

    def __check_init__(self):
        if self.n_bins < 1:
            raise ValueError("ChromaticCMX requires at least one bin")
        for attr in ("cmx_names", "cmxr1_names", "cmxr2_names"):
            if len(getattr(self, attr)) != self.n_bins:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_bins ({self.n_bins})"
                )

    def __call__(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute piecewise chromatic delay contribution.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (MJD times, frequencies).
        params : ParameterVector
            Timing-model parameters containing CMX, CMXR1, CMXR2, TNCHROMIDX.
        delay : array, shape (n_toas,)
            Accumulated signal delay from prior components in seconds.

        Returns
        -------
        array, shape (n_toas,)
            Chromatic delay in seconds.
        """
        toa_mjd = toa_data.mjd.total

        cm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            r1 = params.epoch_dual(self.cmxr1_names[i]).total
            r2 = params.epoch_dual(self.cmxr2_names[i]).total

            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)
            cmx_val = params.param_value(self.cmx_names[i])
            cm = cm + jnp.where(in_bin, cmx_val, 0.0)

        alpha = params.param_value(self.tnchromidx_name)
        return cm * DMCONST * toa_data.freq ** (-alpha)
