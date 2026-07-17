"""Piecewise chromatic measure delay component (ChromaticCMX).

The chromatic measure is modelled as piecewise-constant within user-defined MJD bins:

    CM(t) = Σ CMX_i   for each bin i where CMXR1_i <= t <= CMXR2_i

and the delay for each TOA is:

    delay = CM(t) * K_DM * freq^(-alpha)

where freq is in MHz and alpha = TNCHROMIDX.

"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import ChromaticDelayComponent, ParamDecl
from jaxpint.types import TOAData, ParameterVector


class ChromaticCMX(ChromaticDelayComponent):
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
    # tnchromidx_name inherited from ChromaticDelayComponent (kw_only).

    def __check_init__(self):
        self.check_name_tuples(
            "n_bins", "cmx_names", "cmxr1_names", "cmxr2_names", label="bin"
        )

    def compute_cm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Piecewise-constant chromatic measure ``CM(t)`` over CMX bins.

        The base ``__call__`` applies ``· K_DM · freq^(-TNCHROMIDX)`` to give
        the delay in seconds.
        """
        toa_mjd = toa_data.mjd.total

        cm = jnp.zeros(toa_data.n_toas)

        for i in range(self.n_bins):
            r1 = params.epoch_dual(self.cmxr1_names[i]).total
            r2 = params.epoch_dual(self.cmxr2_names[i]).total

            in_bin = (toa_mjd >= r1) & (toa_mjd <= r2)
            cmx_val = params.param_value(self.cmx_names[i])
            cm = cm + jnp.where(in_bin, cmx_val, 0.0)

        return cm
