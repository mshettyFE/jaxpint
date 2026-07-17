"""Fourier-basis chromatic noise delay component (CMWaveX).

The chromatic measure (CM) is modelled as a Fourier sum and converted to delay via
chromatic dispersion:

    CM(t) = Σ_i (CMWXSIN_i * sin(2π * CMWXFREQ_i * (t - CMWXEPOCH))
               + CMWXCOS_i * cos(2π * CMWXFREQ_i * (t - CMWXEPOCH)))

    delay = CM(t) * K_DM * freq^(-alpha)

where alpha = TNCHROMIDX.

"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import ChromaticDelayComponent, ParamDecl
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import fourier_sum


class CMWaveX(ChromaticDelayComponent):
    """Fourier-basis chromatic noise (CMWaveX).

    Parameters
    ----------
    n_components : int
        Number of Fourier components.
    cmwxepoch_name : str
        Name of the reference epoch parameter.
    cmwxfreq_names : tuple[str, ...]
        Names of the frequency parameters (1/day).
    cmwxsin_names : tuple[str, ...]
        Names of the sine amplitude parameters (cmu).
    cmwxcos_names : tuple[str, ...]
        Names of the cosine amplitude parameters (cmu).
    tnchromidx_name : str
        Name of the chromatic index parameter.

    Raises
    ------
    ValueError
        If ``n_components`` is less than 1.
    ValueError
        If the length of ``cmwxfreq_names``, ``cmwxsin_names``, or
        ``cmwxcos_names`` does not match ``n_components``.
    """

    PARAMS = (
        ParamDecl("CMWXFREQ_0001", prefix="CMWXFREQ_"),
        ParamDecl("CMWXSIN_0001", prefix="CMWXSIN_", frozen_default=False),
        ParamDecl("CMWXCOS_0001", prefix="CMWXCOS_", frozen_default=False),
        ParamDecl("CMWXEPOCH", kind="mjd"),
        ParamDecl("TNCHROMIDX"),
    )

    n_components: int = eqx.field(static=True)
    cmwxfreq_names: tuple[str, ...] = eqx.field(static=True)
    cmwxsin_names: tuple[str, ...] = eqx.field(static=True)
    cmwxcos_names: tuple[str, ...] = eqx.field(static=True)
    cmwxepoch_name: str = eqx.field(static=True, default="CMWXEPOCH")
    # tnchromidx_name inherited from ChromaticDelayComponent (kw_only).

    def __check_init__(self):
        if self.n_components < 1:
            raise ValueError("CMWaveX requires at least one component")
        for attr in ("cmwxfreq_names", "cmwxsin_names", "cmwxcos_names"):
            if len(getattr(self, attr)) != self.n_components:
                raise ValueError(
                    f"Length of {attr} ({len(getattr(self, attr))}) "
                    f"does not match n_components ({self.n_components})"
                )

    def compute_cm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Chromatic measure ``CM(t)`` as a Fourier series about CMWXEPOCH.

        The base ``__call__`` applies ``· K_DM · freq^(-TNCHROMIDX)`` to give
        the delay in seconds.
        """
        epoch = params.epoch_dual(self.cmwxepoch_name)
        dt_days = (toa_data.tdb - epoch).total

        freqs = params.param_values(self.cmwxfreq_names)
        sins = params.param_values(self.cmwxsin_names)
        coses = params.param_values(self.cmwxcos_names)

        return fourier_sum(dt_days, freqs, sins, coses)
