"""Fourier-basis DM noise delay component (DMWaveX).

The DM is modelled as a Fourier sum and converted to delay via DM dispersion:

    DM(t) = Σ_i (DMWXSIN_i * sin(2π * DMWXFREQ_i * (t - DMWXEPOCH))
               + DMWXCOS_i * cos(2π * DMWXFREQ_i * (t - DMWXEPOCH)))

    delay = DM(t) * K_DM / freq^2
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import DispersionDelayComponent, ParamDecl
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import fourier_sum


class DMWaveX(DispersionDelayComponent):
    """Fourier-basis DM noise (DMWaveX).

    Parameters
    ----------
    n_components : int
        Number of Fourier components.
    dmwxepoch_name : str
        Name of the reference epoch parameter.
    dmwxfreq_names : tuple[str, ...]
        Names of the frequency parameters (1/day).
    dmwxsin_names : tuple[str, ...]
        Names of the sine amplitude parameters (pc cm^-3).
    dmwxcos_names : tuple[str, ...]
        Names of the cosine amplitude parameters (pc cm^-3).

    Raises
    ------
    ValueError
        If ``n_components`` is less than 1.
    ValueError
        If the length of ``dmwxfreq_names``, ``dmwxsin_names``, or
        ``dmwxcos_names`` does not match ``n_components``.
    """

    PARAMS = (
        ParamDecl("DMWXFREQ_0001", prefix="DMWXFREQ_"),
        ParamDecl("DMWXSIN_0001", prefix="DMWXSIN_", frozen_default=False),
        ParamDecl("DMWXCOS_0001", prefix="DMWXCOS_", frozen_default=False),
        ParamDecl("DMWXEPOCH", kind="mjd"),
    )

    n_components: int = eqx.field(static=True)
    dmwxfreq_names: tuple[str, ...] = eqx.field(static=True)
    dmwxsin_names: tuple[str, ...] = eqx.field(static=True)
    dmwxcos_names: tuple[str, ...] = eqx.field(static=True)
    dmwxepoch_name: str = eqx.field(static=True, default="DMWXEPOCH")

    def __check_init__(self):
        self.check_name_tuples(
            "n_components",
            "dmwxfreq_names",
            "dmwxsin_names",
            "dmwxcos_names",
            label="component",
        )

    def compute_dm(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        delay: Float[Array, " n_toas"],
    ) -> Float[Array, " n_toas"]:
        """Compute DM as a Fourier sum about DMWXEPOCH.

        Parameters
        ----------
        toa_data : TOAData
            Pre-extracted TOA data (TDB times for dt from DMWXEPOCH).
        params : ParameterVector
            Timing-model parameters containing DMWXFREQ, DMWXSIN,
            DMWXCOS, and DMWXEPOCH values.
        delay : array, shape (n_toas,)
            Accumulated signal delay in seconds (unused by this method).

        Returns
        -------
        array, shape (n_toas,)
            Fourier-basis DM in pc cm^-3 at each TOA.
        """
        epoch = params.epoch_dual(self.dmwxepoch_name)
        dt_days = (toa_data.tdb - epoch).total

        freqs = params.param_values(self.dmwxfreq_names)
        sins = params.param_values(self.dmwxsin_names)
        coses = params.param_values(self.dmwxcos_names)

        return fourier_sum(dt_days, freqs, sins, coses)

    # __call__ (dm · K_DM / freq²) is inherited from DispersionDelayComponent.
