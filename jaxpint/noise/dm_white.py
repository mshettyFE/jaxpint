"""White noise model for wideband DM uncertainties: DMEFAC/DMEQUAD.

::

    σ_dm_eff = DMEFAC × √(σ_dm_raw² + DMEQUAD²)

Each parameter applies to a subset of TOAs identified by a boolean
mask (pre-computed by the bridge layer and stored in ``TOAData.flag_masks``).
"""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent, ParamDecl
from jaxpint.noise._white_common import apply_efac_equad
from jaxpint.types import TOAData, ParameterVector


class ScaleDmError(NoiseComponent):
    """White noise scaling of wideband DM uncertainties (DMEFAC/DMEQUAD).

    Parameters
    ----------
    dmefac_names : tuple of str
        Parameter names for DMEFAC instances.
    dmequad_names : tuple of str
        Parameter names for DMEQUAD instances.
        Values must be in pc/cm³.
    """

    PARAMS = (
        ParamDecl("DMEFAC1", kind="mask", aliases=("DMEFAC",), prefix="DMEFAC"),
        ParamDecl("DMEQUAD1", kind="mask", aliases=("DMEQUAD",), prefix="DMEQUAD"),
    )

    dmefac_names: tuple[str, ...] = eqx.field(static=True)
    dmequad_names: tuple[str, ...] = eqx.field(static=True)

    def scaled_dm_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute noise-scaled DM uncertainties in pc/cm³.

        Applies DMEQUAD in quadrature first, then multiplies by DMEFAC,
        matching PINT's ``ScaleDmError`` convention.

        Parameters
        ----------
        toa_data : TOAData
            Must contain ``dm_errors`` (pc/cm³) and ``flag_masks`` with
            entries for every name in ``dmefac_names`` and ``dmequad_names``.
        params : ParameterVector
            Must contain values for all DMEFAC/DMEQUAD parameters.

        Returns
        -------
        sigma_scaled : (n_toas,)
            Scaled DM uncertainties in pc/cm³.
        """
        assert toa_data.dm_errors is not None  # DM white noise requires DM data
        return apply_efac_equad(
            toa_data.dm_errors, toa_data, params, self.dmefac_names, self.dmequad_names
        )
