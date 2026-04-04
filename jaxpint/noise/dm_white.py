"""White noise model for wideband DM uncertainties: DMEFAC/DMEQUAD.

::

    σ_dm_eff = DMEFAC × √(σ_dm_raw² + DMEQUAD²)

Each parameter applies to a subset of TOAs identified by a boolean
mask (pre-computed by the bridge layer and stored in ``TOAData.flag_masks``).
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector


class ScaleDmError(eqx.Module):
    """White noise scaling of wideband DM uncertainties (DMEFAC/DMEQUAD).

    Parameters
    ----------
    dmefac_names : tuple of str
        Parameter names for DMEFAC instances.
    dmequad_names : tuple of str
        Parameter names for DMEQUAD instances.
        Values must be in pc/cm³.
    """

    dmefac_names: tuple[str, ...] = eqx.field(static=True)
    dmequad_names: tuple[str, ...] = eqx.field(static=True)

    def scaled_dm_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute noise-scaled DM uncertainties in pc/cm³."""
        sigma_sq = toa_data.dm_errors ** 2

        for dmequad_name in self.dmequad_names:
            mask = toa_data.flag_masks[dmequad_name]
            dmequad_val = params.param_value(dmequad_name)
            sigma_sq = jnp.where(mask, sigma_sq + dmequad_val ** 2, sigma_sq)

        sigma = jnp.sqrt(sigma_sq)

        for dmefac_name in self.dmefac_names:
            mask = toa_data.flag_masks[dmefac_name]
            dmefac_val = params.param_value(dmefac_name)
            sigma = jnp.where(mask, sigma * dmefac_val, sigma)

        return sigma
