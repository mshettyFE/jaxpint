"""White noise model: EFAC/EQUAD scaling of TOA uncertainties.

::

    σ_eff = EFAC × √(σ_raw² + EQUAD²)

Each parameter applies to a subset of TOAs identified by a boolean
mask (pre-computed by the bridge layer and stored in ``TOAData.flag_masks``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.components import NoiseComponent
from jaxpint.types import TOAData, ParameterVector


class ScaleToaError(NoiseComponent):
    """White noise model: EFAC/EQUAD scaling of TOA uncertainties.

    Stores the names of EFAC and EQUAD parameters (static metadata).
    The boolean masks selecting which TOAs each parameter applies to
    live in ``TOAData.flag_masks`` (extracted by the bridge).

    Parameters
    ----------
    efac_names : tuple of str
        Parameter names for EFAC instances (e.g. ``("EFAC1", "EFAC2")``).
    equad_names : tuple of str
        Parameter names for EQUAD instances (e.g. ``("EQUAD1", "EQUAD2")``).
        Values must be in **seconds** (the bridge converts from PINT's
        native microseconds).
    """

    efac_names: tuple[str, ...] = eqx.field(static=True)
    equad_names: tuple[str, ...] = eqx.field(static=True)

    def scaled_sigma(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> Float[Array, " n_toas"]:
        """Compute noise-scaled TOA uncertainties.

        Applies EQUAD in quadrature first, then multiplies by EFAC,
        matching PINT's ``ScaleToaError.scale_toa_sigma()`` convention.

        Parameters
        ----------
        toa_data : TOAData
            Must contain ``error`` (seconds) and ``flag_masks`` with
            entries for every name in ``efac_names`` and ``equad_names``.
        params : ParameterVector
            Must contain values for all EFAC/EQUAD parameters.

        Returns
        -------
        sigma_scaled : (n_toas,)
            Scaled uncertainties in seconds.
        """
        sigma_sq = toa_data.error ** 2

        # Step 1: add EQUAD in quadrature (per mask)
        for equad_name in self.equad_names:
            mask = toa_data.flag_masks[equad_name]
            equad_val = params.param_value(equad_name)
            sigma_sq = jnp.where(mask, sigma_sq + equad_val ** 2, sigma_sq)

        sigma = jnp.sqrt(sigma_sq)

        # Step 2: multiply by EFAC (per mask)
        for efac_name in self.efac_names:
            mask = toa_data.flag_masks[efac_name]
            efac_val = params.param_value(efac_name)
            sigma = jnp.where(mask, sigma * efac_val, sigma)

        return sigma

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[Float[Array, " n_toas"], None, None]:
        sigma = self.scaled_sigma(toa_data, params)
        return sigma ** 2, None, None

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        sigma = self.scaled_sigma(toa_data, params)
        return sigma * jax.random.normal(key, shape=(toa_data.n_toas,))
