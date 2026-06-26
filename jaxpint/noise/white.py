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

from jaxpint.components import NoiseComponent, ParamDecl
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

    PARAMS = (
        ParamDecl(
            "EFAC1",
            kind="mask",
            prefix="EFAC",
            aliases=("EFAC", "T2EFAC", "T2EFAC1", "TNEF", "TNEF1"),
            prefix_aliases=("T2EFAC", "TNEF"),
        ),
        ParamDecl(
            "EQUAD1",
            kind="mask",
            unit="us",
            prefix="EQUAD",
            aliases=("EQUAD", "T2EQUAD", "T2EQUAD1"),
            prefix_aliases=("T2EQUAD",),
        ),
    )

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
        sigma_sq = toa_data.error**2

        for equad_name in self.equad_names:
            mask = toa_data.flag_mask(equad_name)
            equad_val = params.param_value(equad_name)
            sigma_sq = jnp.where(mask, sigma_sq + equad_val**2, sigma_sq)

        sigma = jnp.sqrt(sigma_sq)

        for efac_name in self.efac_names:
            mask = toa_data.flag_mask(efac_name)
            efac_val = params.param_value(efac_name)
            sigma = jnp.where(mask, sigma * efac_val, sigma)

        return sigma

    def covariance(
        self,
        toa_data: TOAData,
        params: ParameterVector,
    ) -> tuple[
        Float[Array, " n_toas"],
        Float[Array, "n_toas 0"],
        Float[Array, " 0"],
    ]:
        """Compute the white noise diagonal covariance.

        Returns the EFAC/EQUAD-scaled variance as the diagonal term,
        with empty-shaped basis arrays (white noise is purely diagonal).

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including raw uncertainties and flag masks.
        params : ParameterVector
            Current parameter values for all EFAC/EQUAD parameters.

        Returns
        -------
        Ndiag : (n_toas,)
            Scaled variance for each TOA (seconds squared).
        U : (n_toas, 0)
            Zero-width basis (white noise has no low-rank contribution).
        Phidiag : (0,)
            Zero-length basis weights.
        """
        sigma = self.scaled_sigma(toa_data, params)
        return (
            sigma**2,
            jnp.zeros((toa_data.n_toas, 0)),
            jnp.zeros(0),
        )

    def generate(
        self,
        toa_data: TOAData,
        params: ParameterVector,
        key: jax.Array,
    ) -> Float[Array, " n_toas"]:
        """Draw a random white noise realization.

        Samples independent Gaussian noise scaled by the EFAC/EQUAD-adjusted
        TOA uncertainties.

        Parameters
        ----------
        toa_data : TOAData
            Observed TOA data including raw uncertainties and flag masks.
        params : ParameterVector
            Current parameter values for all EFAC/EQUAD parameters.
        key : jax.Array
            PRNG key for random sampling.

        Returns
        -------
        noise : (n_toas,)
            White noise realization in seconds.
        """
        sigma = self.scaled_sigma(toa_data, params)
        return sigma * jax.random.normal(key, shape=(toa_data.n_toas,))
