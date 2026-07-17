"""Shared EFAC/EQUAD scaling for white-noise components."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import TOAData, ParameterVector


def apply_efac_equad(
    sigma_raw: Float[Array, " n_toas"],
    toa_data: TOAData,
    params: ParameterVector,
    efac_names: tuple[str, ...],
    equad_names: tuple[str, ...],
) -> Float[Array, " n_toas"]:
    """Scale raw per-TOA uncertainties: EQUAD in quadrature, then EFAC.

    Matches PINT's convention (add EQUAD² first, multiply by EFAC after).  Each
    name is applied only where its flag mask is set.  Shared by
    :meth:`~jaxpint.noise.white.ScaleToaError.scaled_sigma` (TOA errors, seconds)
    and :meth:`~jaxpint.noise.dm_white.ScaleDmError.scaled_dm_sigma`
    (DM errors, pc/cm³) — same algebra, different base array and parameter names.

    Parameters
    ----------
    sigma_raw : (n_toas,)
        Base uncertainty (``toa_data.error`` or ``toa_data.dm_errors``).
    toa_data, params
        Provide the flag masks and parameter values.
    efac_names, equad_names
        The multiplicative / in-quadrature parameter names to apply.
    """
    sigma_sq = sigma_raw**2
    for equad_name in equad_names:
        mask = toa_data.flag_mask(equad_name)
        equad_val = params.param_value(equad_name)
        sigma_sq = jnp.where(mask, sigma_sq + equad_val**2, sigma_sq)

    sigma = jnp.sqrt(sigma_sq)

    for efac_name in efac_names:
        mask = toa_data.flag_mask(efac_name)
        efac_val = params.param_value(efac_name)
        sigma = jnp.where(mask, sigma * efac_val, sigma)

    return sigma
