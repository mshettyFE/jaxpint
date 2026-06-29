"""Fisher matrix utilities for PTA likelihood.

Provides flatten/unflatten helpers to pack all differentiable parameters
into a single flat array for ``jax.hessian``, and a convenience wrapper
for computing the Fisher information matrix.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

try:
    from beartype import beartype
except ModuleNotFoundError:  # dev-only extra; without it jaxtyped is a no-op
    beartype = None
from jaxtyping import Array, Float, jaxtyped

from jaxpint.types import ParameterVector
from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import PTAConfig, pta_logL


def flatten_params(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
) -> Float[Array, " n_params_total"]:
    """Pack all differentiable parameters into a single flat array.

    Layout: ``[global_params.values | pp[0].values | pp[1].values | ...]``

    Parameters
    ----------
    global_params : GlobalParams
        Shared PTA parameters.
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.

    Returns
    -------
    flat : (n_params_total,) array
        Concatenated parameter values, where
        ``n_params_total = n_global + sum(n_pp_i)``.
    """
    return jnp.concatenate([global_params.values] + [pp.values for pp in pulsar_params])


def unflatten_params(
    flat: Float[Array, " n_params_total"],
    global_template: GlobalParams,
    pulsar_templates: tuple[ParameterVector, ...],
) -> tuple[GlobalParams, tuple[ParameterVector, ...]]:
    """Unpack a flat array back into structured parameter objects.

    *Templates* carry the static metadata (names, frozen mask, units, etc.).
    Only ``.values`` is replaced from slices of the flat array; everything
    else is preserved from the template.

    Layout must match :func:`flatten_params`::

        flat[0 : n_global]               -> GlobalParams.values
        flat[n_global : n_global + n_pp0] -> pulsar_params[0].values
        ...

    Parameters
    ----------
    flat : (n_params_total,) array
        Flat parameter vector.
    global_template : GlobalParams
        Template with correct static metadata for the global params.
    pulsar_templates : tuple of ParameterVector
        Templates with correct static metadata for each pulsar.

    Returns
    -------
    global_params : GlobalParams
    pulsar_params : tuple of ParameterVector
    """
    offset = 0

    # Reconstruct GlobalParams (replace the values leaf, keep names/index).
    n_global = global_template.n_params
    gp = global_template.with_values(flat[offset : offset + n_global])
    offset += n_global

    # Reconstruct each ParameterVector
    pp_list = []
    for template in pulsar_templates:
        n = template.n_params
        new_pp = template.with_values(flat[offset : offset + n])
        pp_list.append(new_pp)
        offset += n

    return gp, tuple(pp_list)


@jaxtyped(typechecker=beartype)
def fisher_matrix(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: PTAConfig,
) -> Float[Array, "n_params_total n_params_total"]:
    """Compute the Fisher information matrix via ``jax.hessian``.

    Parameters
    ----------
    global_params : GlobalParams
        Current global parameter values (evaluation point).
    pulsar_params : tuple of ParameterVector
        Current per-pulsar parameter values (evaluation point).
    config : PTAConfig
        Static PTA configuration.

    Returns
    -------
    fisher : (n_params_total, n_params_total) array
        Fisher matrix, where ``n_params_total = n_global + sum(n_pp_i)``.
    """
    flat = flatten_params(global_params, pulsar_params)

    def logL_flat(flat_params):
        gp, pp = unflatten_params(flat_params, global_params, pulsar_params)
        return pta_logL(gp, pp, config)

    return -jax.hessian(logL_flat)(flat)
