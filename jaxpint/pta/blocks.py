"""Low-level noise-weighted inner-product primitives."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.types import GlobalParams, ParameterVector
from jaxpint.utils import SMWhitener

if TYPE_CHECKING:
    from jaxpint.pta.likelihood import PTAConfig

__all__ = [
    "WoodburyBlock",
    "pulsar_woodbury_blocks",
    "whitened_quadratic",
]

# (Ndiag, U, Phi, whitener) — the per-pulsar Woodbury covariance factors.
WoodburyBlock = tuple[
    Float[Array, " n_toas"],
    Float[Array, "n_toas n_basis"],
    Float[Array, " n_basis"],
    SMWhitener | None,
]


def pulsar_woodbury_blocks(
    config: PTAConfig,
    global_params: GlobalParams,
    pulsar_params_p: ParameterVector,
    p: int,
) -> WoodburyBlock:
    r"""``(Ndiag, U, Phi, whitener)`` for pulsar ``p``, ready for solves."""
    from jaxpint.pta.likelihood import _collect_injector_ext_cov
    from jaxpint.utils import concat_woodbury_blocks

    toa_data_p = config.toa_data_list[p]
    noise_model_p = config.noise_models[p]
    Ndiag, U_noise, Phi_noise = noise_model_p.covariance(toa_data_p, pulsar_params_p)
    ext_cov = _collect_injector_ext_cov(
        p, toa_data_p, pulsar_params_p, global_params, config.signal_injectors
    )
    woodbury = concat_woodbury_blocks((U_noise, Phi_noise), ext_cov)
    assert woodbury is not None  # the noise block is always present
    U, Phi = woodbury
    whitener = (
        noise_model_p.ecorr_kernel.ops(Ndiag, pulsar_params_p)
        if noise_model_p.ecorr_kernel is not None
        else None
    )
    if whitener is not None:
        U = whitener.whiten(U)
        Ndiag = jnp.ones_like(Ndiag)
    return Ndiag, U, Phi, whitener


def whitened_quadratic(
    block: WoodburyBlock,
    r: Float[Array, " n_toas"],
) -> Float[Array, ""]:
    r"""``1/2 r^T C^-1 r`` for one pulsar's block (square-root Woodbury form)."""
    from jaxpint.utils import woodbury_dot_qr

    Ndiag, U, Phi, whitener = block
    if whitener is not None:
        r = whitener.whiten(r)
    q, _logdet = woodbury_dot_qr(Ndiag, U, Phi, r, r)
    return 0.5 * q
