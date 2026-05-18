"""Batched PTA log-likelihood via vmap + lax.switch.

Provides :class:`BatchedPTAConfig` and :func:`pta_logL_batched` as a
vmapped alternative to the Python-loop in :func:`pta_logL`.

Architecture
------------
Each pulsar gets its own ``lax.switch`` branch that captures its
timing model, noise model, and padded TOAData in the closure.
Only parameter values are vmapped (batched across pulsars).

This produces a single fused XLA trace instead of N separate traces,
enabling GPU batching and reducing JIT overhead.

The original :func:`pta_logL` is unchanged and serves as the
correctness reference.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

from jaxpint.fitters import compute_time_residuals
from jaxpint.model import TimingModel
from jaxpint.noise import NoiseModel
from jaxpint.types import TOAData, ParameterVector
from jaxpint.utils import concat_woodbury_blocks, woodbury_dot

from jaxpint.pta.likelihood import PTAConfig, SignalInjector
from jaxpint.pta.params import GlobalParams
from jaxpint.pta.padding import (
    UniversalParamLayout,
    build_universal_param_layout,
    reindex_param_values,
    make_universal_parameter_vector,
    pad_toa_data,
    pad_noise_model,
)


# ---------------------------------------------------------------------------
# Branch construction
# ---------------------------------------------------------------------------


def _make_branch(
    p: int,
    timing_model: TimingModel,
    noise_model: NoiseModel,
    toa_data: TOAData,
    signal_injectors: tuple[SignalInjector, ...],
    n_max: int,
    n_actual: int,
    layout: UniversalParamLayout,
):
    """Create a lax.switch branch for pulsar *p*.

    The branch captures:
    - p (int): pulsar index for SignalInjector.delay/covariance calls
    - timing_model, noise_model: this pulsar's models (padded)
    - toa_data: padded to n_max (compile-time constant)
    - signal_injectors: shared across all pulsars
    - n_actual: real TOA count (compile-time constant)
    - layout: universal parameter layout

    Arguments at call time:
    - param_values: (n_params_universal,) — this pulsar's parameter values
    - global_params: GlobalParams — shared across all pulsars
    """

    def branch(param_values, global_params):
        # Reconstruct ParameterVector with universal layout
        params = make_universal_parameter_vector(param_values, layout)

        # 1. Residuals
        r = compute_time_residuals(timing_model, toa_data, params)

        # 2. External delay from signal injectors
        ext_delay = jnp.zeros(n_max, dtype=jnp.float64)
        for inj in signal_injectors:
            d = inj.delay(p, toa_data, params, global_params)
            if d is not None:
                ext_delay = ext_delay + d
        r = r - ext_delay

        # 3. Mask padded TOAs
        mask = jnp.arange(n_max) < n_actual
        r = jnp.where(mask, r, 0.0)

        # 4. Noise covariance
        Ndiag, U_noise, Phi_noise = noise_model.covariance(toa_data, params)
        Ndiag = jnp.where(mask, Ndiag, 1.0)

        # 5. Concatenate noise (U, Φ) with all per-injector covariance blocks.
        ext_covs = [
            inj.covariance(p, toa_data, params, global_params)
            for inj in signal_injectors
        ]
        U, Phi = concat_woodbury_blocks((U_noise, Phi_noise), *ext_covs)

        # 6. Woodbury log-likelihood
        rCr, logdetC = woodbury_dot(Ndiag, U, Phi, r, r)
        return (
            -0.5 * rCr
            - 0.5 * logdetC
            - 0.5 * n_actual * jnp.log(2 * jnp.pi)
        )

    return branch


# ---------------------------------------------------------------------------
# BatchedPTAConfig
# ---------------------------------------------------------------------------


class BatchedPTAConfig(eqx.Module):
    """Pre-computed batched configuration for vmapped PTA likelihood.

    Constructed from the same inputs as :class:`PTAConfig`, with the
    addition of ``pulsar_params`` (needed to build the universal layout).

    All padding, branch construction, and layout computation happens
    at construction time (once, in Python/numpy).
    """

    # Original config (for reference / fallback to loop path)
    config: PTAConfig = eqx.field(static=True)

    # Universal parameter layout
    layout: UniversalParamLayout = eqx.field(static=True)

    # Pre-computed for vmap
    n_max: int = eqx.field(static=True)
    n_pulsars: int = eqx.field(static=True)
    _branches: list = eqx.field(static=True)  # list of branch functions
    _branch_indices: Array  # (n_psr,) int32

    def __init__(
        self,
        toa_data_list: tuple[TOAData, ...],
        timing_models: tuple[TimingModel, ...],
        noise_models: tuple[NoiseModel, ...],
        signal_injectors: tuple[SignalInjector, ...],
        pulsar_params: tuple[ParameterVector, ...],
    ):
        # Store original config
        self.config = PTAConfig(
            toa_data_list=toa_data_list,
            timing_models=timing_models,
            noise_models=noise_models,
            signal_injectors=signal_injectors,
        )

        n_psr = len(toa_data_list)
        self.n_pulsars = n_psr

        # 1. Build universal parameter layout
        self.layout = build_universal_param_layout(pulsar_params)

        # 2. Compute n_max
        self.n_max = max(td.n_toas for td in toa_data_list)

        # 3. Pad TOAData and noise models
        padded_toa_data = [pad_toa_data(td, self.n_max) for td in toa_data_list]
        padded_noise_models = [
            pad_noise_model(nm, self.n_max) for nm in noise_models
        ]

        # 4. Build per-pulsar branches
        self._branches = [
            _make_branch(
                p=p,
                timing_model=timing_models[p],
                noise_model=padded_noise_models[p],
                toa_data=padded_toa_data[p],
                signal_injectors=signal_injectors,
                n_max=self.n_max,
                n_actual=toa_data_list[p].n_toas,
                layout=self.layout,
            )
            for p in range(n_psr)
        ]

        # 5. Branch indices: identity mapping (each pulsar → its own branch)
        self._branch_indices = jnp.arange(n_psr, dtype=jnp.int32)


# ---------------------------------------------------------------------------
# Batched log-likelihood
# ---------------------------------------------------------------------------


def pta_logL_batched(
    global_params: GlobalParams,
    pulsar_params: tuple[ParameterVector, ...],
    config: BatchedPTAConfig,
) -> Float[Array, ""]:
    """Batched multi-pulsar log-likelihood via vmap + lax.switch.

    Same signature and semantics as :func:`pta_logL`, but uses a single
    fused XLA trace.

    Parameters
    ----------
    global_params : GlobalParams
        Shared parameters (CW source properties, GWB spectrum, etc.).
    pulsar_params : tuple of ParameterVector
        Per-pulsar timing and noise parameters.
    config : BatchedPTAConfig
        Pre-computed batched configuration.

    Returns
    -------
    logL : scalar
        Sum of per-pulsar log-likelihoods.
    """
    # Reindex all pulsars' param values to universal layout
    batched_values = jnp.stack([
        reindex_param_values(pp, config.layout) for pp in pulsar_params
    ])  # (n_psr, n_params_universal)

    # vmapped per-pulsar logL via lax.switch
    branches = config._branches

    def _per_pulsar(idx, param_values, global_params):
        return jax.lax.switch(idx, branches, param_values, global_params)

    logL_per_psr = jax.vmap(
        _per_pulsar, in_axes=(0, 0, None)
    )(config._branch_indices, batched_values, global_params)

    return jnp.sum(logL_per_psr)
