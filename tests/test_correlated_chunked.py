"""Tests for the chunked correlated PTA likelihood.

Validates that :func:`pta_logL_correlated_chunked` matches
:func:`pta_logL_correlated` across chunk-size variants for HD-correlated
GWB injectors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.pta.params import GlobalParams
from jaxpint.pta.correlated_likelihood import (
    CorrelatedPTAConfig,
    pta_logL_correlated,
    pta_logL_correlated_chunked,
)
from jaxpint.pta.signals.correlated_gwb import HDCorrelatedGWBInjector
from jaxpint.pta.signals.orf import dipole_orf

from tests.helpers import make_simple_pulsar


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_multi_pulsar_setup(n_pulsars=3, n_toas_list=None):
    """Multi-pulsar setup with random sky positions."""
    if n_toas_list is None:
        n_toas_list = [20 + i * 5 for i in range(n_pulsars)]

    rng = np.random.default_rng(42)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    toa_data_list = []
    timing_models = []
    noise_models = []
    pulsar_params = []

    for i in range(n_pulsars):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=n_toas_list[i],
            f0=200.0 + i * 10.0,
            f1=-1e-15 * (1 + i * 0.5),
            seed=42 + i,
        )
        toa_data_list.append(td)
        timing_models.append(tm)
        noise_models.append(nm)
        pulsar_params.append(pp)

    return (
        tuple(toa_data_list),
        tuple(timing_models),
        tuple(noise_models),
        tuple(pulsar_params),
        positions,
    )


def _build_config(n_pulsars, *, orf_func=None, n_toas_list=None):
    (toa_data_list, timing_models, noise_models, pulsar_params, positions) = (
        _make_multi_pulsar_setup(n_pulsars=n_pulsars, n_toas_list=n_toas_list)
    )

    T_span = 365.25 * 86400.0
    n_components = 4

    kw = dict(
        pulsar_positions=positions,
        n_components=n_components,
        T_span=T_span,
        initial_values={"log10_A": -14.0, "gamma": 4.33},
    )
    if orf_func is not None:
        kw["orf_func"] = orf_func
    gwb_injector = HDCorrelatedGWBInjector(**kw)

    global_params = gwb_injector.register_params(GlobalParams.empty())

    config = CorrelatedPTAConfig(
        toa_data_list=toa_data_list,
        timing_models=timing_models,
        noise_models=noise_models,
        signal_injectors=(),
        correlated_injectors=(gwb_injector,),
    )
    return global_params, pulsar_params, config


# ---------------------------------------------------------------------------
# Numeric equivalence: chunked == loop
# ---------------------------------------------------------------------------


CHUNK_SIZES = (1, 2, 3)


class TestCorrelatedChunkedMatchesLoop:
    """pta_logL_correlated_chunked must match pta_logL_correlated."""

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_hd_correlated(self, chunk_size):
        global_params, pulsar_params, config = _build_config(n_pulsars=5)

        logL_loop = float(
            pta_logL_correlated(global_params, pulsar_params, config)
        )
        logL_chunked = pta_logL_correlated_chunked(
            global_params, pulsar_params, config, chunk_size=chunk_size,
        )

        np.testing.assert_allclose(
            logL_chunked, logL_loop, rtol=1e-10, atol=1e-13,
            err_msg=f"chunk_size={chunk_size} mismatch (HD ORF)",
        )

    @pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
    def test_dipole_orf(self, chunk_size):
        # Dipole ORF: Gamma = positions @ positions.T has rank <= 3 (the
        # dimensionality of the dipole moment), so n_pulsars must be <= 3
        # for Gamma to be invertible — see CorrelatedSignalInjector docs.
        global_params, pulsar_params, config = _build_config(
            n_pulsars=3, orf_func=dipole_orf,
        )

        logL_loop = float(
            pta_logL_correlated(global_params, pulsar_params, config)
        )
        logL_chunked = pta_logL_correlated_chunked(
            global_params, pulsar_params, config, chunk_size=chunk_size,
        )

        np.testing.assert_allclose(
            logL_chunked, logL_loop, rtol=1e-10, atol=1e-13,
        )

    def test_chunk_size_equals_n(self):
        global_params, pulsar_params, config = _build_config(n_pulsars=3)
        logL_loop = float(
            pta_logL_correlated(global_params, pulsar_params, config)
        )
        logL_chunked = pta_logL_correlated_chunked(
            global_params, pulsar_params, config, chunk_size=3,
        )
        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-10, atol=1e-13)

    def test_chunk_size_exceeds_n(self):
        global_params, pulsar_params, config = _build_config(n_pulsars=3)
        logL_loop = float(
            pta_logL_correlated(global_params, pulsar_params, config)
        )
        logL_chunked = pta_logL_correlated_chunked(
            global_params, pulsar_params, config, chunk_size=10,
        )
        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-10, atol=1e-13)

    def test_different_toa_counts(self):
        global_params, pulsar_params, config = _build_config(
            n_pulsars=4, n_toas_list=[15, 35, 55, 80],
        )
        logL_loop = float(
            pta_logL_correlated(global_params, pulsar_params, config)
        )
        logL_chunked = pta_logL_correlated_chunked(
            global_params, pulsar_params, config, chunk_size=2,
        )
        np.testing.assert_allclose(logL_chunked, logL_loop, rtol=1e-10, atol=1e-13)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestCorrelatedChunkedValidation:
    def test_zero_chunk_size_raises(self):
        global_params, pulsar_params, config = _build_config(n_pulsars=2)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            pta_logL_correlated_chunked(
                global_params, pulsar_params, config, chunk_size=0,
            )

    def test_negative_chunk_size_raises(self):
        global_params, pulsar_params, config = _build_config(n_pulsars=2)
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            pta_logL_correlated_chunked(
                global_params, pulsar_params, config, chunk_size=-1,
            )
