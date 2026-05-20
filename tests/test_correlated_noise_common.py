"""Shared parametrized tests for power-law correlated noise components.

Covers the byte-identical tests previously duplicated across
``test_red_noise.py``, ``test_dm_noise.py``, ``test_chrom_noise.py``, and
``test_sw_noise.py``. Each spec wraps the existing per-model ``_make_pl*``
builder and exposes a normalized ``(component, params, toa_data)`` tuple.

Model-specific tests (basis scaling, alpha sensitivity, geometry,
NoiseModel-with-EFAC integration, GLS fitter end-to-end) stay in their
per-model files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.constants import FYR

from tests.test_chrom_noise import _make_plchrom
from tests.test_dm_noise import _make_pldm
from tests.test_red_noise import _make_plred
from tests.test_sw_noise import _make_plsw


# ---------------------------------------------------------------------------
# Spec: normalize each per-model builder to a uniform (component, params,
# toa_data) tuple. The extra arrays each builder returns vary by model and
# are only needed by model-specific tests, so they live in those files.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseSpec:
    name: str
    build: Callable[..., tuple]  # (n_toas, n_freqs, T) -> (component, params, toa_data)
    # SW noise amplitudes are ~1e-(huge), so plain np.allclose default atol
    # treats different draws as "close" — override per spec.
    different_keys_kwargs: dict = field(default_factory=dict)


def _build_red(n_toas, n_freqs, T):
    plred, params, toa_data, *_ = _make_plred(n_toas=n_toas, n_freqs=n_freqs, T=T)
    return plred, params, toa_data


def _build_dm(n_toas, n_freqs, T):
    pldm, params, toa_data, *_ = _make_pldm(n_toas=n_toas, n_freqs=n_freqs, T=T)
    return pldm, params, toa_data


def _build_chrom(n_toas, n_freqs, T):
    plchrom, params, toa_data, *_ = _make_plchrom(n_toas=n_toas, n_freqs=n_freqs, T=T)
    return plchrom, params, toa_data


def _build_sw(n_toas, n_freqs, T):
    plsw, params, toa_data, *_ = _make_plsw(n_toas=n_toas, n_freqs=n_freqs, T=T)
    return plsw, params, toa_data


NOISE_SPECS = [
    NoiseSpec(name="plred", build=_build_red),
    NoiseSpec(name="pldm", build=_build_dm),
    NoiseSpec(name="plchrom", build=_build_chrom),
    NoiseSpec(name="plsw", build=_build_sw, different_keys_kwargs=dict(atol=0.0, rtol=0.01)),
]


@pytest.fixture(params=NOISE_SPECS, ids=[s.name for s in NOISE_SPECS])
def noise_spec(request):
    return request.param


# ---------------------------------------------------------------------------
# Tests that were byte-identical (modulo builder choice) across the four
# per-model files.
# ---------------------------------------------------------------------------


class TestCorrelatedNoiseShared:
    """Shape, PSD, and generate() contracts shared by all PL noise classes."""

    def test_covariance_shape(self, noise_spec):
        n_toas, n_freqs = 50, 5
        T = 3.0 * 365.25 * 86400.0
        component, params, toa_data = noise_spec.build(n_toas, n_freqs, T)

        Ndiag, U, Phidiag = component.covariance(toa_data, params)

        assert Ndiag.shape == (n_toas,)
        assert U.shape == (n_toas, 2 * n_freqs)
        assert Phidiag.shape == (2 * n_freqs,)
        npt.assert_array_equal(Ndiag, jnp.zeros(n_toas))

    def test_psd_weights_positive(self, noise_spec):
        component, params, _ = noise_spec.build(100, 5, 3.0 * 365.25 * 86400.0)
        weights = component.psd_weights(params)
        assert jnp.all(weights > 0)
        assert jnp.all(jnp.isfinite(weights))

    def test_psd_weights_values(self, noise_spec):
        """All four PL noise classes share the same A^2 / (12 pi^2) PSD formula."""
        n_freqs = 3
        T = 5.0 * 365.25 * 86400.0
        component, params, _ = noise_spec.build(20, n_freqs, T)

        log10_A = -13.0
        gamma = 3.5
        A = 10.0 ** log10_A

        freqs = component.freqs
        df = component.freq_bin_widths
        expected_psd = (
            A ** 2 / (12.0 * np.pi ** 2)
            * FYR ** (gamma - 3.0)
            * np.array(freqs) ** (-gamma)
        )
        expected_weights = np.repeat(expected_psd * np.array(df), 2)

        weights = component.psd_weights(params)
        npt.assert_allclose(np.array(weights), expected_weights, rtol=1e-12)

    def test_generate_shape(self, noise_spec):
        n_toas = 50
        component, params, toa_data = noise_spec.build(n_toas, 5, 3.0 * 365.25 * 86400.0)
        draws = component.generate(toa_data, params, jax.random.PRNGKey(42))
        assert draws.shape == (n_toas,)

    def test_generate_reproducible(self, noise_spec):
        component, params, toa_data = noise_spec.build(100, 5, 3.0 * 365.25 * 86400.0)
        key = jax.random.PRNGKey(42)
        d1 = component.generate(toa_data, params, key)
        d2 = component.generate(toa_data, params, key)
        npt.assert_array_equal(d1, d2)

    def test_generate_different_keys(self, noise_spec):
        component, params, toa_data = noise_spec.build(100, 5, 3.0 * 365.25 * 86400.0)
        d1 = component.generate(toa_data, params, jax.random.PRNGKey(0))
        d2 = component.generate(toa_data, params, jax.random.PRNGKey(1))
        assert not np.allclose(d1, d2, **noise_spec.different_keys_kwargs)
