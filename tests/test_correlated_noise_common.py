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

        log10_A = float(params.param_value(component._amp_name))
        gamma = float(params.param_value(component._gam_name))
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


# ---------------------------------------------------------------------------
# Basis-neutral contract (_BasisGPNoise): every basis-GP component -- the four
# power-law Fourier components plus FreeSpectrumNoise and the epoch-indicator
# EcorrNoise -- must satisfy the shared covariance/generate/caching contract,
# independent of what its basis columns are.
# ---------------------------------------------------------------------------

from jaxpint.noise._basis_gp import _BasisGPNoise
from jaxpint.noise.ecorr import EcorrNoise
from jaxpint.noise.free_spectrum import FreeSpectrumNoise
from jaxpint.utils import build_quantization_matrix
from tests.helpers import make_fourier_basis, make_params, make_toa_data


def _build_freespec(n_toas, n_freqs, T):
    F, freqs, df, _ = make_fourier_basis(n_toas, n_freqs, T)
    rho_names = tuple(f"TNFREERHO_{k:04d}" for k in range(n_freqs))
    comp = FreeSpectrumNoise(
        fourier_basis=F, freqs=freqs, freq_bin_widths=df, rho_names=rho_names
    )
    params = make_params(rho_names, [-7.0] * n_freqs, units=("",) * n_freqs)
    return comp, params, make_toa_data(n_toas=n_toas)


def _build_ecorr(n_toas, n_freqs, T):
    # n_freqs/T unused: the epoch basis is built from the TOA times alone.
    toa_data = make_toa_data(n_toas=n_toas)
    tdb_s = np.asarray(toa_data.tdb_seconds)
    masks = {
        "ECORR1": np.arange(n_toas) % 2 == 0,
        "ECORR2": np.arange(n_toas) % 2 == 1,
    }
    U, eslices = build_quantization_matrix(tdb_s, masks, dt=86400.0)
    assert U.shape[1] > 0, "epoch grouping produced no epochs -- trivial test"
    comp = EcorrNoise(
        ecorr_names=("ECORR1", "ECORR2"),
        quantization_matrix=jnp.asarray(U),
        ecorr_epoch_slices=(eslices["ECORR1"], eslices["ECORR2"]),
    )
    params = make_params(("ECORR1", "ECORR2"), [5e-7, 3e-7], units=("s", "s"))
    return comp, params, toa_data


BASIS_GP_SPECS = NOISE_SPECS + [
    NoiseSpec(name="freespec", build=_build_freespec),
    NoiseSpec(name="ecorr", build=_build_ecorr),
]


@pytest.fixture(params=BASIS_GP_SPECS, ids=[s.name for s in BASIS_GP_SPECS])
def basis_gp_spec(request):
    return request.param


class TestBasisGPContract:
    """Contract shared by every _BasisGPNoise subclass, basis-agnostic."""

    def test_covariance_triple_invariants(self, basis_gp_spec):
        n_toas = 40
        component, params, toa_data = basis_gp_spec.build(n_toas, 3, 365.25 * 86400.0)
        Ndiag, U, Phi = component.covariance(toa_data, params)
        assert Ndiag.shape == (n_toas,)
        npt.assert_array_equal(Ndiag, jnp.zeros(n_toas))  # purely low-rank
        assert U.shape[0] == n_toas
        assert Phi.shape == (U.shape[1],)
        assert bool(jnp.all(jnp.isfinite(U)))
        assert bool(jnp.all(jnp.isfinite(Phi)))
        assert bool(jnp.all(Phi >= 0))

    def test_generate_consistent_with_covariance(self, basis_gp_spec):
        """generate() and covariance() must consume the same (U, w) through the
        same hooks with the same key convention: a draw equals the manual
        projection U @ (sqrt(w) * z) of covariance()'s own outputs, bit-exact."""

        component, params, toa_data = basis_gp_spec.build(40, 3, 365.25 * 86400.0)
        key = jax.random.PRNGKey(7)
        draw = component.generate(toa_data, params, key)
        _, U, Phi = component.covariance(toa_data, params)
        z = jax.random.normal(key, shape=(U.shape[1],))
        npt.assert_array_equal(np.asarray(draw), np.asarray(U @ (jnp.sqrt(Phi) * z)))

    def test_columns_cache_concrete_and_guarded(self, basis_gp_spec):
        """The lazy device cache must (a) actually cache, (b) hold a concrete
        array equal to the host columns, and (c) refuse to cache the tracer
        that _host_columns() yields on a tree-reconstructed instance inside a
        jit trace (the leak the guard in _columns_jax exists to prevent)."""
        import equinox as eqx

        component, params, toa_data = basis_gp_spec.build(40, 3, 365.25 * 86400.0)
        # The assertions below inspect
        # _BasisGPNoise internals and would pass vacuously on a component
        # that silently left the hierarchy.
        assert isinstance(component, _BasisGPNoise)

        # (a) caching: repeated access returns the same object.
        first = component._columns_jax
        assert component._columns_jax is first
        # (b) concrete and faithful to the host source of truth.
        assert not isinstance(first, jax.core.Tracer)
        npt.assert_array_equal(np.asarray(first), np.asarray(component._host_columns()))

        # (c) the guard branch: partition/combine inside jit rebuilds the
        # component with tracer fields, so _host_columns() returns a tracer
        # there.  Capture the ephemeral instance at trace time and verify the
        # guard refused to cache it.
        dynamic, static = eqx.partition(component, eqx.is_array)
        captured = []

        @jax.jit
        def evaluate(dyn):
            comp = eqx.combine(dyn, static)
            captured.append(comp)  # host-side capture, runs during tracing
            _, U, Phi = comp.covariance(toa_data, params)
            return jnp.sum(U) + jnp.sum(Phi)

        evaluate(dynamic)
        assert len(captured) == 1, "expected exactly one trace"
        reconstructed = captured[0]
        assert "_columns_jax_cache" not in reconstructed.__dict__, (
            "tracer leaked into the reconstructed instance's cache"
        )
        # The persistent instance's cache is untouched by the traced call.
        assert component.__dict__.get("_columns_jax_cache") is first
