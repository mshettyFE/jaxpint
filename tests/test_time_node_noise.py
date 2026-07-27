"""Tests for TimeNodeGPNoise (tent / linear-interp time-domain GP).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.bayes.marginal import marginalize_single_pulsar
from jaxpint.noise import NoiseModel, TimeNodeGPNoise
from jaxpint.utils import build_linear_interp_basis
from tests.helpers import make_params, make_simple_pulsar, make_toa_data

_DAY = 86400.0


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TestBuildLinearInterpBasis:
    def test_enterprise_parity(self):
        """Bit-identical to enterprise's linear_interp_basis (same grid,
        interval assignment, and pruning)."""
        ent_utils = pytest.importorskip("enterprise.signals.utils")
        rng = np.random.default_rng(42)
        t = np.sort(rng.uniform(0.0, 3.0 * 365.25 * _DAY, 300))
        U_j, x_j = build_linear_interp_basis(t, dt=30 * _DAY)
        U_e, x_e = ent_utils.linear_interp_basis(t, dt=30 * _DAY)
        npt.assert_array_equal(U_j, U_e)
        npt.assert_array_equal(x_j, x_e)

    def test_partition_of_unity(self):
        """Within the node span, each TOA's tent weights sum to exactly 1."""
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0.0, 200 * _DAY, 100))
        U, _ = build_linear_interp_basis(t, dt=30 * _DAY)
        npt.assert_allclose(U.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    def test_gap_prunes_unsupported_nodes(self):
        """Nodes inside a TOA gap have no support and are dropped."""
        t = np.concatenate(
            [np.linspace(0, 10 * _DAY, 20), np.linspace(50 * _DAY, 60 * _DAY, 20)]
        )
        nodes = np.arange(0.0, 70 * _DAY, 10 * _DAY)  # 0..60d
        U, kept = build_linear_interp_basis(t, node_times_s=nodes)
        # Nodes at 30d and 40d border no populated interval.
        assert 30 * _DAY not in kept
        assert 40 * _DAY not in kept
        assert U.shape == (len(t), len(kept))
        # Rows still sum to 1: each TOA's two supporting nodes survive.
        npt.assert_allclose(U.sum(axis=1), 1.0, rtol=0, atol=1e-12)

    def test_explicit_nodes_used(self):
        t = np.linspace(0, 100 * _DAY, 50)
        nodes = np.array([0.0, 40 * _DAY, 100 * _DAY])
        U, kept = build_linear_interp_basis(t, node_times_s=nodes)
        npt.assert_array_equal(kept, nodes)
        assert U.shape == (50, 3)


# ---------------------------------------------------------------------------
# Component behavior
# ---------------------------------------------------------------------------


def _make_timenode(n_toas=40, n_freqs=8, T=100.0 * _DAY, chrom=False):
    """Build a TimeNodeGPNoise component and matching params for tests.

    Same ``(n_toas, n_freqs, T)`` signature convention as the ``_make_pl*``
    builders so ``test_correlated_noise_common.py`` can wrap it in its spec
    table; ``n_freqs`` is repurposed as the node-count divisor
    (``dt = T / max(n_freqs, 2)``).  Extra return values follow the uniform
    ``(component, params, toa_data, *extras)`` layout.
    """
    toa_data = make_toa_data(n_toas=n_toas)
    # Basis on an independent uniform grid (same convention as the
    # Fourier component tests: basis times need not equal TOA times).
    t = np.linspace(0.0, T, n_toas)
    comp = TimeNodeGPNoise.from_times(
        t,
        sigma_name="TNNODESIG",
        dt=T / max(n_freqs, 2),
        chrom_idx_name="TNNODEIDX" if chrom else None,
    )
    names = ("TNNODESIG",) + (("TNNODEIDX",) if chrom else ())
    values = (-7.0,) + ((4.0,) if chrom else ())
    params = make_params(names, values, units=("",) * len(names))
    return comp, params, toa_data, t


class TestTimeNodeGPNoise:
    def test_psd_weights_iid(self):
        comp, params, _, _ = _make_timenode()
        w = comp.psd_weights(params)
        assert w.shape == (comp.interp_basis.shape[1],)
        npt.assert_allclose(np.asarray(w), (10.0**-7.0) ** 2, rtol=1e-12)

    def test_static_when_achromatic(self):
        comp, _, _, _ = _make_timenode()
        assert comp.static_basis() is comp._host_columns()

    def test_basis_at_matches_training_basis(self):
        comp, params, _, t = _make_timenode()
        U_at = comp.basis_at(jnp.asarray(t), params)
        npt.assert_allclose(
            np.asarray(U_at), np.asarray(comp.interp_basis), rtol=0, atol=1e-12
        )

    def test_basis_at_outside_span_is_zero(self):
        comp, params, _, t = _make_timenode()
        outside = jnp.asarray([t.min() - 5 * _DAY, t.max() + 5 * _DAY])
        U_at = comp.basis_at(outside, params)
        npt.assert_array_equal(np.asarray(U_at), 0.0)

    def test_basis_at_none_for_chromatic(self):
        comp, params, _, t = _make_timenode(chrom=True)
        assert comp.basis_at(jnp.asarray(t), params) is None


# ---------------------------------------------------------------------------
# Degeneracy with the timing model 
# ---------------------------------------------------------------------------


class TestTimingModelDegeneracy:
    def test_marginalized_logl_finite_and_differentiable(self):
        """Tent span overlaps the spindown quadratic; with the timing model
        analytically marginalized (QR Woodbury path) the likelihood must stay
        finite and differentiable """
        toa_data, timing_model, base_nm, _ = make_simple_pulsar(
            40, f0=200.0, f1=-1e-15
        )
        t = np.asarray(toa_data.tdb_seconds)
        tent = TimeNodeGPNoise.from_times(
            t, sigma_name="TNNODESIG", dt=(t.max() - t.min()) / 6
        )
        nm = NoiseModel(white_noise=base_nm.white_noise, correlated=(tent,))
        params = make_params(
            names=("F0", "F1", "PEPOCH", "EFAC1", "EQUAD1", "TNNODESIG"),
            values=(200.0, -1e-15, 0.0, 1.0, 0.0, -7.0),
            frozen_mask=(False, False, True, True, True, False),
            epoch_int_values={"PEPOCH": 59000.0},
        )
        g, over, reduced = marginalize_single_pulsar(
            over=("F0", "F1"),
            toa_data=toa_data,
            timing_model=timing_model,
            noise_model=nm,
            fiducial_params=params,
        )
        assert over == frozenset({"F0", "F1"})

        val = g(reduced)
        assert bool(jnp.isfinite(val))

        # Only TNNODESIG remains free in the reduced vector.
        def logl_of_free(v):
            return g(reduced.with_free_values(v))

        grad = jax.grad(logl_of_free)(reduced.free_values())
        assert bool(jnp.all(jnp.isfinite(grad)))
