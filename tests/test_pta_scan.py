"""Tests for the dependency-aware scan helper.

Validates :func:`jaxpint.pta.scan.scan_logL` against the per-cell
:func:`jaxpint.pta.likelihood.pta_logL` reference at machine precision,
exercises the per-pulsar dependency-detection rules, and pins the
output-shape conventions against :func:`numpy.meshgrid`.
"""

from __future__ import annotations

from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxpint.types import GlobalParams
from jaxpint.pta.likelihood import (
    PTAConfig,
    precompute_single_pulsar_pta_factor,
    pta_logL,
    single_pulsar_pta_logL,
    single_pulsar_pta_logL_with_factor,
)
from jaxpint.pta.scan import (
    PerPulsarScanAxis,
    GlobalScanAxis,
    scan_logL,
    _axes_touch_covariance,
    _injectors_contribute_covariance,
)
from jaxpint.likelihood import (
    precompute_single_pulsar_factor,
    single_pulsar_logL,
    single_pulsar_logL_with_factor,
)
from jaxpint.utils import (
    apply_woodbury_dot_factor,
    precompute_woodbury_factor,
    woodbury_dot,
)
from jaxpint.pta.signals.cw import CWInjectorStack

from tests.helpers import make_simple_pulsar, make_params


jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_setup(n_pulsars=3, n_toas_list=None):
    """Multi-pulsar setup with no signal injectors (uniform Spindown)."""
    if n_toas_list is None:
        n_toas_list = [40 + i * 15 for i in range(n_pulsars)]
    toa_list, tm_list, nm_list, pp_list = [], [], [], []
    for i in range(n_pulsars):
        td, tm, nm, pp = make_simple_pulsar(
            n_toas=n_toas_list[i],
            f0=200.0 + i * 10.0,
            f1=-1e-15 * (1 + i * 0.5),
            seed=42 + i,
        )
        toa_list.append(td)
        tm_list.append(tm)
        nm_list.append(nm)
        pp_list.append(pp)
    return (
        tuple(toa_list),
        tuple(tm_list),
        tuple(nm_list),
        tuple(pp_list),
        GlobalParams.empty(),
    )


def _make_cw_setup(n_pulsars=3, n_cw_sources=1):
    """Multi-pulsar setup + CWInjectorStack; adds PX to each pulsar's params."""
    toa_list, tm_list, nm_list, pp_list, gp = _make_setup(n_pulsars)
    rng = np.random.default_rng(123)
    positions = rng.normal(size=(n_pulsars, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions = jnp.array(positions)

    new_pp = []
    for pp in pp_list:
        new_pp.append(make_params(
            names=pp.names + ("PX",),
            values=list(np.array(pp.values)) + [0.5],
            frozen_mask=pp.frozen_mask + (True,),
            epoch_int_values=pp.epoch_int_values,
        ))
    pp_list = tuple(new_pp)

    cw_inj = CWInjectorStack(pulsar_positions=positions, n_sources=n_cw_sources)
    gp = cw_inj.register_params(gp)
    return toa_list, tm_list, nm_list, pp_list, (cw_inj,), gp


def _build_config(toa_list, tm_list, nm_list, signal_injectors=()):
    return PTAConfig(
        toa_data_list=toa_list,
        timing_models=tm_list,
        noise_models=nm_list,
        signal_injectors=signal_injectors,
    )


# ---------------------------------------------------------------------------
# single_pulsar_pta_logL primitive
# ---------------------------------------------------------------------------


class TestSinglePulsarPtaLogL:
    """The per-pulsar primitive must sum to the full pta_logL."""

    def test_sums_to_pta_logL_no_injectors(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm, signal_injectors=())
        full = float(pta_logL(gp, pp, config))
        per_pulsar_sum = float(sum(
            single_pulsar_pta_logL(p, gp, pp[p], config) for p in range(3)
        ))
        np.testing.assert_allclose(per_pulsar_sum, full, rtol=1e-12, atol=1e-15)

    def test_sums_to_pta_logL_with_cw(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=3, n_cw_sources=2)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        full = float(pta_logL(gp, pp, config))
        per_pulsar_sum = float(sum(
            single_pulsar_pta_logL(p, gp, pp[p], config) for p in range(3)
        ))
        np.testing.assert_allclose(per_pulsar_sum, full, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Dependency analysis (per-pulsar vs global axes)
# ---------------------------------------------------------------------------


class TestScanLogLDependencyAnalysis:
    """Verify that constant-pulsar contributions are evaluated only once."""

    def test_one_per_pulsar_axis_only_target_recomputed(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=4)
        config = _build_config(toa, tm, nm)
        target = 2
        grid = jnp.linspace(199.0, 201.0, 5)

        # Patch both the constant path (`single_pulsar_pta_logL`) and the
        # factor path (`single_pulsar_pta_logL_with_factor`) so we can
        # see whichever the dispatcher picks. Constant pulsars go via the
        # plain function once; the target pulsar goes via the factor
        # path inside vmap (traced once).
        call_counts = {p: 0 for p in range(4)}
        from jaxpint.pta import scan as scan_module
        real_plain = scan_module.single_pulsar_pta_logL
        real_factor = scan_module.single_pulsar_pta_logL_with_factor

        def counting_plain(p, *args, **kwargs):
            call_counts[p] += 1
            return real_plain(p, *args, **kwargs)

        def counting_factor(p, *args, **kwargs):
            call_counts[p] += 1
            return real_factor(p, *args, **kwargs)

        with (
            patch.object(scan_module, "single_pulsar_pta_logL",
                         side_effect=counting_plain),
            patch.object(scan_module, "single_pulsar_pta_logL_with_factor",
                         side_effect=counting_factor),
        ):
            _ = scan_logL(
                gp, pp, config,
                axes=[PerPulsarScanAxis(pulsar_idx=target, param_name="F0", values=grid)],
            )

        # Each pulsar's logL is invoked exactly once (constants once,
        # target once because vmap traces the body once).
        assert call_counts[0] == 1
        assert call_counts[1] == 1
        assert call_counts[3] == 1
        assert call_counts[target] == 1

    def test_one_global_axis_all_pulsars_traced(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=3, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid = jnp.linspace(-15.0, -13.0, 4)

        result = scan_logL(
            gp, pp, config,
            axes=[GlobalScanAxis(param_name="cw0_log10_h", values=grid)],
        )
        # 1D output → shape (4,).
        assert result.shape == (4,)


# ---------------------------------------------------------------------------
# Numerical equivalence with per-cell pta_logL
# ---------------------------------------------------------------------------


class TestScanLogLNumeric:

    def test_axes_empty_returns_pta_logL_scalar(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=2)
        config = _build_config(toa, tm, nm)
        result = scan_logL(gp, pp, config, axes=[])
        ref = pta_logL(gp, pp, config)
        np.testing.assert_allclose(float(result), float(ref), rtol=1e-12, atol=1e-15)
        assert result.shape == ()

    def test_1d_per_pulsar_matches_loop(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        grid = jnp.linspace(199.0, 201.0, 6)
        result = scan_logL(
            gp, pp, config,
            axes=[PerPulsarScanAxis(pulsar_idx=1, param_name="F0", values=grid)],
        )
        ref = np.array([
            float(pta_logL(
                gp,
                pp[:1] + (pp[1].with_value("F0", v),) + pp[2:],
                config,
            ))
            for v in grid
        ])
        np.testing.assert_allclose(np.array(result), ref, rtol=1e-12, atol=1e-15)
        assert result.shape == (6,)

    def test_2d_per_pulsar_matches_loop(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=4, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid_a = jnp.linspace(0.4, 0.6, 5)
        grid_b = jnp.linspace(0.3, 0.7, 7)
        A, B = 0, 2
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=A, param_name="PX", values=grid_a),
                PerPulsarScanAxis(pulsar_idx=B, param_name="PX", values=grid_b),
            ],
            indexing="xy",
        )
        # 'xy' → shape (n_b, n_a). Reference: nested loop.
        ref = np.empty((len(grid_b), len(grid_a)))
        for j, vb in enumerate(grid_b):
            for i, va in enumerate(grid_a):
                pp_mod = list(pp)
                pp_mod[A] = pp[A].with_value("PX", float(va))
                pp_mod[B] = pp[B].with_value("PX", float(vb))
                ref[j, i] = float(pta_logL(gp, tuple(pp_mod), config))
        np.testing.assert_allclose(np.array(result), ref, rtol=1e-12, atol=1e-15)

    def test_2d_per_pulsar_plus_global_matches_loop(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=3, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid_local = jnp.linspace(0.4, 0.6, 4)
        grid_global = jnp.linspace(-15.0, -13.0, 5)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=1, param_name="PX", values=grid_local),
                GlobalScanAxis(param_name="cw0_log10_h", values=grid_global),
            ],
            indexing="ij",
        )
        # 'ij' → shape (n_local, n_global).
        ref = np.empty((len(grid_local), len(grid_global)))
        for i, vl in enumerate(grid_local):
            for j, vg in enumerate(grid_global):
                pp_mod = list(pp)
                pp_mod[1] = pp[1].with_value("PX", float(vl))
                gp_mod = gp.with_value("cw0_log10_h", float(vg))
                ref[i, j] = float(pta_logL(gp_mod, tuple(pp_mod), config))
        np.testing.assert_allclose(np.array(result), ref, rtol=1e-12, atol=1e-15)

    def test_3d_mixed_matches_loop(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=3, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        g0 = jnp.linspace(199.5, 200.5, 3)
        g1 = jnp.linspace(0.4, 0.6, 4)
        g2 = jnp.linspace(-15.0, -13.0, 5)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="F0", values=g0),
                PerPulsarScanAxis(pulsar_idx=1, param_name="PX", values=g1),
                GlobalScanAxis(param_name="cw0_log10_h", values=g2),
            ],
            indexing="ij",
        )
        # 'ij' → shape (3, 4, 5).
        ref = np.empty((3, 4, 5))
        for i in range(3):
            for j in range(4):
                for k in range(5):
                    pp_mod = list(pp)
                    pp_mod[0] = pp[0].with_value("F0", float(g0[i]))
                    pp_mod[1] = pp[1].with_value("PX", float(g1[j]))
                    gp_mod = gp.with_value("cw0_log10_h", float(g2[k]))
                    ref[i, j, k] = float(pta_logL(gp_mod, tuple(pp_mod), config))
        np.testing.assert_allclose(np.array(result), ref, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Output shape conventions (numpy.meshgrid parity)
# ---------------------------------------------------------------------------


class TestScanLogLShape:

    def test_2d_xy_shape(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=2)
        config = _build_config(toa, tm, nm)
        gx = jnp.linspace(199.0, 201.0, 3)
        gy = jnp.linspace(199.0, 201.0, 5)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="F0", values=gx),
                PerPulsarScanAxis(pulsar_idx=1, param_name="F0", values=gy),
            ],
            indexing="xy",
        )
        # numpy default: (n_y, n_x) = (5, 3).
        assert result.shape == (5, 3)

    def test_2d_ij_shape(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=2)
        config = _build_config(toa, tm, nm)
        gx = jnp.linspace(199.0, 201.0, 3)
        gy = jnp.linspace(199.0, 201.0, 5)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="F0", values=gx),
                PerPulsarScanAxis(pulsar_idx=1, param_name="F0", values=gy),
            ],
            indexing="ij",
        )
        assert result.shape == (3, 5)

    def test_3d_xy_only_first_two_swap(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        g0 = jnp.linspace(199.0, 201.0, 3)
        g1 = jnp.linspace(199.0, 201.0, 5)
        g2 = jnp.linspace(199.0, 201.0, 7)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="F0", values=g0),
                PerPulsarScanAxis(pulsar_idx=1, param_name="F0", values=g1),
                PerPulsarScanAxis(pulsar_idx=2, param_name="F0", values=g2),
            ],
            indexing="xy",
        )
        # numpy 'xy' for 3D: (n_1, n_0, n_2) — only first two axes swap.
        assert result.shape == (5, 3, 7)

    def test_shape_matches_numpy_meshgrid(self):
        """Programmatic check: scan_logL output shape matches np.meshgrid."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=4)
        config = _build_config(toa, tm, nm)
        grids = [
            jnp.linspace(199.0, 201.0, n) for n in (3, 5, 7, 4)
        ]
        axes = [
            PerPulsarScanAxis(pulsar_idx=p, param_name="F0", values=grids[p])
            for p in range(4)
        ]
        for indexing in ("xy", "ij"):
            result = scan_logL(gp, pp, config, axes=axes, indexing=indexing)
            expected_shape = np.meshgrid(
                *[np.asarray(g) for g in grids], indexing=indexing,
            )[0].shape
            assert result.shape == expected_shape, (
                f"indexing={indexing!r}: scan_logL shape {result.shape} "
                f"!= np.meshgrid shape {expected_shape}"
            )


# ---------------------------------------------------------------------------
# Autodifferentiability
# ---------------------------------------------------------------------------


class TestScanLogLGradient:
    """jax.grad and jax.vmap compose with scan_logL."""

    def test_grad_through_global_axis_argnums_0(self):
        """grad w.r.t. base_global_params at a fixed scan."""
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=2, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid = jnp.linspace(-15.0, -13.0, 4)

        # Sum-of-grid as a scalar so we can take its grad.
        def f_scan(gp_values):
            gp_local = GlobalParams(
                gp_values, gp.names, gp._name_to_index,
            )
            r = scan_logL(
                gp_local, pp, config,
                axes=[GlobalScanAxis(param_name="cw0_log10_h", values=grid)],
            )
            return jnp.sum(r)

        def f_loop(gp_values):
            gp_local = GlobalParams(
                gp_values, gp.names, gp._name_to_index,
            )
            return sum(
                pta_logL(gp_local.with_value("cw0_log10_h", float(v)), pp, config)
                for v in grid
            )

        grad_scan = jax.grad(f_scan)(gp.values)
        grad_loop = jax.grad(f_loop)(gp.values)
        np.testing.assert_allclose(
            np.array(grad_scan), np.array(grad_loop), rtol=1e-10,
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestScanLogLValidation:

    def test_invalid_indexing_raises(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=1)
        config = _build_config(toa, tm, nm)
        with pytest.raises(ValueError, match="indexing must be"):
            scan_logL(gp, pp, config, axes=[], indexing="bogus")

    def test_unknown_per_pulsar_param_raises(self):
        """A typo'd PerPulsarScanAxis param fails early, not as a deep KeyError."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        grid = jnp.linspace(0.0, 1.0, 3)
        with pytest.raises(ValueError, match="not in pulsar"):
            scan_logL(
                gp, pp, config,
                axes=[PerPulsarScanAxis(pulsar_idx=1, param_name="NOPE", values=grid)],
            )

    def test_pulsar_idx_out_of_range_raises(self):
        """An out-of-range pulsar_idx fails early instead of being silently dropped."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        grid = jnp.linspace(0.0, 1.0, 3)
        with pytest.raises(ValueError, match="out of range"):
            scan_logL(
                gp, pp, config,
                axes=[PerPulsarScanAxis(pulsar_idx=99, param_name="F0", values=grid)],
            )

    def test_unknown_global_param_raises(self):
        """A GlobalScanAxis naming an absent global param fails early."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=2)  # gp == GlobalParams.empty()
        config = _build_config(toa, tm, nm)
        grid = jnp.linspace(0.0, 1.0, 3)
        with pytest.raises(ValueError, match="base_global_params"):
            scan_logL(
                gp, pp, config,
                axes=[GlobalScanAxis(param_name="nope", values=grid)],
            )


class TestScanLogLChunking:
    """Chunked outer-axis evaluation must match the unchunked result."""

    def test_chunked_matches_unchunked_1d(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        grid = jnp.linspace(199.0, 201.0, 17)  # not a clean multiple
        axes = [PerPulsarScanAxis(pulsar_idx=1, param_name="F0", values=grid)]
        ref = scan_logL(gp, pp, config, axes=axes)
        for cs in (1, 4, 5, 16, 17, 100):
            got = scan_logL(gp, pp, config, axes=axes, chunk_size=cs)
            np.testing.assert_allclose(
                np.array(got), np.array(ref), rtol=1e-12, atol=1e-15,
                err_msg=f"chunk_size={cs}",
            )

    def test_chunked_matches_unchunked_2d(self):
        """Chunking the 2D PX×PX scan must reproduce the unchunked result."""
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=4, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid_a = jnp.linspace(0.4, 0.6, 13)
        grid_b = jnp.linspace(0.45, 0.55, 7)
        axes = [
            PerPulsarScanAxis(pulsar_idx=0, param_name="PX", values=grid_a),
            PerPulsarScanAxis(pulsar_idx=2, param_name="PX", values=grid_b),
        ]
        ref = scan_logL(gp, pp, config, axes=axes, indexing="ij")
        for cs in (1, 5, 13):
            got = scan_logL(gp, pp, config, axes=axes,
                            indexing="ij", chunk_size=cs)
            np.testing.assert_allclose(
                np.array(got), np.array(ref), rtol=1e-12, atol=1e-15,
                err_msg=f"chunk_size={cs}",
            )


# ---------------------------------------------------------------------------
# Woodbury precompute / apply split
# ---------------------------------------------------------------------------


class TestWoodburyFactorSplit:
    """The split utility must reproduce ``woodbury_dot`` bit-for-bit."""

    def test_factor_apply_matches_woodbury_dot(self):
        rng = np.random.default_rng(0)
        n, k = 50, 8
        Ndiag = jnp.array(rng.uniform(0.5, 2.0, size=n))
        U = jnp.array(rng.normal(size=(n, k)))
        Phi = jnp.array(rng.uniform(0.1, 1.0, size=k))
        x = jnp.array(rng.normal(size=n))
        y = jnp.array(rng.normal(size=n))

        ref_xCy, ref_logdet = woodbury_dot(Ndiag, U, Phi, x, y)
        factor = precompute_woodbury_factor(Ndiag, U, Phi)
        new_xCy, new_logdet = apply_woodbury_dot_factor(factor, x, y)

        np.testing.assert_allclose(float(new_xCy), float(ref_xCy), rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(float(new_logdet), float(ref_logdet), rtol=1e-12, atol=1e-15)

    def test_single_pulsar_logL_with_factor_matches(self):
        """`single_pulsar_logL_with_factor` ≡ `single_pulsar_logL` per cell."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=1)
        toa_p, tm_p, nm_p, pp_p = toa[0], tm[0], nm[0], pp[0]
        ref = float(single_pulsar_logL(toa_p, tm_p, nm_p, pp_p))
        factor = precompute_single_pulsar_factor(toa_p, nm_p, pp_p)
        got = float(single_pulsar_logL_with_factor(toa_p, tm_p, factor, pp_p))
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


class TestSinglePulsarPtaLogLWithFactor:
    """`single_pulsar_pta_logL_with_factor` ≡ `single_pulsar_pta_logL`."""

    def test_no_injectors(self):
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=3)
        config = _build_config(toa, tm, nm)
        for p in range(3):
            ref = float(single_pulsar_pta_logL(p, gp, pp[p], config))
            factor = precompute_single_pulsar_pta_factor(p, gp, pp[p], config)
            got = float(single_pulsar_pta_logL_with_factor(
                p, gp, pp[p], factor, config,
            ))
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)

    def test_with_cw_injector(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=2, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        for p in range(2):
            ref = float(single_pulsar_pta_logL(p, gp, pp[p], config))
            factor = precompute_single_pulsar_pta_factor(p, gp, pp[p], config)
            got = float(single_pulsar_pta_logL_with_factor(
                p, gp, pp[p], factor, config,
            ))
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)

    def test_factor_is_invariant_to_timing_param_change(self):
        """Factor built at base PX still gives correct logL after PX change."""
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=2, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        p = 0
        factor = precompute_single_pulsar_pta_factor(p, gp, pp[p], config)
        # Vary PX (timing-domain) — factor should still be valid.
        for px_value in (0.4, 0.55, 0.7):
            pp_p_new = pp[p].with_value("PX", px_value)
            ref = float(single_pulsar_pta_logL(p, gp, pp_p_new, config))
            got = float(single_pulsar_pta_logL_with_factor(
                p, gp, pp_p_new, factor, config,
            ))
            np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)


class TestScanLogLPrecomputeDispatch:
    """`scan_logL` must select the precompute path for noise-invariant axes."""

    def test_cw_only_setup_classifies_axes_as_safe(self):
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=2, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        # CW injectors override delay only, not covariance.
        assert _injectors_contribute_covariance(injectors) is False
        # PerPulsarScanAxis on PX (timing-domain) → safe for the target pulsar.
        ax = PerPulsarScanAxis(pulsar_idx=0, param_name="PX",
                               values=jnp.array([0.5, 0.6]))
        assert _axes_touch_covariance(0, [ax], (0,), config) is False
        # GlobalScanAxis on cw0_log10_h → safe (no injector covariance).
        ax_g = GlobalScanAxis(param_name="cw0_log10_h",
                              values=jnp.array([-15.0, -14.0]))
        assert _axes_touch_covariance(0, [ax_g], (0,), config) is False

    def test_factor_path_matches_per_cell_loop(self):
        """End-to-end: scan over PX (factor path) ≡ per-cell pta_logL."""
        toa, tm, nm, pp, injectors, gp = _make_cw_setup(n_pulsars=3, n_cw_sources=1)
        config = _build_config(toa, tm, nm, signal_injectors=injectors)
        grid_a = jnp.linspace(0.4, 0.6, 4)
        grid_b = jnp.linspace(0.45, 0.55, 5)
        result = scan_logL(
            gp, pp, config,
            axes=[
                PerPulsarScanAxis(pulsar_idx=0, param_name="PX", values=grid_a),
                PerPulsarScanAxis(pulsar_idx=2, param_name="PX", values=grid_b),
            ],
            indexing="ij",
        )
        ref = np.empty((len(grid_a), len(grid_b)))
        for i, va in enumerate(grid_a):
            for j, vb in enumerate(grid_b):
                pp_mod = list(pp)
                pp_mod[0] = pp[0].with_value("PX", float(va))
                pp_mod[2] = pp[2].with_value("PX", float(vb))
                ref[i, j] = float(pta_logL(gp, tuple(pp_mod), config))
        np.testing.assert_allclose(np.array(result), ref, rtol=1e-12, atol=1e-15)

    def test_noise_param_axis_falls_back(self):
        """Per-pulsar EFAC scan must NOT use the precompute path."""
        toa, tm, nm, pp, gp = _make_setup(n_pulsars=2)
        config = _build_config(toa, tm, nm)
        # Find an actual noise param name on pulsar 0; if none, skip.
        noise_params: list[str] = []
        for comp in nm[0].components:
            if hasattr(comp, "required_params"):
                noise_params.extend(comp.required_params())
        if not noise_params:
            pytest.skip("simple-pulsar fixture has no noise params to scan over")
        param_name = noise_params[0]
        # Confirm the dispatch helper marks the axis as noise-touching.
        ax = PerPulsarScanAxis(
            pulsar_idx=0, param_name=param_name,
            values=jnp.array([1.0, 1.01]),
        )
        assert _axes_touch_covariance(0, [ax], (0,), config) is True
