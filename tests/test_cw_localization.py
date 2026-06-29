"""Tests for the CW Fisher sky-localization machinery (:mod:`jaxpint.pta.cw_localization`).

Exercises the ``make_logL_2sky`` construction helper, the bilinear-derivative
Gram extraction (``gram_at_pixel`` / ``gram_block_at_pair``), the joint-Fisher
assembly (input validation + structure), and the per-source marginal credible
areas — using a toy, exactly-bilinear log-likelihood so the derivative trick is
exercised end-to-end without a real PTA.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from jaxpint.types import GlobalParams
from jaxpint.pta.cw_localization import (
    assemble_joint_fisher,
    gram_at_pixel,
    gram_block_at_pair,
    make_logL_2sky,
    per_source_credible_areas_deg2,
)

jax.config.update("jax_enable_x64", True)

_STR_TO_DEG2 = (180.0 / np.pi) ** 2
# CW fields each injector exposes (the three varied ones + fixed orientation).
_FIELDS = ("h0", "cos_gwtheta", "gwphi", "cos_inc", "psi", "phase0")


def _toy_gp(*prefixes):
    names, vals = [], []
    for p in prefixes:
        for f in _FIELDS:
            names.append(f"{p}_{f}")
            vals.append(0.0)
    return GlobalParams.empty().add_params(names, vals)


def _Z(sa, sb):
    """A smooth, sky-dependent cross-template overlap with nonzero mixed Hessian."""
    return (1.0 + jnp.dot(sa, sb)) ** 2


def _toy_g(prefix_a, prefix_b):
    """``g(gp, reduced_pp)`` exactly bilinear in the two amplitudes."""
    def g(gp, reduced_pp):
        ha = gp.param_value(f"{prefix_a}_h0")
        hb = gp.param_value(f"{prefix_b}_h0")
        sa = jnp.array([gp.param_value(f"{prefix_a}_cos_gwtheta"),
                        gp.param_value(f"{prefix_a}_gwphi")])
        sb = jnp.array([gp.param_value(f"{prefix_b}_cos_gwtheta"),
                        gp.param_value(f"{prefix_b}_gwphi")])
        return ha + hb - 0.5 * ha**2 - 0.5 * hb**2 - ha * hb * _Z(sa, sb)
    return g


class TestMakeLogL2Sky:
    def test_binds_amplitudes_and_sky(self):
        gp = _toy_gp("cwt", "cwd")
        g = _toy_g("cwt", "cwd")
        logL = make_logL_2sky(g, gp, (), "cwt", "cwd")
        ha, hb = 1.3, 0.7
        sa, sb = jnp.array([0.2, 1.0]), jnp.array([-0.4, 2.1])
        gp_ref = (
            gp.with_value("cwt_h0", ha).with_value("cwt_cos_gwtheta", sa[0])
            .with_value("cwt_gwphi", sa[1]).with_value("cwd_h0", hb)
            .with_value("cwd_cos_gwtheta", sb[0]).with_value("cwd_gwphi", sb[1])
        )
        npt.assert_allclose(float(logL(ha, hb, sa, sb)), float(g(gp_ref, ())), rtol=1e-12)

    def test_leaves_fixed_params_untouched(self):
        # A pinned fixed param must survive into the closure unchanged.
        gp = _toy_gp("cwt", "cwd").with_value("cwt_phase0", 0.55)
        seen = {}

        def g(gp_in, _):
            seen["phase0"] = float(gp_in.param_value("cwt_phase0"))
            return gp_in.param_value("cwt_h0")

        make_logL_2sky(g, gp, (), "cwt", "cwd")(1.0, 0.0, jnp.zeros(2), jnp.zeros(2))
        assert seen["phase0"] == 0.55


class TestGram:
    def test_gram_matches_mixed_Z_hessian(self):
        logL = make_logL_2sky(_toy_g("cwt", "cwd"), _toy_gp("cwt", "cwd"), (), "cwt", "cwd")
        sa, sb = jnp.array([0.3, 0.5]), jnp.array([0.1, -0.2])
        ref = jax.jacfwd(jax.jacrev(_Z, argnums=0), argnums=1)(sa, sb)
        npt.assert_allclose(np.array(gram_block_at_pair(logL, sa, sb)), np.array(ref), rtol=1e-9)

    def test_gram_at_pixel_is_diagonal_case(self):
        logL = make_logL_2sky(_toy_g("cwt", "cwd"), _toy_gp("cwt", "cwd"), (), "cwt", "cwd")
        s = jnp.array([0.4, 1.2])
        npt.assert_allclose(
            np.array(gram_at_pixel(logL, s)),
            np.array(gram_block_at_pair(logL, s, s)),
            rtol=1e-12,
        )


class TestAssembleJointFisher:
    def test_assembles_scaled_and_symmetric(self):
        blocks = {
            (0, 0): jnp.eye(2),
            (0, 1): jnp.array([[1.0, 0.0], [0.0, 2.0]]),
            (1, 1): 3.0 * jnp.eye(2),
        }
        F = assemble_joint_fisher(blocks, jnp.array([1.0, 2.0]), 2)
        assert F.shape == (4, 4)
        npt.assert_allclose(np.array(F), np.array(F).T)
        # off-diagonal block scaled by h0[0]*h0[1] = 2, and symmetrized.
        npt.assert_allclose(np.array(F[0:2, 2:4]), np.array([[2.0, 0.0], [0.0, 4.0]]))
        npt.assert_allclose(np.array(F[2:4, 0:2]), np.array([[2.0, 0.0], [0.0, 4.0]]))

    def test_rejects_missing_block(self):
        with pytest.raises(ValueError, match="Missing"):
            assemble_joint_fisher({(0, 0): jnp.eye(2), (1, 1): jnp.eye(2)}, jnp.ones(2), 2)

    def test_rejects_out_of_range_block(self):
        blocks = {(0, 0): jnp.eye(2), (0, 1): jnp.eye(2), (1, 1): jnp.eye(2), (0, 2): jnp.eye(2)}
        with pytest.raises(ValueError, match="Unexpected"):
            assemble_joint_fisher(blocks, jnp.ones(2), 2)

    def test_rejects_wrong_h0_length(self):
        blocks = {(0, 0): jnp.eye(2), (0, 1): jnp.eye(2), (1, 1): jnp.eye(2)}
        with pytest.raises(ValueError, match="h0_targets has length"):
            assemble_joint_fisher(blocks, jnp.ones(1), 2)


class TestPerSourceCredibleAreas:
    def test_matches_lu_inverse_reference(self):
        rng = np.random.default_rng(0)
        K = 3
        A = rng.normal(size=(2 * K, 2 * K))
        F = jnp.asarray(A @ A.T + 2.0 * np.eye(2 * K))  # SPD, well-conditioned
        got = np.array(per_source_credible_areas_deg2(F, K, level=0.9))
        Sig = np.linalg.inv(np.array(F))
        ref = np.array([
            np.pi * (-2 * np.log(0.1))
            * np.sqrt(np.linalg.det(Sig[2 * k:2 * k + 2, 2 * k:2 * k + 2]))
            * _STR_TO_DEG2
            for k in range(K)
        ])
        assert got.shape == (K,)
        npt.assert_allclose(got, ref, rtol=1e-10)

    def test_singular_fisher_gives_inf(self):
        # Rank-deficient PSD joint Fisher (duplicated rows) -> Cholesky NaN -> inf.
        S = jnp.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                       [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        areas = np.array(per_source_credible_areas_deg2(S, 2, level=0.9))
        assert np.all(np.isinf(areas))
