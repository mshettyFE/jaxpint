"""Tests for jaxpint.utils."""

import jax
import jax.numpy as jnp
import pytest


from jaxpint.utils import (
    taylor_horner,
    taylor_horner_deriv,
    weighted_mean,
    normalize_designmatrix,
    sherman_morrison_dot,
    woodbury_dot,
)


# ===========================================================================
# Taylor polynomial tests
# ===========================================================================

class TestTaylorHorner:
    def test_constant(self):
        result = taylor_horner(jnp.array(99.0), jnp.array([5.0]))
        assert jnp.isclose(result, 5.0)

    def test_linear(self):
        # 10 + 3*x  at x=2  -> 16
        result = taylor_horner(jnp.array(2.0), jnp.array([10.0, 3.0]))
        assert jnp.isclose(result, 16.0)

    def test_cubic_pint_example(self):
        # From PINT docstring: 10 + 3*x/1! + 4*x^2/2! + 12*x^3/3! at x=2 -> 40
        result = taylor_horner(jnp.array(2.0), jnp.array([10.0, 3.0, 4.0, 12.0]))
        assert jnp.isclose(result, 40.0)

    def test_array_input(self):
        x = jnp.array([0.0, 1.0, 2.0])
        coeffs = jnp.array([10.0, 3.0, 4.0, 12.0])
        result = taylor_horner(x, coeffs)
        assert result.shape == (3,)
        # At x=0: just the constant term
        assert jnp.isclose(result[0], 10.0)
        # At x=2: 40 (PINT example)
        assert jnp.isclose(result[2], 40.0)

    def test_jit(self):
        f = jax.jit(taylor_horner)
        result = f(jnp.array(2.0), jnp.array([10.0, 3.0, 4.0, 12.0]))
        assert jnp.isclose(result, 40.0)


class TestTaylorHornerDeriv:
    def test_zeroth_matches_horner(self):
        x = jnp.array(2.0)
        coeffs = jnp.array([10.0, 3.0, 4.0, 12.0])
        assert jnp.isclose(
            taylor_horner_deriv(x, coeffs, 0),
            taylor_horner(x, coeffs),
        )

    def test_first_deriv_pint_example(self):
        # From PINT docstring: d/dx [10 + 3x + 4x^2/2! + 12x^3/3!] at x=2 -> 35
        result = taylor_horner_deriv(
            jnp.array(2.0), jnp.array([10.0, 3.0, 4.0, 12.0]), 1
        )
        assert jnp.isclose(result, 35.0)

    def test_first_deriv_linear(self):
        # d/dx [a + b*x] = b
        result = taylor_horner_deriv(
            jnp.array(7.0), jnp.array([5.0, 3.0]), 1
        )
        assert jnp.isclose(result, 3.0)

    def test_second_deriv_quadratic(self):
        # f = a + b*x + c*x^2/2!  ->  f'' = c
        result = taylor_horner_deriv(
            jnp.array(7.0), jnp.array([5.0, 3.0, 11.0]), 2
        )
        assert jnp.isclose(result, 11.0)

    def test_deriv_exceeds_degree(self):
        # 3rd derivative of a linear poly -> 0
        result = taylor_horner_deriv(
            jnp.array(2.0), jnp.array([10.0, 3.0]), 2
        )
        assert jnp.isclose(result, 0.0)

    def test_jit(self):
        f = jax.jit(taylor_horner_deriv, static_argnums=(2,))
        result = f(jnp.array(2.0), jnp.array([10.0, 3.0, 4.0, 12.0]), 1)
        assert jnp.isclose(result, 35.0)

    def test_grad_wrt_x(self):
        coeffs = jnp.array([10.0, 3.0, 4.0, 12.0])

        @jax.grad
        def f(x):
            return taylor_horner(x, coeffs)

        # Gradient of the polynomial at x=2 should equal
        # the first derivative = 35.0
        assert jnp.isclose(f(jnp.array(2.0)), 35.0)

    def test_grad_wrt_coeffs(self):
        x = jnp.array(2.0)

        @jax.grad
        def f(coeffs):
            return taylor_horner(x, coeffs)

        # d/d(coeffs[0]) = 1 (constant term)
        # d/d(coeffs[1]) = x/1! = 2
        # d/d(coeffs[2]) = x^2/2! = 2
        # d/d(coeffs[3]) = x^3/3! = 8/6 = 4/3
        grad = f(jnp.array([10.0, 3.0, 4.0, 12.0]))
        assert jnp.isclose(grad[0], 1.0)
        assert jnp.isclose(grad[1], 2.0)
        assert jnp.isclose(grad[2], 2.0)
        assert jnp.isclose(grad[3], 4.0 / 3.0)


# ===========================================================================
# Weighted mean tests
# ===========================================================================

class TestWeightedMean:
    def test_uniform_weights(self):
        arr = jnp.array([1.0, 2.0, 3.0, 4.0])
        w = jnp.ones(4)
        wmean, werr = weighted_mean(arr, w)
        assert jnp.isclose(wmean, 2.5)
        assert jnp.isclose(werr, 1.0 / jnp.sqrt(4.0))

    def test_known_values(self):
        arr = jnp.array([1.0, 3.0])
        w = jnp.array([1.0, 3.0])
        wmean, _ = weighted_mean(arr, w)
        # (1*1 + 3*3) / (1+3) = 10/4 = 2.5
        assert jnp.isclose(wmean, 2.5)

    def test_inputmean_override(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        w = jnp.ones(3)
        wmean, _ = weighted_mean(arr, w, inputmean=99.0)
        assert jnp.isclose(wmean, 99.0)

    def test_calcerr(self):
        arr = jnp.array([1.0, 3.0])
        w = jnp.array([1.0, 3.0])
        wmean, werr = weighted_mean(arr, w, calcerr=True)
        # wmean = 2.5
        # werr = sqrt( (1^2*(1-2.5)^2 + 3^2*(3-2.5)^2) ) / 4
        #      = sqrt(1*2.25 + 9*0.25) / 4 = sqrt(4.5) / 4
        expected = jnp.sqrt(4.5) / 4.0
        assert jnp.isclose(werr, expected)

    def test_sdev(self):
        arr = jnp.array([1.0, 3.0])
        w = jnp.array([1.0, 3.0])
        wmean, werr, wsdev = weighted_mean(arr, w, sdev=True)
        # wvar = (1*(1-2.5)^2 + 3*(3-2.5)^2) / 4 = (2.25 + 0.75)/4 = 0.75
        assert jnp.isclose(wsdev, jnp.sqrt(0.75))

    def test_jit(self):
        arr = jnp.array([1.0, 2.0, 3.0])
        w = jnp.ones(3)
        f = jax.jit(weighted_mean)
        wmean, werr = f(arr, w)
        assert jnp.isclose(wmean, 2.0)


# ===========================================================================
# Design matrix normalization tests
# ===========================================================================

class TestNormalizeDesignmatrix:
    def test_unit_norm_columns_unchanged(self):
        # Columns already unit norm
        M = jnp.eye(3)
        Mn, norms, degen = normalize_designmatrix(M)
        assert jnp.allclose(Mn, M)
        assert jnp.allclose(norms, 1.0)
        assert not jnp.any(degen)

    def test_normalization_correct(self):
        key = jax.random.PRNGKey(0)
        M = jax.random.normal(key, (10, 3))
        Mn, norms, degen = normalize_designmatrix(M)
        # Each column of Mn should have unit L2 norm
        col_norms = jnp.sqrt(jnp.sum(Mn ** 2, axis=0))
        assert jnp.allclose(col_norms, 1.0)
        assert not jnp.any(degen)

    def test_zero_column(self):
        M = jnp.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        Mn, norms, degen = normalize_designmatrix(M)
        # Zero column stays zero, norm is set to 1
        assert jnp.allclose(Mn[:, 1], 0.0)
        assert jnp.isclose(norms[1], 1.0)
        # Non-zero column is normalized
        assert jnp.isclose(jnp.sqrt(jnp.sum(Mn[:, 0] ** 2)), 1.0)
        # Degenerate mask flags the zero column
        assert not degen[0]
        assert degen[1]

    def test_reconstruction(self):
        key = jax.random.PRNGKey(1)
        M = jax.random.normal(key, (5, 4))
        Mn, norms, _ = normalize_designmatrix(M)
        assert jnp.allclose(Mn * norms, M)

    def test_jit(self):
        key = jax.random.PRNGKey(2)
        M = jax.random.normal(key, (5, 3))
        f = jax.jit(normalize_designmatrix)
        Mn, norms, degen = f(M)
        col_norms = jnp.sqrt(jnp.sum(Mn ** 2, axis=0))
        assert jnp.allclose(col_norms, 1.0)


# ===========================================================================
# Sherman–Morrison tests
# ===========================================================================

class TestShermanMorrisonDot:
    def _dense_reference(self, Ndiag, v, w, x, y):
        """Build C explicitly and compute x^T C^{-1} y."""
        C = jnp.diag(Ndiag) + w * jnp.outer(v, v)
        C_inv = jnp.linalg.inv(C)
        result = x @ C_inv @ y
        _, logdet = jnp.linalg.slogdet(C)
        return result, logdet

    def test_against_dense(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n = 8
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        v = jax.random.normal(k2, (n,))
        w = jnp.array(2.0)
        x = jax.random.normal(k3, (n,))
        y = jax.random.normal(k4, (n,))

        result, logdet = sherman_morrison_dot(Ndiag, v, w, x, y)
        ref_result, ref_logdet = self._dense_reference(Ndiag, v, w, x, y)

        assert jnp.isclose(result, ref_result, rtol=1e-10)
        assert jnp.isclose(logdet, ref_logdet, rtol=1e-10)

    def test_symmetric(self):
        key = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(key, 3)
        n = 5
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        v = jax.random.normal(k2, (n,))
        x = jax.random.normal(k3, (n,))
        # x^T C^{-1} x should be positive (C is positive definite)
        result, _ = sherman_morrison_dot(Ndiag, v, jnp.array(1.0), x, x)
        assert result > 0

    def test_jit(self):
        n = 5
        Ndiag = jnp.ones(n)
        v = jnp.ones(n)
        f = jax.jit(sherman_morrison_dot)
        result, logdet = f(Ndiag, v, jnp.array(1.0), v, v)
        assert jnp.isfinite(result)
        assert jnp.isfinite(logdet)

    def test_grad(self):
        key = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(key, 3)
        n = 4
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        v = jax.random.normal(k2, (n,))
        x = jax.random.normal(k3, (n,))

        @jax.grad
        def loss(Ndiag):
            r, _ = sherman_morrison_dot(Ndiag, v, jnp.array(1.0), x, x)
            return r

        g = loss(Ndiag)
        assert g.shape == (n,)
        assert jnp.all(jnp.isfinite(g))


# ===========================================================================
# Woodbury tests
# ===========================================================================

class TestWoodburyDot:
    def _dense_reference(self, Ndiag, U, Phidiag, x, y):
        """Build C explicitly and compute x^T C^{-1} y."""
        C = jnp.diag(Ndiag) + U @ jnp.diag(Phidiag) @ U.T
        C_inv = jnp.linalg.inv(C)
        result = x @ C_inv @ y
        _, logdet = jnp.linalg.slogdet(C)
        return result, logdet

    def test_against_dense(self):
        key = jax.random.PRNGKey(0)
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        n, k = 8, 3
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        U = jax.random.normal(k2, (n, k))
        Phidiag = jnp.abs(jax.random.normal(k3, (k,))) + 0.1
        x = jax.random.normal(k4, (n,))
        y = jax.random.normal(k5, (n,))

        result, logdet = woodbury_dot(Ndiag, U, Phidiag, x, y)
        ref_result, ref_logdet = self._dense_reference(Ndiag, U, Phidiag, x, y)

        assert jnp.isclose(result, ref_result, rtol=1e-10)
        assert jnp.isclose(logdet, ref_logdet, rtol=1e-10)

    def test_logdet(self):
        key = jax.random.PRNGKey(3)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n, k = 6, 2
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        U = jax.random.normal(k2, (n, k))
        Phidiag = jnp.abs(jax.random.normal(k3, (k,))) + 0.1
        x = jax.random.normal(k4, (n,))

        _, logdet = woodbury_dot(Ndiag, U, Phidiag, x, x)
        _, ref_logdet = self._dense_reference(Ndiag, U, Phidiag, x, x)
        assert jnp.isclose(logdet, ref_logdet, rtol=1e-10)

    def test_rank1_matches_sherman_morrison(self):
        key = jax.random.PRNGKey(4)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n = 6
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        v = jax.random.normal(k2, (n,))
        w = jnp.abs(jax.random.normal(k3, ())) + 0.1
        x = jax.random.normal(k4, (n,))

        # Sherman-Morrison
        sm_result, sm_logdet = sherman_morrison_dot(Ndiag, v, w, x, x)

        # Woodbury with rank-1: U = v[:, None], Phidiag = [w]
        wb_result, wb_logdet = woodbury_dot(
            Ndiag, v[:, None], jnp.array([w]), x, x
        )

        assert jnp.isclose(sm_result, wb_result, rtol=1e-10)
        assert jnp.isclose(sm_logdet, wb_logdet, rtol=1e-10)

    def test_jit(self):
        n, k = 5, 2
        Ndiag = jnp.ones(n)
        U = jnp.ones((n, k))
        Phidiag = jnp.ones(k)
        x = jnp.ones(n)
        f = jax.jit(woodbury_dot)
        result, logdet = f(Ndiag, U, Phidiag, x, x)
        assert jnp.isfinite(result)
        assert jnp.isfinite(logdet)

    def test_grad(self):
        key = jax.random.PRNGKey(5)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        n, k = 4, 2
        Ndiag = jnp.abs(jax.random.normal(k1, (n,))) + 1.0
        U = jax.random.normal(k2, (n, k))
        Phidiag = jnp.abs(jax.random.normal(k3, (k,))) + 0.1
        x = jax.random.normal(k4, (n,))

        @jax.grad
        def loss(Ndiag):
            r, _ = woodbury_dot(Ndiag, U, Phidiag, x, x)
            return r

        g = loss(Ndiag)
        assert g.shape == (n,)
        assert jnp.all(jnp.isfinite(g))
