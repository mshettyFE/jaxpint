"""Tests for the Kepler equation solver and common orbital mechanics."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxpint.types.dual_float import DualFloat
import numpy.testing as npt
import pytest

from jaxpint.binary.kepler import solve_kepler
from jaxpint.constants import SECS_PER_DAY
from jaxpint.binary.common import (
    compute_orbits_pb,
    compute_mean_anomaly,
    compute_ecc,
    compute_a1,
    compute_true_anomaly,
    compute_tt0,
)


class TestSolveKepler:
    """Tests for the Kepler equation solver."""

    def test_circular_orbit(self):
        """For e=0, E=M exactly."""
        M = jnp.linspace(0, 2 * jnp.pi, 100)
        e = jnp.zeros_like(M)
        E = solve_kepler(M, e)
        npt.assert_allclose(E, M, atol=1e-14)

    def test_known_solution(self):
        """E=pi solves M=pi for any eccentricity (sin(pi)=0), so a nonzero e
        still recovers E=pi -- a known solution distinct from the e=0 case."""
        M = jnp.array([jnp.pi])
        e = jnp.array([0.5])
        E = solve_kepler(M, e)
        npt.assert_allclose(E, jnp.pi, atol=1e-14)

    def test_low_eccentricity(self):
        """Verify residual E - e*sin(E) - M is tiny for low e."""
        M = jnp.linspace(0, 2 * jnp.pi, 200)
        e = jnp.full_like(M, 0.1)
        E = solve_kepler(M, e)
        residual = E - e * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < 1e-14

    def test_high_eccentricity(self):
        """Verify convergence for high eccentricity (e=0.9)."""
        M = jnp.linspace(0.01, 2 * jnp.pi - 0.01, 200)
        e = jnp.full_like(M, 0.9)
        E = solve_kepler(M, e)
        residual = E - e * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < 1e-13

    def test_very_high_eccentricity(self):
        """Verify convergence for very high eccentricity (e=0.99)."""
        M = jnp.linspace(0.01, 2 * jnp.pi - 0.01, 200)
        e = jnp.full_like(M, 0.99)
        E = solve_kepler(M, e)
        residual = E - e * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < 1e-12

    def test_jit_compatible(self):
        """Solver works under jax.jit."""
        M = jnp.linspace(0, 2 * jnp.pi, 50)
        e = jnp.full_like(M, 0.5)
        E = jax.jit(solve_kepler)(M, e)
        residual = E - e * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < 1e-14

    def test_differentiable(self):
        """Solver is differentiable via JAX autodiff."""
        def f(M):
            e = jnp.full_like(M, 0.3)
            return jnp.sum(solve_kepler(M, e))

        M = jnp.array([1.0, 2.0, 3.0])
        grad = jax.grad(f)(M)
        # dE/dM = 1/(1 - e*cos(E)), should be finite and positive
        assert jnp.all(jnp.isfinite(grad))
        assert jnp.all(grad > 0)

    def test_compare_pint(self):
        """Compare against PINT's Kepler solver."""
        pytest.importorskip("pint")
        from pint.models.stand_alone_psr_binaries.binary_generic import PSR_BINARY

        bm = PSR_BINARY()
        M_np = np.linspace(0.01, 2 * np.pi - 0.01, 100)
        for ecc_val in [0.0, 0.1, 0.5, 0.8, 0.95]:
            e_np = np.full_like(M_np, ecc_val)
            E_pint = bm.compute_eccentric_anomaly(e_np, M_np).value

            M_jax = jnp.array(M_np)
            e_jax = jnp.array(e_np)
            E_jax = np.array(solve_kepler(M_jax, e_jax))

            npt.assert_allclose(E_jax, E_pint, atol=1e-12,
                                err_msg=f"Mismatch at e={ecc_val}")


class TestCommonOrbital:
    """Tests for shared orbital mechanics functions."""

    def test_compute_tt0_precision(self):
        """int/frac split preserves precision for large MJDs."""
        tdb = DualFloat(int=jnp.array([59000.0, 59001.0]), frac=jnp.array([0.5, 0.75]))
        epoch = DualFloat(int=jnp.array(59000.0), frac=jnp.array(0.0))
        tt0 = compute_tt0(tdb, epoch)
        expected = jnp.array([0.5 * SECS_PER_DAY, 1.75 * SECS_PER_DAY])
        npt.assert_allclose(tt0, expected, atol=1e-10)

    def test_orbits_pb_circular(self):
        """For PBDOT=XPBDOT=0, orbits = tt0/PB."""
        pb_d = 1.5  # days
        tt0_s = jnp.array([0.0, 1.5 * SECS_PER_DAY, 3.0 * SECS_PER_DAY])
        orbits = compute_orbits_pb(tt0_s, pb_d)
        npt.assert_allclose(orbits, jnp.array([0.0, 1.0, 2.0]), atol=1e-14)

    def test_mean_anomaly_range(self):
        """Mean anomaly should be in [0, 2*pi)."""
        orbits = jnp.array([0.0, 0.5, 1.0, 1.25, 2.75])
        M = compute_mean_anomaly(orbits)
        assert jnp.all(M >= 0.0)
        # Strict upper bound: the interval is half-open [0, 2*pi), so a failure
        # to wrap (returning exactly/above 2*pi) must fail here.
        assert jnp.all(M < 2.0 * jnp.pi)

    def test_ecc_time_dependence(self):
        """Eccentricity evolves linearly with time."""
        ecc0 = 0.1
        edot = 1e-15  # 1/s
        tt0 = jnp.array([0.0, 1e10])
        ecc = compute_ecc(ecc0, edot, tt0)
        npt.assert_allclose(ecc, jnp.array([0.1, 0.1 + 1e-5]), atol=1e-15)

    def test_a1_time_dependence(self):
        """Semi-major axis evolves linearly with time."""
        a1_0 = 10.0  # ls
        a1dot = 1e-14  # ls/s
        tt0 = jnp.array([0.0, 1e10])
        a1 = compute_a1(a1_0, a1dot, tt0)
        npt.assert_allclose(a1, jnp.array([10.0, 10.0 + 1e-4]), atol=1e-15)

    def test_true_anomaly_circular(self):
        """For e=0, true anomaly equals mean anomaly."""
        M = jnp.linspace(0.01, 2 * jnp.pi - 0.01, 50)
        e = jnp.zeros_like(M)
        E = solve_kepler(M, e)
        orbits = M / (2 * jnp.pi)
        nu = compute_true_anomaly(E, e, orbits, M)
        npt.assert_allclose(nu, M, atol=1e-12)

    def test_true_anomaly_symmetry(self):
        """True anomaly at E=pi should be pi regardless of eccentricity."""
        E = jnp.array([jnp.pi])
        for ecc_val in [0.0, 0.1, 0.5, 0.9]:
            ecc = jnp.array([ecc_val])
            M = E - ecc * jnp.sin(E)
            orbits = M / (2 * jnp.pi)
            nu = compute_true_anomaly(E, ecc, orbits, M)
            npt.assert_allclose(nu, jnp.array([jnp.pi]), atol=1e-12,
                                err_msg=f"Failed at e={ecc_val}")
