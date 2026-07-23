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
        """For e=0, E=M exactly.
        https://en.wikipedia.org/wiki/Mean_anomaly"""
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

    @pytest.mark.parametrize(
        "e, tol",
        [(0.1, 1e-14), (0.9, 1e-13), (0.99, 1e-12)],
    )
    def test_residual_range_monotonic(self, e, tol):
        """Backward error, root selection, and monotonicity at each regime.

        The residual ``E - e*sin(E) - M`` proves E satisfies Kepler's
        equation; the range and monotonicity asserts prove it is the
        *unique* root in [0, 2*pi) (the LHS is monotonic in E), which the
        residual alone cannot.  e=0.99 is deliberately beyond the e<=0.95
        range covered by the PINT parity test below -- keep it.
        """
        M = jnp.linspace(0.0, 2 * jnp.pi, 200, endpoint=False)
        ecc = jnp.full_like(M, e)
        E = solve_kepler(M, ecc)

        residual = E - ecc * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < tol
        assert jnp.all((E >= 0.0) & (E < 2.0 * jnp.pi))
        # E(M) is strictly increasing at fixed e: catches any wrong-branch
        # or wrong-root failure mode invisible to the residual.
        assert jnp.all(jnp.diff(E) > 0.0)

    @pytest.mark.parametrize("M0", [1e-8, 1e-4, 1e-2])
    def test_high_ecc_near_periapsis(self, M0):
        """The pathological corner: M -> 0 at e = 0.99.

        Convergence is slowest near periapsis at high eccentricity
        (E ~ (6M)^(1/3) for small M as e -> 1).  The grid tests above
        start at M = 0 but step past this region; a fitter wandering
        here must still get a converged solution.
        """
        M = jnp.array([M0])
        e = jnp.array([0.99])
        E = solve_kepler(M, e)
        residual = E - e * jnp.sin(E) - M
        assert jnp.max(jnp.abs(residual)) < 1e-12
        assert float(E[0]) > 0.0

    def test_jit_matches_eager(self):
        """Solver produces identical results under jax.jit."""
        M = jnp.linspace(0.0, 2 * jnp.pi, 50, endpoint=False)
        e = jnp.full_like(M, 0.5)
        npt.assert_array_equal(
            np.asarray(jax.jit(solve_kepler)(M, e)), np.asarray(solve_kepler(M, e))
        )

    @pytest.mark.parametrize("e", [0.0, 0.3, 0.9, 0.99])
    def test_gradient_matches_analytic(self, e):
        """dE/dM and dE/de match the implicit-function closed forms.

        By the implicit function theorem applied to E - e*sin(E) = M:

            dE/dM = 1 / (1 - e*cos(E))
            dE/de = sin(E) / (1 - e*cos(E))

        The solver differentiates through unrolled Halley iterations, so
        this measures actual gradient accuracy 
        """
        M = jnp.linspace(0.1, 2 * jnp.pi - 0.1, 25)
        ecc = jnp.full_like(M, e)
        E = solve_kepler(M, ecc)
        denom = 1.0 - ecc * jnp.cos(E)

        # solve_kepler is elementwise, so the gradient of the sum
        # recovers the per-element derivative.
        dE_dM = jax.grad(lambda m: jnp.sum(solve_kepler(m, ecc)))(M)
        npt.assert_allclose(np.asarray(dE_dM), np.asarray(1.0 / denom), rtol=1e-9)

        dE_de = jax.grad(lambda ee: jnp.sum(solve_kepler(M, ee)))(ecc)
        npt.assert_allclose(
            np.asarray(dE_de),
            np.asarray(jnp.sin(E) / denom),
            rtol=1e-9,
            atol=1e-12,
        )

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

    @pytest.mark.parametrize(
        "fn, x0, xdot, delta",
        [
            (compute_ecc, 0.1, 1e-15, 1e-5),   # ecc0 + edot*tt0
            (compute_a1, 10.0, 1e-14, 1e-4),   # a1_0 + a1dot*tt0
        ],
    )
    def test_secular_evolution_linear(self, fn, x0, xdot, delta):
        """ECC/A1 secular terms evolve linearly with tt0."""
        tt0 = jnp.array([0.0, 1e10])
        out = fn(x0, xdot, tt0)
        npt.assert_allclose(out, jnp.array([x0, x0 + delta]), atol=1e-15)

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

    def test_true_anomaly_unwrap_across_orbits(self):
        """Cumulative nu is continuous and strictly increasing across orbit
        boundaries.

        ``compute_true_anomaly`` takes ``orbits``/``mean_anomaly``
        precisely to unwrap the arctan branch into cumulative phase
        (PINT's nu2 convention); the single-orbit tests above never
        exercise that machinery, and branch unwrapping is a classic
        atan2 bug source.  At e=0.9 with this sampling density,
        d(nu) per step is bounded well below pi, so any 2*pi branch
        jump must trip the asserts.
        """
        pb_d = 1.0
        n_orbits = 3
        tt0 = jnp.linspace(0.0, n_orbits * pb_d * SECS_PER_DAY, 2000)
        orbits = compute_orbits_pb(tt0, pb_d)
        M = compute_mean_anomaly(orbits)
        ecc = jnp.full_like(M, 0.9)
        E = solve_kepler(M, ecc)
        nu = compute_true_anomaly(E, ecc, orbits, M)

        dnu = jnp.diff(nu)
        assert jnp.all(dnu > 0.0), "cumulative true anomaly must be increasing"
        assert jnp.all(dnu < jnp.pi), "2*pi branch jump in true anomaly"
        # Exactly n_orbits full revolutions from first to last sample.
        npt.assert_allclose(
            float(nu[-1] - nu[0]), 2.0 * jnp.pi * n_orbits, rtol=1e-12
        )
