"""Kepler equation solver for JaxPINT binary models.

Solves the transcendental equation  E - e sin(E) = M  via Halley's method
with unrolled iterations.  Fully compatible with ``jax.jit``, ``jax.grad``,
and ``jax.jacobian`` without any custom differentiation rules.

Reference: PINT ``binary_generic.py`` ``compute_eccentric_anomaly()``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from jaxpint.constants import KEPLER_N_ITER


def _kepler_residual(E, e, M):
    """Kepler equation residual: f(E) = E - e sin(E) - M.

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians (scalar, for ``jax.grad``).
    e : float
        Orbital eccentricity.
    M : float
        Mean anomaly in radians.
    """
    return E - e * jnp.sin(E) - M


#: First derivative df/dE = 1 - e cos(E), via ``jax.grad``.
_kepler_dE = jax.grad(_kepler_residual, argnums=0)
#: Second derivative d²f/dE² = e sin(E), via ``jax.grad``.
_kepler_d2E = jax.grad(_kepler_dE, argnums=0)


def solve_kepler(
    mean_anomaly: Float[Array, " *batch"],
    eccentricity: Float[Array, " *batch"],
) -> Float[Array, " *batch"]:
    """Solve Kepler's equation  E - e sin(E) = M  for eccentric anomaly E.

    Uses Halley's method with a Danby (1988) initial guess and unrolled
    iterations so that ``jax.grad`` and ``jax.jacobian`` work natively.

    Parameters
    ----------
    mean_anomaly : array
        Mean anomaly M in radians.
    eccentricity : array
        Orbital eccentricity (0 <= e < 1).  Must broadcast with *mean_anomaly*.

    Returns
    -------
    array
        Eccentric anomaly E in radians, same shape as inputs.
    """
    M = mean_anomaly
    e = eccentricity

    # Danby (1988) initial guess — much better than E=M at high eccentricity.
    E = M + jnp.sign(jnp.sin(M)) * 0.85 * e

    # Halley's method uses f, f', f'' of the Kepler residual w.r.t. E.
    f_fn = jax.vmap(_kepler_residual)
    fp_fn = jax.vmap(_kepler_dE)
    fpp_fn = jax.vmap(_kepler_d2E)

    for _ in range(KEPLER_N_ITER):
        f = f_fn(E, e, M)
        fp = fp_fn(E, e, M)
        fpp = fpp_fn(E, e, M)
        E = E - 2.0 * f * fp / (2.0 * fp ** 2 - f * fpp)

    return E
