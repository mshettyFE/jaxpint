"""Null-calibration primitives shared by frequentist detection statistics.

The empirical-background recipe — perturb the data's coherence, recompute the
statistic, repeat — reappears across the frequentist arm (CW F-statistics
today; the optimal statistic's sky scrambles and phase shifts next).  This
module holds the small, statistic-agnostic pieces of that recipe: drawing the
perturbations and scoring the observed value against the resulting null
samples.  The statistic-specific folding lives with each statistic (e.g.
:mod:`jaxpint.frequentist.detection`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

__all__ = ["isotropic_positions", "pvalue", "rotate_quadratures"]


def isotropic_positions(
    key: Array,
    n: int,
) -> Float[Array, "n 3"]:
    """``n`` unit vectors drawn isotropically on the sphere.

    Normalized standard-normal draws — the perturbation used by sky-scramble
    nulls, which destroy the geometric (antenna / overlap-reduction) coherence
    between a pulsar array and a correlated signal while preserving every
    per-pulsar data product.
    """
    v = jax.random.normal(key, (n, 3))
    return v / jnp.linalg.norm(v, axis=1, keepdims=True)


def rotate_quadratures(
    sc: Float[Array, "n 2"],
    phases: Float[Array, " n"],
) -> Float[Array, "n 2"]:
    """Rotate per-row (sin, cos) quadrature pairs by per-row phases.

    Row ``i``'s ``(S, C)`` pair is rotated by ``phases[i]`` — the perturbation
    used by phase-shift nulls, which destroy inter-pulsar *phase* coherence
    while preserving each pulsar's spectrum (the quadrature Gram is invariant
    under rotation).  For a single-frequency statistic ``n`` indexes pulsars;
    a multi-frequency generalization (per-frequency 2x2 rotations of a
    ``2 n_freq`` vector) is the optimal statistic's phase shift.
    """
    s, c = sc[:, 0], sc[:, 1]
    cph, sph = jnp.cos(phases), jnp.sin(phases)
    return jnp.stack([cph * s - sph * c, sph * s + cph * c], axis=1)


def pvalue(stat: float, background: Float[Array, " n_real"]) -> float:
    """One-sided p-value: fraction of the ``background`` at least as large as ``stat``."""
    return float((jnp.asarray(background) >= stat).mean())
