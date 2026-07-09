"""Phase 4a: `prior=` argument on the CW marginalized upper limit.

Anchors: `prior=None` is the unchanged improper-uniform closed form; a broad
`dist.Uniform(0, huge)` via the grid path reproduces it; a non-uniform prior
(`LogUniform`) runs deterministically and shifts the limit.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("numpyro")
import numpyro.distributions as dist

from jaxpint.pta.cw_upper_limit import h0_95_marginalized
from jaxpint.stats.regions import truncated_gaussian_upper_limit


# A small multi-orientation example (matched filter / power per orientation).
XS = jnp.array([0.5, -0.3, 0.1])
YS = jnp.array([2.0, 1.5, 3.0])


def test_none_is_improper_uniform_closed_form():
    # Single clean non-detection: mu=0, sigma=1/sqrt(Y) -> UL = sigma * Phi^{-1}(0.975).
    ul = h0_95_marginalized(jnp.array([0.0]), jnp.array([4.0]))  # sigma = 0.5
    expected = truncated_gaussian_upper_limit(jnp.array(0.0), jnp.array(0.5), 0.95)
    np.testing.assert_allclose(float(ul), float(expected), rtol=1e-10)


def test_grid_matches_closed_form_for_broad_uniform():
    # A very wide uniform box (hi >> posterior width) is the improper-flat limit,
    # so the deterministic grid path must reproduce the closed-form quantile.
    closed = float(h0_95_marginalized(XS, YS))  # prior=None
    grid = float(h0_95_marginalized(XS, YS, prior=dist.Uniform(0.0, 1e6)))
    np.testing.assert_allclose(grid, closed, rtol=5e-3)


def test_loguniform_prior_runs_and_shifts_limit():
    closed = float(h0_95_marginalized(XS, YS))
    lu = float(h0_95_marginalized(XS, YS, prior=dist.LogUniform(1e-3, 10.0)))
    assert np.isfinite(lu) and lu > 0.0
    # Log-uniform weights small amplitudes more heavily -> a different (here
    # tighter) limit than the flat prior; at minimum it must not coincide.
    assert abs(lu - closed) > 1e-3


def test_finite_uniform_box_caps_the_limit():
    # A uniform box whose upper edge sits *below* the improper-flat UL must cap
    # the limit at ~that edge (the prior truncates the posterior).
    closed = float(h0_95_marginalized(XS, YS))
    cap = 0.5 * closed
    ul = float(h0_95_marginalized(XS, YS, prior=dist.Uniform(0.0, cap)))
    assert ul <= cap + 1e-3
    assert ul < closed


def test_grid_path_is_deterministic():
    # No PRNG anywhere: repeated calls are bit-identical.
    a = float(h0_95_marginalized(XS, YS, prior=dist.LogUniform(1e-3, 10.0)))
    b = float(h0_95_marginalized(XS, YS, prior=dist.LogUniform(1e-3, 10.0)))
    assert a == b
