"""Tests for jaxpint.bayes.samplers.priors: PriorSpec composition + dist factories."""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

pytest.importorskip("numpyro")  # opt-in `sampling` extra
import numpyro.distributions as dist

from jaxpint.bayes.samplers.priors import (
    PriorResolutionError,
    PriorSpec,
    cw_priors,
    distance_priors,
    from_par_file,
    noise_priors_simple,
    resolve_priors,
    timing_marg_set,
)
from jaxpint.types import ParameterVector


# ---------------------------------------------------------------------------
# Lightweight fixtures
# ---------------------------------------------------------------------------


def _pp(names, values, *, uncertainties=None, frozen=None):
    n = len(names)
    return ParameterVector(
        values=jnp.asarray(values, dtype=float),
        frozen_mask=tuple(frozen if frozen is not None else [False] * n),
        names=tuple(names),
        units=("",) * n,
        epoch_int_values={},
        uncertainties=tuple(uncertainties) if uncertainties is not None else (),
    )


def _bundle(*pairs):
    """pairs: (name, ParameterVector) → a PulsarBundle-shaped namespace."""
    return SimpleNamespace(
        pulsar_names=tuple(n for n, _ in pairs),
        pulsar_params_list=tuple(p for _, p in pairs),
    )


@pytest.fixture
def two_pulsars():
    # J1 has PX with an uncertainty; J2 has no PX.
    pp1 = _pp(
        ("F0", "F1", "PX"),
        [100.0, -1e-15, 0.973],
        uncertainties=(math.nan, math.nan, 0.20),
    )
    pp2 = _pp(("F0", "F1"), [250.0, -2e-15])
    return _bundle(("J1", pp1), ("J2", pp2))


# ---------------------------------------------------------------------------
# noise_priors_simple
# ---------------------------------------------------------------------------


def test_noise_priors_simple_emits_dists(two_pulsars):
    spec = noise_priors_simple(two_pulsars)
    assert isinstance(spec, PriorSpec)
    # White noise + red noise per pulsar, keyed by FQN.
    for p in ("J1", "J2"):
        assert isinstance(spec.flat[f"{p}_efac"], dist.Uniform)
        assert float(spec.flat[f"{p}_efac"].low) == 0.1
        assert float(spec.flat[f"{p}_efac"].high) == 10.0
        assert isinstance(spec.flat[f"{p}_rednoise_log10_A"], dist.Uniform)
        assert float(spec.flat[f"{p}_rednoise_log10_A"].low) == -20.0


def test_noise_priors_no_red(two_pulsars):
    spec = noise_priors_simple(two_pulsars, include_red_noise=False)
    assert not any("rednoise" in k for k in spec.flat)


# ---------------------------------------------------------------------------
# Composition precedence
# ---------------------------------------------------------------------------


def test_composition_last_wins(two_pulsars):
    spec = noise_priors_simple(two_pulsars) | {"J1_efac": dist.Uniform(0.5, 2.0)}
    assert float(spec.flat["J1_efac"].low) == 0.5  # override won
    assert float(spec.flat["J2_efac"].low) == 0.1  # untouched


def test_dict_on_left_ror(two_pulsars):
    # {fqn: dist} | PriorSpec must work via __ror__.
    spec = {"J1_efac": dist.Uniform(0.5, 2.0)} | noise_priors_simple(two_pulsars)
    # PriorSpec (RHS) wins here, since RHS overrides LHS per the | contract.
    assert float(spec.flat["J1_efac"].low) == 0.1


def test_owned_names(two_pulsars):
    spec = noise_priors_simple(two_pulsars)
    assert "J1_efac" in spec.owned_names()
    assert spec.owned_names() == set(spec.flat)


# ---------------------------------------------------------------------------
# distance_priors — per-instance Normal from par-file uncertainty
# ---------------------------------------------------------------------------


def test_distance_priors_par_file_gaussian(two_pulsars):
    spec = distance_priors(two_pulsars)  # prior=None → par-file Normal
    px = spec.flat["J1_PX"]
    assert isinstance(px, dist.Normal)
    assert float(px.loc) == pytest.approx(0.973)
    assert float(px.scale) == pytest.approx(0.20)
    # J2 has no PX → silently skipped.
    assert "J2_PX" not in spec.flat


def test_distance_priors_n_sigma_widens(two_pulsars):
    spec = distance_priors(two_pulsars, n_sigma=3.0)
    assert float(spec.flat["J1_PX"].scale) == pytest.approx(0.60)


def test_distance_priors_explicit_dist(two_pulsars):
    spec = distance_priors(two_pulsars, prior=dist.Uniform(0.1, 5.0))
    assert isinstance(spec.flat["J1_PX"], dist.Uniform)


def test_distance_priors_missing_uncert_raises():
    pp = _pp(("PX",), [1.0])  # uncertainties default → NaN
    with pytest.raises(ValueError, match="uncertainty"):
        distance_priors(_bundle(("J1", pp)))


# ---------------------------------------------------------------------------
# from_par_file
# ---------------------------------------------------------------------------


def test_from_par_file(two_pulsars):
    spec = from_par_file(two_pulsars, {"J1": {"PX": (0.9, 0.1)}})
    assert float(spec.flat["J1_PX"].loc) == pytest.approx(0.9)
    assert float(spec.flat["J1_PX"].scale) == pytest.approx(0.1)


def test_from_par_file_unknown_pulsar(two_pulsars):
    with pytest.raises(KeyError, match="not in"):
        from_par_file(two_pulsars, {"J999": {"PX": (1.0, 0.1)}})


# ---------------------------------------------------------------------------
# cw_priors
# ---------------------------------------------------------------------------


def test_cw_priors():
    spec = cw_priors()
    assert set(spec.flat) == {
        "cw_log10_h", "cw_log10_fgw", "cw_cos_gwtheta", "cw_gwphi",
        "cw_cos_inc", "cw_psi", "cw_phase0",
    }
    assert float(spec.flat["cw_gwphi"].high) == pytest.approx(2 * math.pi)
    assert float(spec.flat["cw_cos_inc"].low) == -1.0


# ---------------------------------------------------------------------------
# timing_marg_set
# ---------------------------------------------------------------------------


def test_timing_marg_set_all_free(two_pulsars):
    over = timing_marg_set(two_pulsars)
    assert over == {"J1_F0", "J1_F1", "J1_PX", "J2_F0", "J2_F1"}


def test_timing_marg_set_only_filter(two_pulsars):
    over = timing_marg_set(two_pulsars, only={"F0", "F1"})
    assert over == {"J1_F0", "J1_F1", "J2_F0", "J2_F1"}


# ---------------------------------------------------------------------------
# resolve_priors — partition enforcement
# ---------------------------------------------------------------------------


def test_resolve_priors_success(two_pulsars):
    spec = noise_priors_simple(two_pulsars) | cw_priors()
    free = ["J1_efac", "J2_efac", "cw_log10_h"]
    out = resolve_priors(free, spec)
    assert set(out) == set(free)
    assert all(isinstance(d, dist.Distribution) for d in out.values())


def test_resolve_priors_missing_raises(two_pulsars):
    spec = noise_priors_simple(two_pulsars)
    with pytest.raises(PriorResolutionError, match="no prior assigned"):
        resolve_priors(["J1_efac", "cw_log10_h"], spec)  # cw not in spec


def test_resolve_priors_accepts_bare_dict():
    out = resolve_priors(["a"], {"a": dist.Normal(0.0, 1.0)})
    assert isinstance(out["a"], dist.Normal)
