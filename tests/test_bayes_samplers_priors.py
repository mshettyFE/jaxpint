"""Tests for jaxpint.bayes.samplers.priors: PriorSpec composition + dist factories."""

from __future__ import annotations

import math
from types import SimpleNamespace

import jax.numpy as jnp
import pytest

pytest.importorskip("numpyro")  # opt-in `sampling` extra
import numpyro.distributions as dist

from jaxpint.bayes.samplers.priors import (
    PRIOR_DEFAULTS,
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


def test_cw_priors_defaults_overridable():
    # Bounds are now table-driven, so a `defaults` override flows through.
    custom = {**PRIOR_DEFAULTS, "log10_h": lambda: dist.Uniform(-20.0, -10.0)}
    spec = cw_priors(defaults=custom)
    assert float(spec.flat["cw_log10_h"].low) == -20.0
    assert float(spec.flat["cw_log10_h"].high) == -10.0


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


# ---------------------------------------------------------------------------
# LinearExp (upper-limit prior) + TruncNormal conventions
# ---------------------------------------------------------------------------


def _linear_exp_cdf_reference(x):
    """Analytic LinearExp(-18, -11) CDF, transcribed independently of the
    implementation (never calls ``d.cdf``) — the KS anchor that pins the
    full *shape* of the sampled distribution, not just its mean."""
    lo, hi = 10.0**-18.0, 10.0**-11.0
    return (10.0**x - lo) / (hi - lo)


def test_linear_exp_matches_analytic_pdf():
    """log_prob == ln(10)·10^x / (10^pmax − 10^pmin) on the support."""
    import numpy as np

    from jaxpint.bayes.samplers import LinearExp

    d = LinearExp(-18.0, -11.0)
    xs = np.linspace(-17.9, -11.1, 7)
    pdf = np.exp(np.asarray([float(d.log_prob(x)) for x in xs]))
    expected = np.log(10.0) * 10.0**xs / (10.0**-11.0 - 10.0**-18.0)
    np.testing.assert_allclose(pdf, expected, rtol=1e-12)


def test_linear_exp_normalizes_and_inverts():
    """∫pdf = 1 over the support; cdf/icdf are exact inverses; edges map."""
    import numpy as np

    from jaxpint.bayes.samplers import LinearExp

    d = LinearExp(-18.0, -11.0)
    xs = np.linspace(-18.0, -11.0, 20001)
    pdf = np.exp(np.asarray(d.log_prob(jnp.asarray(xs))))
    np.testing.assert_allclose(np.trapezoid(pdf, xs), 1.0, rtol=1e-6)

    q = np.linspace(0.0, 1.0, 101)
    x = np.asarray(d.icdf(jnp.asarray(q)))
    np.testing.assert_allclose(np.asarray(d.cdf(jnp.asarray(x))), q, atol=1e-12)
    assert float(d.icdf(0.0)) == -18.0
    np.testing.assert_allclose(float(d.icdf(1.0)), -11.0, rtol=1e-15)


def test_linear_exp_samples_are_uniform_in_amplitude():
    """The full CDF of the draws matches the analytic LinearExp CDF (KS).

    A shape test, not a moment test: pins the ``icdf``-based sampling path
    against the independently transcribed CDF, so a symmetric-but-warped
    inverse (right mean, wrong distribution) fails here.
    """
    import jax
    import numpy as np
    from scipy.stats import kstest

    from jaxpint.bayes.samplers import LinearExp

    d = LinearExp(-18.0, -11.0)
    x = np.asarray(d.sample(jax.random.PRNGKey(0), (40000,)))
    assert ((x >= -18.0) & (x <= -11.0)).all()
    res = kstest(x, _linear_exp_cdf_reference)
    assert res.pvalue > 1e-3, f"KS rejected: D={res.statistic:.4g}, p={res.pvalue:.3g}"


def test_linear_exp_nuts_recovers_prior():
    """Prior-only NUTS: the constrained-support transform Jacobian is right.

    Sampling a LinearExp site with no likelihood must reproduce the prior —
    uniform in the linear amplitude.  NUTS samples the unconstrained
    ``u = logit((x-a)/(b-a))``; numpyro adds the pushforward Jacobian
    ``|dx/du| = (b-a)·σ(u)(1-σ(u))`` for the ``biject_to(interval)``
    transform (Stan Reference Manual, "Constraint Transforms", lower/upper
    bounded scalar), keyed off the distribution's declared ``support``.
    The classic hand-rolled-distribution bug — declaring ``real`` support,
    or baking a transform into ``log_prob`` without the compensating
    Jacobian — skews the amplitude mean detectably here.
    """
    import jax
    import numpy as np
    import numpyro
    from numpyro.infer import MCMC, NUTS

    from jaxpint.bayes.samplers import LinearExp

    def model():
        numpyro.sample("x", LinearExp(-18.0, -11.0))

    mcmc = MCMC(
        NUTS(model), num_warmup=300, num_samples=1200, num_chains=1,
        progress_bar=False,
    )
    mcmc.run(jax.random.PRNGKey(1))
    x = np.asarray(mcmc.get_samples()["x"])
    assert ((x >= -18.0) & (x <= -11.0)).all()
    # Full-shape KS check against the independently transcribed CDF; NUTS
    # draws are autocorrelated (KS assumes iid), so thin before testing and
    # keep the rejection threshold loose.
    from scipy.stats import kstest

    res = kstest(x[::5], _linear_exp_cdf_reference)
    assert res.pvalue > 1e-3, f"KS rejected: D={res.statistic:.4g}, p={res.pvalue:.3g}"


def test_ul_switches_swap_amplitude_priors():
    """cw_priors(ul=True) / turnover_priors(ul=True) emit LinearExp."""
    from jaxpint.bayes.samplers import LinearExp, turnover_priors

    spec = cw_priors(ul=True)
    d = spec.flat["cw_log10_h"]
    assert isinstance(d, LinearExp)
    assert float(d.pmin) == -18.0 and float(d.pmax) == -11.0
    # Other CW entries unchanged; default form unchanged.
    assert isinstance(spec.flat["cw_cos_inc"], dist.Uniform)
    assert isinstance(cw_priors().flat["cw_log10_h"], dist.Uniform)

    to = turnover_priors(ul=True)
    assert isinstance(to.flat["gwb_log10_A"], LinearExp)
    assert isinstance(turnover_priors().flat["gwb_log10_A"], dist.Uniform)
    # UL entries exist in the defaults table.
    assert isinstance(PRIOR_DEFAULTS["rednoise_log10_A_ul"](), LinearExp)


def test_truncated_normal_is_enterprise_truncnormal_convention():
    """numpyro's TruncatedNormal == scipy/enterprise's renormalized pdf.

    Enterprise's TruncNormalPrior is scipy.stats.truncnorm — a Gaussian
    renormalized by the mass inside [low, high].  Verify numpyro follows
    the identical convention (analytically, via jax.scipy norm pdf/cdf) so
    ``dist.TruncatedNormal`` can stand in for enterprise's TruncNormal with
    no wrapper.
    """
    import numpy as np
    from jax.scipy.stats import norm

    loc, scale, low, high = 4.33, 1.2, 0.0, 7.0
    d = dist.TruncatedNormal(loc, scale, low=low, high=high)
    xs = np.linspace(0.3, 6.7, 9)
    pdf = np.exp(np.asarray([float(d.log_prob(x)) for x in xs]))
    mass = float(norm.cdf((high - loc) / scale) - norm.cdf((low - loc) / scale))
    expected = np.asarray(norm.pdf((xs - loc) / scale)) / scale / mass
    np.testing.assert_allclose(pdf, expected, rtol=1e-10)
