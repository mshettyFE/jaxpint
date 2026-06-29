"""Tests for jaxpint.bayes.defaults: bulk-prior factories."""

from collections import namedtuple

import jax.numpy as jnp
import pytest

from jaxpint.bayes import (
    NANOGRAV_NOISE_DEFAULTS,
    Gaussian,
    ImproperPrior,
    Uniform,
    cw_phi_psr_priors,
    cw_priors,
    distance_priors,
    from_par_file,
    noise_priors_simple,
    timing_priors,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures: minimal fake "pulsar" objects with the shape that the
# helpers expect (mirrors NanogravPTA's namedtuple structure).
# ---------------------------------------------------------------------------


class FakeParams:
    """Stand-in for ParameterVector with controllable .names + values + uncerts."""

    def __init__(self, names, values, uncerts=None):
        self.names = tuple(names)
        self._values = dict(zip(names, values))
        self._uncerts = dict(zip(names, uncerts)) if uncerts else {}

    def param_value(self, name):
        return self._values[name]

    def param_uncert(self, name):
        if name not in self._uncerts:
            raise KeyError(name)
        return self._uncerts[name]


FakePTA = namedtuple("FakePTA", ["pulsar_names", "pulsar_params_list"])


@pytest.fixture
def two_pulsar_pta():
    pps = [
        FakeParams(
            names=("F0", "F1", "PX"),
            values=(186.0, -1e-15, 0.973),
            uncerts=(1e-12, 1e-18, 0.20),
        ),
        FakeParams(
            names=("F0", "F1", "PX"),
            values=(218.0, -2e-15, 3.042),
            uncerts=(2e-12, 2e-18, 0.10),
        ),
    ]
    return FakePTA(pulsar_names=("J0023", "J0030"), pulsar_params_list=tuple(pps))


@pytest.fixture
def pta_no_px():
    pps = [
        FakeParams(names=("F0", "F1"), values=(186.0, -1e-15)),
        FakeParams(names=("F0", "F1", "PX"), values=(218.0, -2e-15, 3.042),
                   uncerts=(0, 0, 0.10)),
    ]
    return FakePTA(pulsar_names=("noPX", "withPX"), pulsar_params_list=tuple(pps))


# ===========================================================================
# timing_priors
# ===========================================================================


class TestTimingPriors:
    def test_default_is_improper(self, two_pulsar_pta):
        priors = timing_priors(two_pulsar_pta)
        assert "J0023_F0" in priors
        assert "J0030_PX" in priors
        for v in priors.values():
            assert isinstance(v, ImproperPrior)

    def test_custom_prior(self, two_pulsar_pta):
        priors = timing_priors(two_pulsar_pta, prior=Uniform(-1, 1))
        for v in priors.values():
            assert isinstance(v, Uniform)
            assert v.low == -1 and v.high == 1

    def test_all_params_covered(self, two_pulsar_pta):
        priors = timing_priors(two_pulsar_pta)
        # 2 pulsars * 3 params each
        assert len(priors) == 6


# ===========================================================================
# distance_priors
# ===========================================================================


class TestDistancePriors:
    def test_par_file_gaussian_default(self, two_pulsar_pta):
        priors = distance_priors(two_pulsar_pta)
        assert isinstance(priors["J0023_PX"], Gaussian)
        assert priors["J0023_PX"].mu == pytest.approx(0.973)
        # Default n_sigma=1.0 → sigma matches par-file uncertainty.
        assert priors["J0023_PX"].sigma == pytest.approx(0.20)
        assert priors["J0030_PX"].sigma == pytest.approx(0.10)

    def test_n_sigma_override(self, two_pulsar_pta):
        priors = distance_priors(two_pulsar_pta, n_sigma=10.0)
        assert priors["J0023_PX"].sigma == pytest.approx(2.0)
        assert priors["J0030_PX"].sigma == pytest.approx(1.0)

    def test_uniform_override_for_all(self, two_pulsar_pta):
        priors = distance_priors(two_pulsar_pta, prior=Uniform(0.1, 100))
        for v in priors.values():
            assert isinstance(v, Uniform)
            assert v.low == 0.1

    def test_improper_override(self, two_pulsar_pta):
        priors = distance_priors(two_pulsar_pta, prior=ImproperPrior())
        for v in priors.values():
            assert isinstance(v, ImproperPrior)

    def test_callable_per_pulsar(self, two_pulsar_pta):
        priors = distance_priors(
            two_pulsar_pta,
            prior=lambda pp: Gaussian(mu=pp.param_value("PX"), sigma=0.5),
        )
        assert priors["J0023_PX"].mu == pytest.approx(0.973)
        assert priors["J0023_PX"].sigma == pytest.approx(0.5)
        assert priors["J0030_PX"].mu == pytest.approx(3.042)

    def test_pulsar_without_px_skipped(self, pta_no_px):
        priors = distance_priors(pta_no_px)
        assert "noPX_PX" not in priors
        assert "withPX_PX" in priors

    def test_zero_sigma_raises(self, two_pulsar_pta):
        # Construct a pulsar with zero PX uncertainty
        pp = FakeParams(names=("PX",), values=(1.0,), uncerts=(0.0,))
        bad = FakePTA(("X",), (pp,))
        with pytest.raises(ValueError, match="unusable"):
            distance_priors(bad)

    def test_no_uncert_raises_with_helpful_message(self):
        pp = FakeParams(names=("PX",), values=(1.0,))   # no uncerts
        bad = FakePTA(("X",), (pp,))
        with pytest.raises(ValueError, match="from_par_file"):
            distance_priors(bad)


# ===========================================================================
# from_par_file
# ===========================================================================


class TestFromParFile:
    def test_explicit_mapping(self, two_pulsar_pta):
        values = {
            "J0023": {"PX": (0.973, 0.05), "F0": (186.0, 1e-12)},
            "J0030": {"PX": (3.042, 0.10)},
        }
        priors = from_par_file(two_pulsar_pta, values)
        assert priors["J0023_PX"].mu == pytest.approx(0.973)
        assert priors["J0023_PX"].sigma == pytest.approx(0.05)
        assert priors["J0023_F0"].sigma == pytest.approx(1e-12)
        assert priors["J0030_PX"].mu == pytest.approx(3.042)

    def test_n_sigma_scaling(self, two_pulsar_pta):
        values = {"J0023": {"PX": (0.973, 0.05)}}
        priors = from_par_file(two_pulsar_pta, values, n_sigma=3.0)
        assert priors["J0023_PX"].sigma == pytest.approx(0.15)

    def test_unknown_pulsar_raises(self, two_pulsar_pta):
        values = {"J9999": {"PX": (1.0, 0.1)}}
        with pytest.raises(KeyError, match="J9999"):
            from_par_file(two_pulsar_pta, values)

    def test_bad_sigma_raises(self, two_pulsar_pta):
        values = {"J0023": {"PX": (1.0, 0.0)}}
        with pytest.raises(ValueError, match="bad sigma"):
            from_par_file(two_pulsar_pta, values)


# ===========================================================================
# cw_priors / cw_phi_psr_priors
# ===========================================================================


class TestCwPriors:
    def test_seven_canonical_params(self):
        priors = cw_priors()
        expected = {
            "cw_log10_h", "cw_log10_fgw", "cw_cos_gwtheta", "cw_gwphi",
            "cw_cos_inc", "cw_psi", "cw_phase0",
        }
        assert set(priors.keys()) == expected

    def test_no_phi_psr_by_default(self):
        priors = cw_priors()
        assert not any("phi_psr" in k for k in priors)

    def test_prefix_override(self):
        priors = cw_priors(prefix="src1_")
        assert all(k.startswith("src1_") for k in priors)
        assert "src1_log10_h" in priors

    def test_all_uniform(self):
        priors = cw_priors()
        for v in priors.values():
            assert isinstance(v, Uniform)


class TestCwPhiPsrPriors:
    def test_one_per_pulsar(self, two_pulsar_pta):
        priors = cw_phi_psr_priors(two_pulsar_pta)
        assert set(priors.keys()) == {"J0023_cw_phi_psr", "J0030_cw_phi_psr"}

    def test_uniform_zero_to_2pi(self, two_pulsar_pta):
        priors = cw_phi_psr_priors(two_pulsar_pta)
        for v in priors.values():
            assert isinstance(v, Uniform)
            assert v.low == 0.0
            assert v.high == pytest.approx(2 * jnp.pi)


# ===========================================================================
# noise_priors_simple
# ===========================================================================


class TestNoisePriorsSimple:
    def test_default_suffixes(self, two_pulsar_pta):
        priors = noise_priors_simple(two_pulsar_pta)
        for psr in ("J0023", "J0030"):
            for s in ("efac", "t2equad", "log10_ecorr",
                      "rednoise_log10_A", "rednoise_gamma"):
                assert f"{psr}_{s}" in priors

    def test_no_red_noise(self, two_pulsar_pta):
        priors = noise_priors_simple(two_pulsar_pta, include_red_noise=False)
        assert "J0023_rednoise_log10_A" not in priors
        assert "J0023_efac" in priors

    def test_uses_nanograv_defaults(self, two_pulsar_pta):
        priors = noise_priors_simple(two_pulsar_pta)
        assert priors["J0023_efac"] is NANOGRAV_NOISE_DEFAULTS["efac"]


# ===========================================================================
# Composition (the user-facing pattern)
# ===========================================================================


class TestComposition:
    def test_dict_union_overrides(self, two_pulsar_pta):
        priors = (
            timing_priors(two_pulsar_pta, prior=ImproperPrior())
            | distance_priors(two_pulsar_pta)
        )
        # PX overridden by distance_priors (Gaussian), other params still improper
        assert isinstance(priors["J0023_PX"], Gaussian)
        assert isinstance(priors["J0023_F0"], ImproperPrior)
        assert isinstance(priors["J0023_F1"], ImproperPrior)

    def test_full_nanograv_style_composition(self, two_pulsar_pta):
        priors = (
            timing_priors(two_pulsar_pta, prior=ImproperPrior())
            | noise_priors_simple(two_pulsar_pta)
            | distance_priors(two_pulsar_pta)
            | cw_priors()
            | {"crn_log10_A": Uniform(-18, -11)}
        )
        # Spot-check coverage
        assert isinstance(priors["J0023_F0"], ImproperPrior)
        assert isinstance(priors["J0023_PX"], Gaussian)
        assert isinstance(priors["J0023_efac"], Uniform)
        assert isinstance(priors["cw_log10_h"], Uniform)
        assert priors["crn_log10_A"].low == -18
