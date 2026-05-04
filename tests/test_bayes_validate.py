"""Tests for jaxpint.bayes.validate: prior-dict completeness checks."""

import warnings

import pytest

from jaxpint.bayes import (
    Gaussian,
    ImproperPrior,
    PriorValidationError,
    Uniform,
    validate_priors,
)


# ===========================================================================
# Happy paths
# ===========================================================================


class TestSuccess:
    def test_exact_match(self):
        priors = {"a": Uniform(0, 1), "b": Gaussian(0, 1)}
        validate_priors(priors, ["a", "b"])  # no raise

    def test_extras_allowed_when_opted_in(self):
        # Defaults reject extras; pass allow_extras=True to silence.
        priors = {"a": Uniform(0, 1), "b": Uniform(0, 1), "c": Gaussian(0, 1)}
        validate_priors(
            priors, ["a", "b"], allow_extras=True, extras_warn=False,
        )

    def test_iterator_input(self):
        priors = {"a": Uniform(0, 1)}
        validate_priors(priors, iter(["a"]))  # iterable, not a list


# ===========================================================================
# Missing-prior detection (the load-bearing failure mode)
# ===========================================================================


class TestMissing:
    def test_single_missing(self):
        priors = {"a": Uniform(0, 1)}
        with pytest.raises(PriorValidationError, match="missing priors for 1"):
            validate_priors(priors, ["a", "b"])

    def test_multiple_missing(self):
        priors = {"a": Uniform(0, 1)}
        with pytest.raises(PriorValidationError) as exc_info:
            validate_priors(priors, ["a", "b", "c", "d"])
        msg = str(exc_info.value)
        assert "missing priors for 3" in msg
        assert "'b'" in msg
        assert "'c'" in msg
        assert "'d'" in msg

    def test_missing_list_truncated_at_10(self):
        priors = {}
        many = [f"p{i}" for i in range(15)]
        with pytest.raises(PriorValidationError) as exc_info:
            validate_priors(priors, many)
        msg = str(exc_info.value)
        assert "missing priors for 15" in msg
        assert "..." in msg

    def test_empty_priors_with_expected(self):
        with pytest.raises(PriorValidationError):
            validate_priors({}, ["x"])

    def test_empty_both_ok(self):
        validate_priors({}, [])  # vacuous — no missing, no extras


# ===========================================================================
# Extras
# ===========================================================================


class TestExtras:
    def test_extras_rejected_by_default(self):
        priors = {"a": Uniform(0, 1), "extra": Gaussian(0, 1)}
        with pytest.raises(PriorValidationError, match="unexpected priors"):
            validate_priors(priors, ["a"])

    def test_extras_warn_when_allowed(self):
        priors = {"a": Uniform(0, 1), "extra": Uniform(0, 1)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_priors(priors, ["a"], allow_extras=True, extras_warn=True)
        assert any("extra" in str(w.message) for w in caught)

    def test_extras_silent_when_explicitly_silenced(self):
        priors = {"a": Uniform(0, 1), "extra": Uniform(0, 1)}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_priors(priors, ["a"], allow_extras=True, extras_warn=False)
        assert not any("extra" in str(w.message) for w in caught)


# ===========================================================================
# Type checking on values
# ===========================================================================


class TestTypeChecks:
    def test_non_prior_value_raises(self):
        priors = {"a": Uniform(0, 1), "b": "not a prior"}
        with pytest.raises(TypeError, match="must be Prior instances"):
            validate_priors(priors, ["a", "b"])

    def test_int_value_raises(self):
        priors = {"a": 42}
        with pytest.raises(TypeError):
            validate_priors(priors, ["a"])

    def test_improper_is_a_valid_prior(self):
        priors = {"a": ImproperPrior()}
        validate_priors(priors, ["a"])  # no raise — improper still subclasses Prior
