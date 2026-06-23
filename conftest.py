"""Pytest configuration for JaxPINT tests.

Registers Hypothesis profiles for different execution contexts:

- **interactive** (default): No deadline, for local development.
- **ci**: Deterministic (``derandomize=True``) with ``print_blob`` for
  reproducing failures in CI.
- **fuzzing**: 1000 examples per test, for thorough property-based testing.
  Activated by setting ``HYPOTHESIS_PROFILE=fuzzing``.
"""

import os

import jax
jax.config.update("jax_enable_x64", True)

import pytest
import hypothesis


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

hypothesis.settings.register_profile("interactive", deadline=None)
hypothesis.settings.register_profile(
    "ci", deadline=None, print_blob=True, derandomize=True
)
hypothesis.settings.register_profile(
    "fuzzing", deadline=None, print_blob=True, max_examples=1000
)
_VALID_PROFILES = {"interactive", "ci", "fuzzing"}
_requested = os.environ.get("HYPOTHESIS_PROFILE", "interactive")
default = _requested if _requested in _VALID_PROFILES else "interactive"
hypothesis.settings.load_profile(default)
