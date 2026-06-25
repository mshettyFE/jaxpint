"""Pytest configuration for JaxPINT tests.

Registers Hypothesis profiles for different execution contexts:

- **interactive** (default): No deadline, for local development.
- **ci**: Deterministic (``derandomize=True``) with ``print_blob`` for
  reproducing failures in CI.
- **fuzzing**: 1000 examples per test, for thorough property-based testing.
  Activated by setting ``HYPOTHESIS_PROFILE=fuzzing``.
"""

import functools
import importlib.util
import os
import re

import jax

jax.config.update("jax_enable_x64", True)

import pytest
import hypothesis

# Most tests are PINT-parity tests.  PINT is an optional dependency, so rather
# than hand-mark ~50 files, auto-detect any test module that imports PINT and
# (a) tag it ``requires_pint`` for selection and (b) skip it cleanly when PINT
# is not installed instead of erroring on a function-level ``import pint``.
_HAS_PINT = importlib.util.find_spec("pint") is not None
_PINT_RE = re.compile(
    r"""^\s*(?:import\s+pint|from\s+pint)|importorskip\(\s*["']pint""", re.M
)


@functools.lru_cache(maxsize=None)
def _module_uses_pint(path: str) -> bool:
    try:
        return bool(_PINT_RE.search(open(path, encoding="utf-8").read()))
    except OSError:
        return False


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_pint: test imports PINT (auto-detected per module; skipped "
        "when PINT is not installed).",
    )


def pytest_collection_modifyitems(config, items):
    run_slow = config.getoption("--runslow")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_no_pint = pytest.mark.skip(reason="requires PINT (pip install jaxpint[pint])")
    for item in items:
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if _module_uses_pint(str(item.fspath)):
            item.add_marker(pytest.mark.requires_pint)  # denote for -m selection
            if not _HAS_PINT:
                item.add_marker(skip_no_pint)  # graceful skip, not a hard error


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
