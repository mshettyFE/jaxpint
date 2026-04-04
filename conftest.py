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

import hypothesis

hypothesis.settings.register_profile("interactive", deadline=None)
hypothesis.settings.register_profile(
    "ci", deadline=None, print_blob=True, derandomize=True
)
hypothesis.settings.register_profile(
    "fuzzing", deadline=None, print_blob=True, max_examples=1000
)
default = (
    "fuzzing"
    if os.environ.get("HYPOTHESIS_PROFILE") == "fuzzing"
    else "interactive"
)
hypothesis.settings.load_profile(default)
