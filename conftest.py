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

# Test-time runtime shape checking, scoped to the JAX-array core.
# jaxtyping shape annotations are documentation unless a runtime typechecker is
# installed; this import hook wraps the listed modules' functions with beartype
# so their Float[Array, "..."] shapes (and shared dim names across args) are
# verified whenever a test exercises them.  Must run before those modules are
# first imported (conftest loads before any test module).
#
# Scoped to ``jaxpint.utils`` -- the pure-JAX-array numerical core, where the
# annotations are exact and beartype runs clean.  Broadening package-wide was
# tried and is NOT currently viable: beartype surfaces pervasive *annotation
# imprecision* (not shape bugs) across the rest of the codebase, e.g.
#   * ``: float`` params that receive a JAX tracer under jax.grad/jit (binary),
#   * ``Float[Array, ...]`` returns that are actually NumPy arrays (noise ECORR),
#   * ``np.ndarray`` annotations standing in for ``np.float64`` scalars (bridge),
#   * int literals passed to ``: float`` Prior constructors -- ``Uniform(0, 1)``
#     -- (bayes; beartype doesn't apply the PEP 484 numeric tower by default).
# All are harmless at runtime but rejected by strict checking, so expanding the
# scope is a deliberate annotation-cleanup project, not a one-line change.
from jaxtyping import install_import_hook

install_import_hook("jaxpint.utils", "beartype.beartype")

import pytest
import hypothesis

# Most tests are PINT-parity tests.  PINT is an optional dependency, so rather
# than hand-mark ~50 files, auto-detect any test module that imports PINT and
# (a) tag it ``requires_pint`` for selection and (b) skip it cleanly when PINT
# is not installed instead of erroring on a module/function-level ``import pint``.
#
# We deliberately match only ``import pint`` / ``from pint`` statements -- NOT
# ``pytest.importorskip("pint")``.  importorskip already skips precisely the one
# test (or fixture) that calls it, so escalating it to a module-wide skip would
# wrongly skip PINT-free sibling tests in mixed modules (e.g. the meta-tests in
# test_public_api.py / test_registry_consistency.py and the pure-JAX tests in
# test_dual_float.py).  Modules whose every test needs PINT still carry a real
# ``import pint`` statement (often function-level) and are matched by the rule below.
_HAS_PINT = importlib.util.find_spec("pint") is not None
_PINT_RE = re.compile(r"^\s*(?:import\s+pint|from\s+pint)", re.M)

# Reason string for the no-PINT skip.  Shared between the skip marker and the
# end-of-run banner (pytest_terminal_summary) so the two can't drift apart.
_SKIP_NO_PINT_REASON = "requires PINT (pip install jaxpint[pint])"


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
    skip_no_pint = pytest.mark.skip(reason=_SKIP_NO_PINT_REASON)
    for item in items:
        if not run_slow and "slow" in item.keywords:
            item.add_marker(skip_slow)
        if _module_uses_pint(str(item.fspath)):
            item.add_marker(pytest.mark.requires_pint)  # denote for -m selection
            if not _HAS_PINT:
                item.add_marker(skip_no_pint)  # graceful skip, not a hard error


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """End a PINT-free run with a loud, un-missable banner.

    The PINT-parity tests are the behavior-preservation suite -- they pin
    JaxPINT's numbers against the reference implementation.  When PINT is
    absent they skip gracefully (not error), but a large ``N skipped`` count
    is easy to scroll past, letting a green run masquerade as full validation.
    This counts the no-PINT skips and warns prominently as the last thing
    printed.  (To make CI *fail* instead of skip, gate the marker above on an
    env var; this hook only reports.)
    """
    if _HAS_PINT:
        return
    n = 0
    for report in terminalreporter.stats.get("skipped", []):
        longrepr = report.longrepr
        # A skip's longrepr is a (path, lineno, reason) tuple; the reason reads
        # "Skipped: <reason>".  Fall back to str() for any other shape.
        reason = (
            longrepr[2]
            if isinstance(longrepr, tuple) and len(longrepr) == 3
            else str(longrepr or "")
        )
        if _SKIP_NO_PINT_REASON in reason:
            n += 1
    if not n:
        return
    terminalreporter.write_sep("=", "PINT NOT INSTALLED", red=True, bold=True)
    terminalreporter.write_line(
        f"{n} PINT-parity test(s) were skipped because PINT is not installed. "
        "Behavior parity vs. the reference implementation is UNVERIFIED.",
        red=True,
    )
    terminalreporter.write_line(
        "Install the optional dependency to run them:  pip install jaxpint[pint]",
        red=True,
    )


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


@pytest.fixture(autouse=True, scope="module")
def _bound_jax_memory():
    """Optionally drop JAX's compilation cache after each test module.

    JAX/XLA caches one compiled executable per distinct jitted signature and
    never evicts them, so a long pytest session's memory grows monotonically.
    That's fine serially, but under ``pytest -n N`` every worker carries its own
    copy of that growth, which can OOM a memory-constrained machine.

    Setting ``JAXPINT_TEST_CLEAR_CACHES=1`` clears the cache at each module
    boundary, bounding per-worker memory to ~one module's compilations so more
    workers fit in RAM.  Off by default -- normal/CI runs pay no recompilation
    overhead.
    """
    yield
    if os.environ.get("JAXPINT_TEST_CLEAR_CACHES"):
        import jax

        jax.clear_caches()


@pytest.fixture
def _pinned_clock(monkeypatch):
    """Pin both JaxPINT and PINT to the seed clock snapshot.

    Sets ``JAXPINT_CLOCK_REF`` to the committed seed ref, ensures that
    snapshot is present, and points PINT at the same directory via
    ``PINT_CLOCK_OVERRIDE`` so native-vs-PINT parity tests use identical
    clock corrections.  Shared by all native/parity test modules.
    """
    from jaxpint.clock import SEED_CLOCK_REF, clock_dir, ensure_fresh

    monkeypatch.setenv("JAXPINT_CLOCK_REF", SEED_CLOCK_REF)
    ensure_fresh(force=True)
    monkeypatch.setenv("PINT_CLOCK_OVERRIDE", str(clock_dir()))
