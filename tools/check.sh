#!/usr/bin/env bash
#
# Run the CI gates locally, matching the exact `uv` extras each CI job uses.
# This is the fastest way to reproduce a red CI before pushing.
#
#   tools/check.sh            ruff + pyright + import-guard + docs (fast gates)
#   tools/check.sh --tests    also run the full --runslow pytest suite (slow)
#
# PAIRED WITH CI -- keep the *gates* below (ruff/pyright/import-guard/docs) in
# sync with the workflow `run:` steps in .github/workflows/tests.yml and
# docs.yml (sphinx). CI is the authoritative gate; this script is a best-effort
# local preflight, so if they drift, update whichever is behind. (The dependency
# *environment* is already single-sourced via `uv run --extra ...` against
# pyproject/uv.lock.)
#
# EXCEPTION: the pytest runner below is deliberately NOT identical to CI. CI runs
# on a fixed small runner where bare `-n auto` (2-4 workers) is right. Dev boxes
# are heterogeneous -- a high-core / RAM-light machine OOMs under `-n auto` (each
# xdist worker carries its own JAX/XLA state). So locally we size workers by
# available RAM and enable the memory/contention mitigations (cache clearing +
# single-threaded XLA exec). The set of tests run is the same; only the
# parallelism profile differs.
#

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

run_tests=0
[[ "${1:-}" == "--tests" ]] && run_tests=1

failures=()
step() {  # step <label> <cmd...>
  local label="$1"; shift
  printf '\n\033[1m==> %s\033[0m\n' "$label"
  if "$@"; then
    printf '\033[32m    OK  %s\033[0m\n' "$label"
  else
    printf '\033[31m    FAIL %s\033[0m\n' "$label"
    failures+=("$label")
  fi
}

step "ruff (lint)"         uv run --extra cpu --extra dev ruff check
step "ruff (format)"       uv run --extra cpu --extra dev ruff format --check
step "pyright"             uv run --extra cpu --extra dev pyright
step "import without dev"  uv run --extra cpu --extra pint python -c "import jaxpint"
step "docs (-W)"           tools/build_docs.sh
if [[ $run_tests == 1 ]]; then
  # Cap xdist workers by *available RAM* (~2 GiB/worker), not just core count:
  # a high-core / RAM-light box OOMs under bare `-n auto`. JAXPINT_TEST_CLEAR_CACHES
  # bounds per-worker memory (conftest clears the JAX cache between modules) and
  # --xla_cpu_multi_thread_eigen=false stops N workers from oversubscribing cores
  # with execution threads. Override the worker count with PYTEST_WORKERS=N.
  #
  # JAX_PLATFORMS=cpu is REQUIRED, not optional: `--extra cpu` only installs the
  # CPU jaxlib, it does not stop JAX from selecting a GPU if a CUDA jaxlib happens
  # to be present in the env. Running the suite on a (consumer) GPU under xdist
  # both oversubscribes the card and fails f64 autotuning -- force the CPU backend.
  if [[ -n "${PYTEST_WORKERS:-}" ]]; then
    workers="$PYTEST_WORKERS"
  else
    avail_gb=$(free -g | awk '/^Mem:/{print $7}')
    cores=$(nproc)
    workers=$(( avail_gb / 2 ))
    (( workers > cores - 2 )) && workers=$(( cores - 2 ))
    (( workers < 1 )) && workers=1
  fi
  step "pytest --runslow (-n $workers)" \
    env JAX_PLATFORMS=cpu JAXPINT_TEST_CLEAR_CACHES=1 \
        XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
    uv run --extra cpu --extra dev pytest --runslow -n "$workers" --dist loadscope
fi

echo
if (( ${#failures[@]} )); then
  printf '\033[31m%d check(s) FAILED: %s\033[0m\n' "${#failures[@]}" "${failures[*]}"
  exit 1
fi
printf '\033[32mAll checks passed.\033[0m\n'
