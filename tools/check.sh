#!/usr/bin/env bash
#
# Run the CI gates locally, matching the exact `uv` extras each CI job uses.
# This is the fastest way to reproduce a red CI before pushing.
#
#   tools/check.sh            ruff + pyright + import-guard + docs (fast gates)
#   tools/check.sh --tests    also run the full --runslow pytest suite (slow)
#
# PAIRED WITH CI -- keep the checks below in sync with the workflow `run:` steps
# in .github/workflows/tests.yml (ruff/pyright/pytest) and docs.yml (sphinx).
# CI is the authoritative gate; this script is a best-effort local preflight, so
# if they drift, update whichever is behind. (The dependency *environment* is
# already single-sourced via `uv run --extra ...` against pyproject/uv.lock.)
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

docs_build() {
  rm -rf docs/api/generated docs/_build
  JAX_PLATFORMS=cpu uv run --extra cpu --extra docs --extra pint \
    sphinx-build -W --keep-going -b html docs docs/_build/html
}

# NOTE: CI gates lint via `ruff check` (below); code *formatting* is enforced by
# the pre-commit ruff-format hook (auto-fixed on commit), not the CI workflow, so
# it is intentionally not a gate here. Run `ruff format` yourself for a one-shot.
step "ruff (lint)"         uv run --extra cpu --extra dev ruff check
step "pyright"             uv run --extra cpu --extra dev pyright
step "import without dev"  uv run --extra cpu --extra pint python -c "import jaxpint"
step "docs (-W)"           docs_build
if [[ $run_tests == 1 ]]; then
  step "pytest --runslow"  uv run --extra cpu --extra dev pytest --runslow -n auto
fi

echo
if (( ${#failures[@]} )); then
  printf '\033[31m%d check(s) FAILED: %s\033[0m\n' "${#failures[@]}" "${failures[*]}"
  exit 1
fi
printf '\033[32mAll checks passed.\033[0m\n'
