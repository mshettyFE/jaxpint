#!/usr/bin/env bash
#
# Build the Sphinx docs exactly the way CI does (.github/workflows/docs.yml):
# same extras, -W (warnings are errors), and a cleared docs/api/generated so
# stale autosummary stubs can't mask a break. Shared by tools/check.sh and the
# pre-push hook in .pre-commit-config.yaml -- edit here, not in the callers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

rm -rf docs/api/generated docs/_build
JAX_PLATFORMS=cpu exec uv run --extra cpu --extra docs --extra pint \
  sphinx-build -W --keep-going -b html docs docs/_build/html
