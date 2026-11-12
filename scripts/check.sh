#!/usr/bin/env bash
# The project's definition of done as one command: lint, format check, tests.
# No network and no API key are needed. Used by scripts/hooks/pre-push and by hand.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== ruff check"
uv run ruff check .
echo "== ruff format --check"
uv run ruff format --check .
echo "== pytest"
uv run pytest -q
echo "all checks passed"
