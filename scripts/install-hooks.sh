#!/usr/bin/env bash
# Point git at the versioned hooks in scripts/hooks (nothing is copied into .git/hooks).
# The setting is per clone, so run this once after cloning.
set -euo pipefail

cd "$(dirname "$0")/.."
chmod +x scripts/check.sh scripts/hooks/pre-push
git config core.hooksPath scripts/hooks
echo "hooks installed: git will run scripts/hooks/pre-push before every push"
