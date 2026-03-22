#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"

"$ROOT_DIR/scripts/build.sh" "$BUILD_DIR"

if [[ -n "${CTEST_BIN:-}" ]]; then
  CTEST_CMD="$CTEST_BIN"
elif command -v ctest >/dev/null 2>&1; then
  CTEST_CMD="$(command -v ctest)"
elif [[ -x "$HOME/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/ctest" ]]; then
  CTEST_CMD="$HOME/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/ctest"
else
  echo "ctest not found. Set CTEST_BIN or install CTest." >&2
  exit 1
fi

"$CTEST_CMD" --test-dir "$BUILD_DIR" --output-on-failure
