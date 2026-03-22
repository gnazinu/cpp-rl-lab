#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"

if [[ -n "${CMAKE_BIN:-}" ]]; then
  CMAKE_CMD="$CMAKE_BIN"
elif command -v cmake >/dev/null 2>&1; then
  CMAKE_CMD="$(command -v cmake)"
elif [[ -x "$HOME/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake" ]]; then
  CMAKE_CMD="$HOME/.local/share/JetBrains/Toolbox/apps/clion/bin/cmake/linux/x64/bin/cmake"
else
  echo "cmake not found. Set CMAKE_BIN or install CMake." >&2
  exit 1
fi

"$CMAKE_CMD" -S "$ROOT_DIR" -B "$BUILD_DIR" -G Ninja
"$CMAKE_CMD" --build "$BUILD_DIR"
