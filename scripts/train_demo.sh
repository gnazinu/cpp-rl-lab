#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$ROOT_DIR/build}"

"$ROOT_DIR/scripts/build.sh" "$BUILD_DIR"

"$BUILD_DIR/cpp_rl_lab" train \
  --maze "$ROOT_DIR/configs/mazes/basic.txt" \
  --episodes 1500 \
  --seed 42 \
  --output-dir "$ROOT_DIR/outputs/demo_train"
