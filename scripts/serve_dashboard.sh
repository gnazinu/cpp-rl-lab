#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-$ROOT_DIR/outputs}"
PORT="${PORT:-8000}"

cd "$TARGET_DIR"
echo "Serving dashboard files from $TARGET_DIR at http://127.0.0.1:$PORT"
python3 -m http.server "$PORT"
