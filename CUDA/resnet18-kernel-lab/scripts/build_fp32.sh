#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"

echo "[CONFIGURE] cmake -S cpp -B build (Release)"
cmake -S "$ROOT/cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release

echo "[BUILD] cmake --build build -j"
cmake --build "$BUILD" -j
