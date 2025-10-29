#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$ROOT/build/fp32/step3_layer1"
MANI="$ROOT/exports/resnet18/fp32"

if [[ ! -x "$BIN" ]]; then
  echo "Error: $BIN not found. Build first: bash scripts/build_fp32.sh"
  exit 1
fi

"$BIN" --manifest "$MANI"


# 실행 + 로그 저장
{
  echo "[Run] $(date)"
  echo "[Cmd] $BUILD --manifest $MANI --input $MANI/fixtures/input.bin --expect $MANI/fixtures/expected.bin"
  "$BUILD" --manifest "$MANI" --input "$MANI/fixtures/input.bin" --expect "$MANI/fixtures/expected.bin"
} | tee "$LOG"

echo "Saved log -> $LOG"