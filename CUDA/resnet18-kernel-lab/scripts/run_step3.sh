#!/usr/bin/env bash
set -euo pipefail

# === Build target and paths ===
BUILD=build/fp32/infer_stem_block0
MANI=exports/resnet18/fp32
INPUT=$MANI/sample_input.bin
EXPECT=$MANI/sample_block0_out.bin

mkdir -p logs
LOG=logs/step3_$(date +'%Y%m%d_%H%M%S').log

# === Check build ===
if [[ ! -x "$BUILD" ]]; then
  echo "Error: $BUILD not found or not executable. Did you build?"
  echo "Try: bash scripts/build_fp32.sh"
  exit 1
fi

# === Run + log ===
{
  echo "[Run Step3] $(date)"
  echo "[Cmd] $BUILD --manifest $MANI --input $INPUT --expect $EXPECT"
  "$BUILD" --manifest "$MANI" --input "$INPUT" --expect "$EXPECT"
} | tee "$LOG"

echo "Saved log -> $LOG"
