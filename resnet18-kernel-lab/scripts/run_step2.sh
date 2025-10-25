#!/usr/bin/env bash
set -euo pipefail

BUILD=build/fp32/step2_conv1_bn1_relu   # ← 여기 수정
MANI=exports/resnet18/fp32

if [[ ! -x "$BUILD" ]]; then
  echo "Error: $BUILD not found or not executable. Did you build?"
  echo "Try: bash scripts/build_fp32.sh"
  exit 1
fi

"$BUILD" --manifest "$MANI" \
         --input   "$MANI/fixtures/input.bin" \
         --expect  "$MANI/fixtures/expected.bin"
