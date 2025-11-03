#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANI="$ROOT/exports/resnet18/fp32"
BIN="$ROOT/build/fp32/step8_e2e"
IMDIR="$ROOT/data/imagenet_val/ILSVRC2012_img_val"
LIMIT="${LIMIT:-500}"

echo "[INFO] images=$LIMIT under $IMDIR"

# 1) Torch로 GAP 덤프 + CUDA FC 실행까지 한 번에 (fast 버전)
python "$ROOT/tools/bench_fp32_vs_torch_fast.py" \
  --manifest "$MANI" \
  --bin "$BIN" \
  --imagenet_dir "$IMDIR" \
  --limit "$LIMIT" \
  --save_logits
