#!/bin/bash
set -e

# 1) 빌드
./scripts/build_fp32.sh

# 2) 실행
#   NOTE: 빌드 결과 바이너리가 어디에 생성되는지에 따라 경로 조정 필요.
#   예: build/fp32/infer_stem_block0 또는 build/infer_stem_block0 등
#   Step2에서 run_step2.sh가 어떻게 호출했는지 그대로 따라가.
#
#   아래는 예시 (빌드 결과가 build/fp32/ 아래에 있다고 가정):

BUILD_DIR=build/fp32

$BUILD_DIR/infer_stem_block0 \
    --manifest exports/resnet18/fp32 \
    --input exports/resnet18/fp32/sample_input.bin \
    --expect exports/resnet18/fp32/sample_block0_out.bin
