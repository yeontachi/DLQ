#!/bin/bash
set -e

# -----------------------------------------------------------------------------
# Build configuration
# -----------------------------------------------------------------------------
BUILD_DIR=build/fp32
SRC_DIR=cpp/fp32

# 선택: GPU 아키텍처 (너 GPU가 RTX 40시리즈면 sm_89, RTX 30 시리즈면 sm_86, A100이면 sm_80)
ARCH=sm_80

# -----------------------------------------------------------------------------
# Clean + prepare build folder
# -----------------------------------------------------------------------------
if [ "$1" == "clean" ]; then
    echo "[CLEAN] Removing $BUILD_DIR"
    rm -rf $BUILD_DIR
    exit 0
fi

mkdir -p $BUILD_DIR

# -----------------------------------------------------------------------------
# CMake configure
# -----------------------------------------------------------------------------
echo "[CONFIGURE] CMake configuring to $BUILD_DIR"
cmake -S $SRC_DIR -B $BUILD_DIR \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=${ARCH#sm_} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# -----------------------------------------------------------------------------
# Build (with parallel jobs)
# -----------------------------------------------------------------------------
echo "[BUILD] Compiling fp32 targets..."
cmake --build $BUILD_DIR --target infer_conv1_bn1_relu infer_stem_block0 -j$(nproc) --verbose

echo ""
echo "Build complete!"
echo "Executables:"
echo "  - $BUILD_DIR/infer_conv1_bn1_relu"
echo "  - $BUILD_DIR/infer_stem_block0"
echo ""
echo "Run example:"
echo "  ./scripts/run_step3.sh"
