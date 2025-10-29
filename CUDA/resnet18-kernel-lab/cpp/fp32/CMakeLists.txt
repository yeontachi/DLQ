# 실행파일이 build/fp32/ 아래로 나오게
set(FP32_OUTDIR ${CMAKE_BINARY_DIR}/fp32)

set(FP32_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(FP32_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

set(FP32_INCLUDES
  ${FP32_RUNTIME_DIR}
  ${FP32_KERNELS_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# ===== 공통 커널(실제 파일이 존재해야 함) =====
set(KERNELS_COMMON
  ${FP32_KERNELS_DIR}/im2col.cu
  ${FP32_KERNELS_DIR}/sgemm_tiled.cu
  ${FP32_KERNELS_DIR}/bn_inference.cu
  ${FP32_KERNELS_DIR}/relu.cu
)

# ===== Step2 실행파일 =====
add_executable(step2_conv1_bn1_relu
  ${KERNELS_COMMON}
  ${FP32_RUNTIME_DIR}/infer_conv1_bn1_relu.cu  # 확장자 .cu (커널 런치가 있어서)
)

target_include_directories(step2_conv1_bn1_relu PRIVATE ${FP32_INCLUDES})
target_link_libraries(step2_conv1_bn1_relu PRIVATE CUDA::cudart)
set_target_properties(step2_conv1_bn1_relu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
)

message(STATUS "FP32 binaries -> ${FP32_OUTDIR}")
