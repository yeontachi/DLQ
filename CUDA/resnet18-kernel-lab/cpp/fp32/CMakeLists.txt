# 실행파일을 build/fp32/ 아래에 두기
set(FP32_OUTDIR ${CMAKE_BINARY_DIR}/fp32)

set(FP32_KERNELS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernels)
set(FP32_RUNTIME_DIR ${CMAKE_CURRENT_SOURCE_DIR}/runtime)

set(FP32_INCLUDES
  ${FP32_RUNTIME_DIR}
  ${FP32_KERNELS_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# 공통 커널 목록 (실제 파일 존재해야 함)
set(KERNELS_COMMON
  ${FP32_KERNELS_DIR}/im2col.cu
  ${FP32_KERNELS_DIR}/sgemm_tiled.cu
  ${FP32_KERNELS_DIR}/bn_inference.cu
  ${FP32_KERNELS_DIR}/relu.cu
)

# ---- Step2
add_executable(step2_conv1_bn1_relu
  ${KERNELS_COMMON}
  ${FP32_RUNTIME_DIR}/infer_conv1_bn1_relu.cu
)
target_include_directories(step2_conv1_bn1_relu PRIVATE ${FP32_INCLUDES})
target_link_libraries(step2_conv1_bn1_relu PRIVATE CUDA::cudart)
set_target_properties(step2_conv1_bn1_relu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
)

# ---- Step3
add_executable(step3_layer1
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer1.cu
)
target_include_directories(step3_layer1 PRIVATE ${FP32_INCLUDES})
target_link_libraries(step3_layer1 PRIVATE CUDA::cudart)
set_target_properties(step3_layer1 PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
)
