# 소스
set(SRCS
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  runtime/infer_conv1_bn1_relu.cpp
)

add_executable(step2_conv1_bn1_relu ${SRCS})

# CUDA 헤더 경로를 타겟에 연결
target_include_directories(step2_conv1_bn1_relu
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}           # <-- /usr/local/cuda/include 등
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

# CUDA 런타임 라이브러리 링크
target_link_libraries(step2_conv1_bn1_relu
  PRIVATE
    CUDA::cudart                           # 기본 CUDA runtime
)

set_target_properties(step2_conv1_bn1_relu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
