# 소스 목록
set(SRCS
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  runtime/infer_conv1_bn1_relu.cpp
)

# 실행 파일 타겟 정의
add_executable(step2_conv1_bn1_relu ${SRCS})

# 반드시 타겟 생성(add_executable) 이후에 include 디렉토리 추가
target_include_directories(step2_conv1_bn1_relu
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)

# CUDA 분리 컴파일 등 속성
set_target_properties(step2_conv1_bn1_relu PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
