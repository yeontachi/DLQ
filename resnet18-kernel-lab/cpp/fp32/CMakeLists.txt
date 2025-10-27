set(SRCS
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  runtime/infer_conv1_bn1_relu.cu   # ← 여기 .cu
)
add_executable(step2_conv1_bn1_relu ${SRCS})
add_executable(infer_stem_block0
    runtime/infer_stem_block0.cpp
    kernels/im2col.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    kernels/maxpool2d.cu      # NEW
    kernels/add_tensor.cu     # NEW
    kernels/basic_block.cu    # NEW
)

# 그리고 CUDA 설정, include 경로, cublas 링크 등은
# infer_conv1_bn1_relu 타깃과 동일하게 걸어주면 돼.

target_include_directories(step2_conv1_bn1_relu
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)
target_link_libraries(step2_conv1_bn1_relu PRIVATE CUDA::cudart)
set_target_properties(step2_conv1_bn1_relu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
