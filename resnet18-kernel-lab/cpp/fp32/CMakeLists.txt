set(SRCS
  kernels/im2col.cu
  kernels/sgemm_tiled.cu
  kernels/bn_inference.cu
  kernels/relu.cu
  runtime/infer_conv1_bn1_relu.cu   # ← 여기 .cu
)
add_executable(step2_conv1_bn1_relu ${SRCS})

target_include_directories(step2_conv1_bn1_relu
  PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
)
target_link_libraries(step2_conv1_bn1_relu PRIVATE CUDA::cudart)
set_target_properties(step2_conv1_bn1_relu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
