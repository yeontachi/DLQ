set(SRCS
    kernels/im2cols.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    runtime/infer_conv1_bn1_relu.cpp
)
add_executable(step2_conv1_bn1_relu ${SRCS})
target_include_directories((step2_conv1_bn1_relu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/runtime))
set_target_properties(step2_conv1_bn1_relu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
