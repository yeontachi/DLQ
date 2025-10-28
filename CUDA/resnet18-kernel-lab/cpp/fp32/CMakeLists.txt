cmake_minimum_required(VERSION 3.18)
project(resnet18_fp32 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Include dirs
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# 공통 CUDA 커널들
set(FP32_KERNELS
    kernels/im2col.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    kernels/maxpool2d.cu
    kernels/add_tensor.cu
    kernels/basic_block.cu
)

# Step2 실행 타깃
add_executable(infer_conv1_bn1_relu
    runtime/infer_conv1_bn1_relu.cu
    ${FP32_KERNELS}
)

# Step3 실행 타깃
add_executable(infer_stem_block0
    runtime/infer_stem_block0.cu
    ${FP32_KERNELS}
)

# 공통 속성
foreach(tgt infer_conv1_bn1_relu infer_stem_block0)
    set_target_properties(${tgt} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_options(${tgt} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo -arch=sm_80>
    )
endforeach()
