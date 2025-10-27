# cpp/fp32/CMakeLists.txt

cmake_minimum_required(VERSION 3.18)

# 상위 CMakeLists에서 project(...)와 enable_language(CUDA) 를 이미 하고 있으면
# 여기서 다시 project() 안 해도 되지만, Step2에서 이미 어떻게 되어 있는지에 따라 다를 수 있음.
# 만약 상위 cpp/CMakeLists.txt에서 project를 잡아줬다면 여긴 생략해도 돼.
# 여기선 self-contained 예시로 둘게.

project(fp32_kernels LANGUAGES CXX CUDA)

# CUDA 표준 설정 (필요하면)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 공통 include 경로
# utils.hpp 는 runtime/ 쪽에 있으니까 여기도 포함
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/runtime
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)

# 공통 커널 소스들
set(FP32_KERNEL_SRCS
    kernels/im2col.cu
    kernels/sgemm_tiled.cu
    kernels/bn_inference.cu
    kernels/relu.cu
    kernels/maxpool2d.cu
    kernels/add_tensor.cu
    kernels/basic_block.cu
)

#
# 첫 번째 실행 바이너리: infer_conv1_bn1_relu  (Step2)
#
add_executable(infer_conv1_bn1_relu
    runtime/infer_conv1_bn1_relu.cu
    ${FP32_KERNEL_SRCS}
)

# 만약 cublas / cudart 등 링크가 Step2에서 필요했다면 여기에 그대로 유지
# 예:
# target_link_libraries(infer_conv1_bn1_relu
#     cudart
#     cublas
# )

set_target_properties(infer_conv1_bn1_relu PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#
# 두 번째 실행 바이너리: infer_stem_block0  (Step3)
#
add_executable(infer_stem_block0
    runtime/infer_stem_block0.cu
    ${FP32_KERNEL_SRCS}
)

# 마찬가지로 라이브러리 링크
# target_link_libraries(infer_stem_block0
#     cudart
#     cublas
# )

set_target_properties(infer_stem_block0 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

#
# (선택) 최적화 플래그, 아키텍처 지정
#
# 예: Ampere 이상만 타겟이라면
# target_compile_options(infer_stem_block0 PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo -arch=sm_80>
# )
# target_compile_options(infer_conv1_bn1_relu PRIVATE
#     $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo -arch=sm_80>
# )
#
# 위 arch= 부분은 네 GPU에 맞춰 바꾸면 돼.
