cmake_minimum_required(VERSION 3.23)
project(resnet18_kernel_lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 아키텍처는 필요 시 조정
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75;80;86;89;90)
endif()

find_package(CUDAToolkit REQUIRED)

# 모든 실행파일을 build/ 아래로
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

add_subdirectory(fp32)
