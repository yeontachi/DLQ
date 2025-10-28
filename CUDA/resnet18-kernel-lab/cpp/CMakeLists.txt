cmake_minimum_required(VERSION 3.20)
project(resnet18_step2 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# CUDA Toolkit 찾기
find_package(CUDAToolkit REQUIRED)

add_subdirectory(fp32)
