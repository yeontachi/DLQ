# =========================
# cpp/fp32/CMakeLists.txt
# =========================

# (상위 CMakeLists에서 이미 설정했을 수 있음. 안전하게 중복 설정해도 무해)
enable_language(CUDA)

# 실행/라이브러리 산출물 경로 통일
set(FP32_OUTDIR ${CMAKE_BINARY_DIR}/fp32)

# 디렉터리 변수
set(FP32_ROOT        ${CMAKE_CURRENT_SOURCE_DIR})
set(FP32_KERNELS_DIR ${FP32_ROOT}/kernels)
set(FP32_RUNTIME_DIR ${FP32_ROOT}/runtime)

# 인클루드 경로 공통 세트
set(FP32_INCLUDES
  ${FP32_RUNTIME_DIR}
  ${FP32_KERNELS_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# 공통 커널(여기 기재된 파일은 실제로 존재해야 합니다)
set(KERNELS_COMMON
  ${FP32_KERNELS_DIR}/im2col.cu
  ${FP32_KERNELS_DIR}/sgemm_tiled.cu
  ${FP32_KERNELS_DIR}/bn_inference.cu
  ${FP32_KERNELS_DIR}/relu.cu
)

# 공통 프로퍼티
function(set_common_props target_name)
  target_include_directories(${target_name} PRIVATE ${FP32_INCLUDES})
  target_link_libraries(${target_name} PRIVATE CUDA::cudart)
  set_target_properties(${target_name} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
    LIBRARY_OUTPUT_DIRECTORY ${FP32_OUTDIR}
    ARCHIVE_OUTPUT_DIRECTORY ${FP32_OUTDIR}
  )
  # 필요시 아키텍처 자동 선택
  if (POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
  endif()
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES native)
endfunction()

# ---------- Step2: conv1 -> bn1 -> relu ----------
add_executable(step2_conv1_bn1_relu
  ${KERNELS_COMMON}
  ${FP32_RUNTIME_DIR}/infer_conv1_bn1_relu.cu
)
set_common_props(step2_conv1_bn1_relu)

# ---------- Step3: layer1 (2 blocks) ----------
add_executable(step3_layer1
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer1.cu
)
set_common_props(step3_layer1)

# ---------- Step4: layer2 (downsample 포함) ----------
add_executable(step4_layer2
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer2.cu
)
set_common_props(step4_layer2)

# ---------- Step5: layer3 (downsample 포함) ----------
add_executable(step5_layer3
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer3.cu
)
set_common_props(step5_layer3)

# ---------- Step6: layer4 (downsample 포함) ----------
add_executable(step6_layer4
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_RUNTIME_DIR}/infer_layer4.cu
)
set_common_props(step6_layer4)

# ---------- Step7: head(GAP/FC/softmax 등) ----------
#   gap_global.cu, softmax.cu 가 있다면 포함
add_executable(step7_head
  ${FP32_KERNELS_DIR}/sgemm_tiled.cu
  ${FP32_KERNELS_DIR}/gap_global.cu
  ${FP32_KERNELS_DIR}/softmax.cu
  ${FP32_RUNTIME_DIR}/infer_head.cu
)
set_common_props(step7_head)

# ---------- Step8: E2E ----------
add_executable(step8_e2e
  ${KERNELS_COMMON}
  ${FP32_KERNELS_DIR}/add.cu
  ${FP32_KERNELS_DIR}/gap_global.cu
  ${FP32_RUNTIME_DIR}/infer_e2e.cu
)
set_common_props(step8_e2e)

# ---------- FC(배치/헤드 등에서 재사용) 라이브러리 ----------
#  - 중복 정의 방지를 위해 가드
#  - sgemm_tiled.cu를 재사용(헤더가 아니라 .cu를 링크 소스로 포함)
if (NOT TARGET fc)
  set(FC_LIB_SRCS
    ${FP32_KERNELS_DIR}/transpose.cu     # 존재하는 경우에만 유지, 없으면 제거
    ${FP32_KERNELS_DIR}/add_bias.cu      # 존재하는 경우에만 유지, 없으면 제거
    ${FP32_RUNTIME_DIR}/fc_api.cu        # 존재하는 경우에만 유지, 없으면 제거
    ${FP32_KERNELS_DIR}/sgemm_tiled.cu   # GEMM 재사용
  )
  # 실제 존재하지 않는 파일이 있다면 위 목록에서 지워 주세요.
  list(FILTER FC_LIB_SRCS INCLUDE REGEX ".+\\.(cu|cuh)$")

  add_library(fc SHARED ${FC_LIB_SRCS})
  # 일부 환경에서 cuda 드라이버 심볼 필요시 CUDA::cuda_driver 추가
  target_link_libraries(fc PRIVATE CUDA::cudart)
  target_include_directories(fc PRIVATE ${FP32_INCLUDES})
  set_target_properties(fc PROPERTIES
    OUTPUT_NAME "fc"
    LIBRARY_OUTPUT_DIRECTORY ${FP32_OUTDIR}
    RUNTIME_OUTPUT_DIRECTORY ${FP32_OUTDIR}
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

# (필요하면 특정 실행 파일이 fc를 사용하도록 링크)
# target_link_libraries(step7_head PRIVATE fc)
# target_link_libraries(step8_e2e  PRIVATE fc)
