/*
relu.cu
ReLU 활성화 함수 y = max(0, x) 를 in-place로 적용

매우 단순한 요소별(element-wise) 연산으로, BN 뒤에 바로 연결하기 좋음

주의
 - 분기(조건문)로 인한 warp divergence가 있지만, ReLU는 일반적으로 경량이라 큰 문제는 아니다.
 - BN과 하나의 커널로 융합하면 글로벌 메모리 접근을 줄여 속도에 이점
*/
#include <cuda_runtime.h>

// x: 입력/출력(같은 버퍼, in-place), n: 요소 개수
extern "C" __global__
void relu_forward(float* x, int n)
{
    // 전역 인덱스
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // 경계 체크 + 음수이면 0으로 clamp
    if (i < n && x[i] < 0.f) x[i] = 0.f;
}