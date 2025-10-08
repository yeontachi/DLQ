/*
im2col.cu
What : 입력 이미지/feature map(NCHW)에서 컨볼루션의 슬라이딩 윈도우(receptive field)를 열(column) 단위로 평탄화해 모아 만든 행렬
       결과 행렬 col의 모양은 (C*kH*kW)x(N*OH*OW)
Why  : 컨볼루션을 행렬곱(GEMM)으로 바꾸어 고성능 BLAS/cuBLAS를 활용하기 위해
       커널 가중치도 (OC) x (c*kH*kW)로 평탄화(W_col)하면 Y = W_col*col -> 출력 (OC) x (N*OH*OW)를 얻고 reshape
HOW  : 출력 위치마다(oh, ow) 윈도우를 꺼내 길이 C*kH*kW 벡터로 펼침 -> col의 한 열에 저장
       패딩/스트라이드 반영 : OH = (H + 2pH -kH)/sH + 1, OW = (W + 2pW - kW)/sW + 1
*/
#include <cuda_runtime.h>

// x: 입력(N*C*H*W, 여기선 N=1만 처리)
// col: 출력((C*kH*kW) x (N*OH*OW)) 행렬을 1D(row-major)로 저장
extern "C" __global__
void im2col_nchw(const float* __restrict__ x, // input: N*C*H*W
                 int N, int C, int H, int W,  // input size
                 int kH, int kW, int sH, int sW, int pH, int pW, // 커널/스트라이드/패딩
                 float* __restrict__ col)     // output: (C*kH*kW, N*OH*OW) row-major
{
    // 배치 N=1만 지원(단순화), 필요 시 n 차원 병렬 추가
    int n = 0;

    // 출력 feature map 크기 계산
    int OH = (H + 2*pH - kH)/sH + 1;
    int OW = (W + 2*pW - kW)/sW + 1;

    // 블록/스레드에서 출력 위치(oh, ow) 담당
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;

    // 경계 밖은 바로 반환
    if (oh >= OH || ow >= OW) return;

    // 현재 (oh, ow)에 대한 열(column) 인덱스
    // 열 수 = N*OH*OW, N=1이므로 OH*OW
    int out_index = oh * OW + ow;
    int colStride = OH * OW; // row-major에서 한 행(row)의 stride(=열 수)

    // 모든 채널과 커널 위치(kh, kw)를 순회하면서
    // 해당 수용영역의 픽셀을 col의 (row=r, col=out_index)에 기록
    for (int c = 0; c < C; ++c){
        for (int kh = 0; kh < kH; ++kh){
            for(int kw = 0; kw < kW; ++kw){

                // 입력 좌표(ih, iw): 스트라이드/패딩 적용
                int ih = oh * sH - pH + kh;
                int iw = ow * sW - pW + kw;

                // 패딩 영역(경계 밖)은 0으로 채움
                float v = 0.f;
                if (ih >= 0 && iw >= 0 && ih < H && iw < W){
                    // N=1 가정이므로 오프셋은 c*H*W + ih*W + iw
                    int idx = c * H * W + ih * W + iw;
                    v = x[idx];
                }

                // (row = r) = 채널과 커널 위치를 평탄화한 인덱스
                // r ∈ [0, C*kH*kW)
                int r = c * kH * kW + kh * kW + kw;

                // row-major 2D -> 1D: row * colstride + col
                // colstride = 전체 열 수 = N*OH*OW (N=1 -> OH*OW)
                col[r * colStride + out_index] = v;
            }
        }
    }
}