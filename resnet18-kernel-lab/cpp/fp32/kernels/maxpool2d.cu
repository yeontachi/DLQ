/*
maxpool2d.cu

Purpose:
    NCHW 텐서에 대해 2D max pooling (예: kernel=3, stride=2, pad=1)
    출력 shape (N, C, H_out, W_out)

Signature contract:
    extern "C" __global__
    void maxpool2d_kernel_fp32(
        const float* in,
        float* out,
        int N, int C, int H, int W,
        int kernel_h, int kernel_w,
        int stride_h, int stride_w,
        int pad_h, int pad_w,
        int H_out, int W_out);

호출 예 (infer_stem_block0.cu 안에서):
    int total = N*C*H_out*W_out;
    dim3 blk(256);
    dim3 grd((total + blk.x - 1)/blk.x);
    maxpool2d_kernel_fp32<<<grd, blk>>>(
        dConv1Out, dPoolOut,
        N, C1_out, H1, W1,
        3, 3,
        2, 2,
        1, 1,
        H2, W2
    );

Notes:
    - 현재는 batch N=1이긴 하지만, 커널은 일반 N 지원.
    - 읽기/쓰기 모두 global memory. shared mem 없음 (간단/직관).
*/

#include <cuda_runtime.h>
#include <float.h>

extern "C" __global__
void maxpool2d_kernel_fp32(
    const float* __restrict__ in,  // [N,C,H,W]
    float* __restrict__ out,       // [N,C,H_out,W_out]
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out)
{
    // flat output index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // idx -> (n, c, oh, ow)
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp    /= H_out;
    int c  = tmp % C;
    int n  = tmp / C;

    // 원본 영역 시작 (탑레프트 기준)
    int ih0 = oh * stride_h - pad_h;
    int iw0 = ow * stride_w - pad_w;

    float max_val = -FLT_MAX;

    // 커널 윈도우 스캔
    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= W) continue;

            // in offset: ((n*C + c)*H + ih)*W + iw
            int in_off = ((n * C + c) * H + ih) * W + iw;
            float v = in[in_off];
            if (v > max_val) {
                max_val = v;
            }
        }
    }

    // out offset: ((n*C + c)*H_out + oh)*W_out + ow
    int out_off = ((n * C + c) * H_out + oh) * W_out + ow;
    out[out_off] = max_val;
}
