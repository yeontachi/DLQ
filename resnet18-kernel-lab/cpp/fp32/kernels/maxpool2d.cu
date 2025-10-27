// cpp/fp32/kernels/maxpool2d.cu

#include <cuda_runtime.h>
#include <float.h>
#include "utils.hpp"

// maxpool2d_forward
//  - NCHW layout
//  - kernel = 3, stride = 2, pad = 1 (일반화 가능하게 인자 받지만 기본 stem pool 용도)
//  - output shape (H_out, W_out)는 호출자가 계산해서 넣는다
//
// 호출 형태는 Step2 스타일과 맞춰서
//   <<<grid, block>>>
// 직접 런치해도 되고,
//   maxpool2d_forward(...)
//라는 host 함수를 한 번 감싸도 된다.
// Step2 main은 직접 launch하는 스타일이라 여기서도 host wrapper + kernel 같이 둔다.

extern "C" __global__
void maxpool2d_kernel_fp32(
    const float* __restrict__ in,   // [N,C,H,W]
    float* __restrict__ out,        // [N,C,H_out,W_out]
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    // idx → (n,c,oh,ow)
    int ow = idx % W_out;
    int tmp = idx / W_out;
    int oh = tmp % H_out;
    tmp   /= H_out;
    int c  = tmp % C;
    int n  = tmp / C;

    int ih0 = oh * stride_h - pad_h;
    int iw0 = ow * stride_w - pad_w;

    float maxv = -FLT_MAX;

    for (int kh = 0; kh < kernel_h; ++kh) {
        int ih = ih0 + kh;
        if (ih < 0 || ih >= H) continue;
        for (int kw = 0; kw < kernel_w; ++kw) {
            int iw = iw0 + kw;
            if (iw < 0 || iw >= W) continue;

            int in_off = ((n*C + c)*H + ih)*W + iw;
            float v = in[in_off];
            if (v > maxv) maxv = v;
        }
    }

    int out_off = ((n*C + c)*H_out + oh)*W_out + ow;
    out[out_off] = maxv;
}

// 이건 Step2 스타일에서 T.start()~launch~sync 하는 흐름이었지?
// 편하게 쓰라고 host wrapper도 같이 둔다.
// (원하면 직접 kernel launch해도 됨)
inline void launch_maxpool2d_fp32(
    const float* dIn,
    float* dOut,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out)
{
    int total = N*C*H_out*W_out;
    dim3 blk(256);
    dim3 grd((total + blk.x - 1)/blk.x);

    maxpool2d_kernel_fp32<<<grd, blk>>>(
        dIn, dOut,
        N,C,H,W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        H_out, W_out
    );
}
