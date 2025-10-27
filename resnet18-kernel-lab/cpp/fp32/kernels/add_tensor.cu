// cpp/fp32/kernels/add_tensor.cu

#include <cuda_runtime.h>
#include "utils.hpp"

// out[i] = a[i] + b[i], i in [0, total)
// NCHW 전체를 단순 flat하게 본다.

extern "C" __global__
void tensor_add_fp32_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ out,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = a[idx] + b[idx];
}

inline void launch_tensor_add_fp32(
    const float* a,
    const float* b,
    float* out,
    int N, int C, int H, int W)
{
    int total = N*C*H*W;
    dim3 blk(256);
    dim3 grd((total + blk.x - 1)/blk.x);

    tensor_add_fp32_kernel<<<grd, blk>>>(a,b,out,total);
}
