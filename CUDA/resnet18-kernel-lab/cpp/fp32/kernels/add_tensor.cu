/*
add_tensor.cu

Purpose:
    Elementwise residual add for ResNet skip connection.
    out[i] = a[i] + b[i]

Signature contract:
    extern "C" __global__
    void tensor_add_fp32_kernel(
        const float* a,
        const float* b,
        float* out,
        int total);

Notes:
    - a, b, out 은 모두 같은 shape [N,C,H,W] (flattened as 1-D)
    - total = N*C*H*W
    - grid/block: 1D
*/

#include <cuda_runtime.h>

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
