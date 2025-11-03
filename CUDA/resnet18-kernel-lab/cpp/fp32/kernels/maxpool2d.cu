#include <cuda_runtime.h>
#include "maxpool2d.cuh"

__device__ inline float neg_inf() { return -1e30f; }

extern "C" __global__
void maxpool2d_3x3_s2p1_nchw(const float* __restrict__ x,
                             int N, int C, int H, int W,
                             float* __restrict__ y)
{
    // PyTorch stem maxpool: k=3, s=2, p=1 (floor)
    const int OH = (H + 2*1 - 3)/2 + 1;
    const int OW = (W + 2*1 - 3)/2 + 1;

    int n = blockIdx.z;
    int c = blockIdx.y;
    int oh = blockIdx.x / OW;
    int ow = blockIdx.x % OW;

    int ih_center = oh*2 - 1 + 1; // s*oh - p + offset
    int iw_center = ow*2 - 1 + 1;

    float m = neg_inf();
    for (int kh=0; kh<3; ++kh){
        for (int kw=0; kw<3; ++kw){
            int ih = ih_center + kh;
            int iw = iw_center + kw;
            float v = neg_inf();
            if (0<=ih && ih<H && 0<=iw && iw<W){
                int idx = ((n*C + c)*H + ih)*W + iw;
                v = x[idx];
            }
            m = fmaxf(m, v);
        }
    }
    int oidx = ((n*C + c)*OH + oh)*OW + ow;
    y[oidx] = m;
}
