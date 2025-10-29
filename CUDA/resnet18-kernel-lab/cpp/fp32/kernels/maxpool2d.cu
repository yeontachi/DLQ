#include <cuda_runtime.h>
extern "C" __global__
void maxpool2d_nchw(const float* __restrict__ x, // [C,H,W], N=1
                    int C,int H,int W,
                    int kH,int kW,int sH,int sW,int pH,int pW,
                    float* __restrict__ y)       // [C,OH,OW]
{
    int OH = (H + 2*pH - kH)/sH + 1;
    int OW = (W + 2*pW - kW)/sW + 1;

    int c  = blockIdx.z;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    if (c>=C || oh>=OH || ow>=OW) return;

    float m = -1e30f;
    for (int kh=0; kh<kH; ++kh){
        int ih = oh*sH - pH + kh;
        if (ih<0 || ih>=H) continue;
        for (int kw=0; kw<kW; ++kw){
            int iw = ow*sW - pW + kw;
            if (iw<0 || iw>=W) continue;
            float v = x[c*H*W + ih*W + iw];
            m = (v>m)? v : m;
        }
    }
    y[c*OH*OW + oh*OW + ow] = m;
}
