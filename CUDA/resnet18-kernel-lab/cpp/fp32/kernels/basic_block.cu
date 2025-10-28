/*
basic_block.cu

ResNet18 BasicBlock (identity skip, no downsample)
Structure:
    y1 = ReLU( BN1( Conv1(x) ) )
    y2 = BN2( Conv2(y1) )
    out = ReLU( y2 + x )

We assume:
    - stride=1, padding=1, kernel=3x3 for both convs
    - in/out channels are the same (e.g. 64 -> 64)
    - spatial size unchanged across the block
    - N typically = 1 here, but N is passed for generality

This file provides:
    extern "C"
    void basic_block_fp32_forward_identity(...);

It launches kernels:
    im2col_nchw
    sgemm_tiled
    bn_inference
    relu_forward
    tensor_add_fp32_kernel
*/

#include <cuda_runtime.h>
#include <cmath>        // for nothing fancy, but keep parity with other .cu files
#include "../runtime/utils.hpp"  // for CUDA_CHECK if you want to add checks (optional)

// ---- Forward declarations for device kernels (must match definitions exactly)

extern "C" __global__
void im2col_nchw(const float* x,
                 int N,int C,int H,int W,
                 int kH,int kW,
                 int sH,int sW,
                 int pH,int pW,
                 float* col);

extern "C" __global__
void sgemm_tiled(const float* A,
                 const float* B,
                 float* C,
                 int M, int N_, int K);

extern "C" __global__
void bn_inference(float* x,
                  const float* gamma,
                  const float* beta,
                  const float* mean,
                  const float* var,
                  float eps,
                  int C, int H, int W);

extern "C" __global__
void relu_forward(float* x,
                  int total);

extern "C" __global__
void tensor_add_fp32_kernel(const float* a,
                            const float* b,
                            float* out,
                            int total);

// ---------------------------------------------------------------------------
// internal helper: conv + bn + relu
// out shape: [N, Cout, H, W]
// weights: d_wcol is [Cout, Cin * kH * kW]
// d_colBuf: workspace of size (Cin * kH * kW) * (H*W)
static void conv_bn_relu_once(
    const float* d_in,           // [N,Cin,H,W]
    float*       d_out,          // [N,Cout,H,W] (Cout==Cin for this block)
    float*       d_colBuf,       // workspace
    const float* d_wcol,         // [Cout, Cin*k*k]
    const float* d_gamma,        // bn gamma (Cout)
    const float* d_beta,         // bn beta (Cout)
    const float* d_mean,         // bn running_mean (Cout)
    const float* d_var,          // bn running_var (Cout)
    float        bn_eps,
    int N, int C, int H, int W,
    cudaStream_t stream
){
    // kernel config
    const int kH = 3, kW = 3;
    const int sH = 1, sW = 1;
    const int pH = 1, pW = 1;

    // output spatial (stride=1,pad=1,k=3 => same H,W)
    int OH = H;
    int OW = W;

    int KCOL = C * kH * kW;   // rows per output pixel / shared K dimension
    int NCOL = OH * OW;       // number of output pixels per batch (N assumed 1)

    // 1. im2col
    {
        dim3 blk(16,16);
        dim3 grd((OW + blk.x - 1)/blk.x,
                 (OH + blk.y - 1)/blk.y);

        im2col_nchw<<<grd, blk, 0, stream>>>(
            d_in,
            N, C, H, W,
            kH, kW,
            sH, sW,
            pH, pW,
            d_colBuf
        );
        // (Optional safety) CUDA_CHECK(cudaGetLastError());
    }

    // 2. GEMM
    // d_out is [Cout, OH*OW] laid out row-major, we'll treat N=1 so that also stands for [N,Cout,OH,OW]
    {
        dim3 blk(32,32);
        dim3 grd((NCOL + blk.x - 1)/blk.x,
                 (C     + blk.y - 1)/blk.y);
        // NOTE: here Cout == C for identity block. If not, pass Cout instead of C.

        sgemm_tiled<<<grd, blk, 0, stream>>>(
            d_wcol,     // [Cout, KCOL]
            d_colBuf,   // [KCOL, NCOL]
            d_out,      // [Cout, NCOL]
            C,          // M (= Cout)
            NCOL,       // N_
            KCOL        // K
        );
        // CUDA_CHECK(cudaGetLastError());
    }

    // 3. BN inplace on d_out
    {
        int total_spatial = C * OH * OW;
        dim3 blk(256);
        dim3 grd((total_spatial + blk.x - 1)/blk.x);

        bn_inference<<<grd, blk, 0, stream>>>(
            d_out,
            d_gamma,
            d_beta,
            d_mean,
            d_var,
            bn_eps,
            C, OH, OW
        );
        // CUDA_CHECK(cudaGetLastError());
    }

    // 4. ReLU inplace on d_out
    {
        int total_relu = C * OH * OW;
        dim3 blk(256);
        dim3 grd((total_relu + blk.x - 1)/blk.x);

        relu_forward<<<grd, blk, 0, stream>>>(
            d_out,
            total_relu
        );
        // CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// internal helper: conv + bn (NO relu at end)
// writes into d_out
static void conv_bn_only_once(
    const float* d_in,           // [N,C,H,W]
    float*       d_out,          // [N,C,H,W]
    float*       d_colBuf,       // workspace
    const float* d_wcol,         // weights flattened
    const float* d_gamma,
    const float* d_beta,
    const float* d_mean,
    const float* d_var,
    float        bn_eps,
    int N, int C, int H, int W,
    cudaStream_t stream
){
    const int kH = 3, kW = 3;
    const int sH = 1, sW = 1;
    const int pH = 1, pW = 1;

    int OH = H;
    int OW = W;

    int KCOL = C * kH * kW;
    int NCOL = OH * OW;

    // im2col
    {
        dim3 blk(16,16);
        dim3 grd((OW + blk.x - 1)/blk.x,
                 (OH + blk.y - 1)/blk.y);

        im2col_nchw<<<grd, blk, 0, stream>>>(
            d_in,
            N, C, H, W,
            kH, kW,
            sH, sW,
            pH, pW,
            d_colBuf
        );
        // CUDA_CHECK(cudaGetLastError());
    }

    // GEMM
    {
        dim3 blk(32,32);
        dim3 grd((NCOL + blk.x - 1)/blk.x,
                 (C     + blk.y - 1)/blk.y);

        sgemm_tiled<<<grd, blk, 0, stream>>>(
            d_wcol,
            d_colBuf,
            d_out,
            C,        // M
            NCOL,     // N_
            KCOL      // K
        );
        // CUDA_CHECK(cudaGetLastError());
    }

    // BN (no relu)
    {
        int total_spatial = C * OH * OW;
        dim3 blk(256);
        dim3 grd((total_spatial + blk.x - 1)/blk.x);

        bn_inference<<<grd, blk, 0, stream>>>(
            d_out,
            d_gamma,
            d_beta,
            d_mean,
            d_var,
            bn_eps,
            C, OH, OW
        );
        // CUDA_CHECK(cudaGetLastError());
    }
}

// ---------------------------------------------------------------------------
// public API: full residual block forward (identity skip)
//  - d_x: input [N,C,H,W] (also the skip branch)
//  - d_out: final output [N,C,H,W]
//  - d_tmp1: scratch [N,C,H,W]
//  - d_tmp2: scratch [N,C,H,W]
//  - d_colBuf: workspace [(C*3*3)*(H*W)]
//
//  weights/bn params are already on device and flattened
//
//  N,C,H,W are shape of d_x (and also d_out)
//
// NOTE: For ResNet18 layer1[0], C == 64, N == 1, H == W == 56.
//
extern "C"
void basic_block_fp32_forward_identity(
    const float* d_x,        // input / identity skip
    float* d_out,            // final output
    float* d_tmp1,           // after conv1/bn/relu
    float* d_tmp2,           // after conv2/bn
    float* d_colBuf,         // workspace

    const float* d_w1col,    // conv1 weights flattened
    const float* d_bn1_gamma,
    const float* d_bn1_beta,
    const float* d_bn1_mean,
    const float* d_bn1_var,
    float bn1_eps,

    const float* d_w2col,    // conv2 weights flattened
    const float* d_bn2_gamma,
    const float* d_bn2_beta,
    const float* d_bn2_mean,
    const float* d_bn2_var,
    float bn2_eps,

    int N, int C, int H, int W,

    cudaStream_t stream
){
    // 1. y1 = conv1+bn1+relu(d_x)
    conv_bn_relu_once(
        d_x,
        d_tmp1,
        d_colBuf,
        d_w1col,
        d_bn1_gamma,
        d_bn1_beta,
        d_bn1_mean,
        d_bn1_var,
        bn1_eps,
        N, C, H, W,
        stream
    );

    // 2. y2 = conv2+bn2(d_tmp1)
    conv_bn_only_once(
        d_tmp1,
        d_tmp2,
        d_colBuf,
        d_w2col,
        d_bn2_gamma,
        d_bn2_beta,
        d_bn2_mean,
        d_bn2_var,
        bn2_eps,
        N, C, H, W,
        stream
    );

    // 3. out = y2 + x  (residual add)
    {
        int total = N * C * H * W;
        dim3 blk(256);
        dim3 grd((total + blk.x - 1)/blk.x);

        tensor_add_fp32_kernel<<<grd, blk, 0, stream>>>(
            d_tmp2,
            d_x,
            d_out,
            total
        );
        // CUDA_CHECK(cudaGetLastError());
    }

    // 4. ReLU(out)
    {
        int total = N * C * H * W;
        dim3 blk(256);
        dim3 grd((total + blk.x - 1)/blk.x);

        relu_forward<<<grd, blk, 0, stream>>>(
            d_out,
            total
        );
        // CUDA_CHECK(cudaGetLastError());
    }
}
