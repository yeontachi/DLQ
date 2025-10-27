// cpp/fp32/kernels/basic_block.cu

#include <cuda_runtime.h>
#include <cmath>
#include "utils.hpp"

// 외부 커널 선언 (Step2 스타일 그대로)
extern "C" __global__
void im2col_nchw(const float*,
                 int N,int C,int H,int W,
                 int KH,int KW,
                 int SH,int SW,
                 int PH,int PW,
                 float* outCol);

extern "C" __global__
void sgemm_tiled(const float*, const float*, float*,
                 int M, int N_, int K); 
// NOTE: here M=OC, N_=OH*OW, K=C*KH*KW

extern "C" __global__
void bn_inference(float*,
                  const float*,
                  const float*,
                  const float*,
                  const float*,
                  float eps,
                  int C, int H, int W);

extern "C" __global__
void relu_forward(float*, int total);

// 우리가 Step3에서 새로 만든 것들
extern "C" __global__
void tensor_add_fp32_kernel(const float*,const float*,float*,int);
inline void launch_tensor_add_fp32(const float*,const float*,float*,int,int,int,int);

// conv/bn/relu 한 번 (3x3 stride=1 pad=1, 채널 유지 가정)
// in   : [N,C,H,W]
// w    : [Cout, C, kH, kW] (export에서 conv weight)
// out  : [N,Cout,H,W] (stride=1 pad=1 가정이므로 H,W 동일)
// tmpCol, tmpY : 임시버퍼 (col matrix, matmul out 등)
// bn params: gamma,beta,mean,var,eps
//
// 이 helper는 block 내부에서만 쓸 거라 extern "C" 필요 X
static void conv_bn_relu_once(
    const float* d_in,
    const float* d_w,
    float* d_out,
    float* d_colBuf,
    // shapes
    int N, int C, int H, int W,
    int Cout,
    int kH, int kW,
    int sH, int sW,
    int pH, int pW,
    // bn params
    const float* d_gamma,
    const float* d_beta,
    const float* d_mean,
    const float* d_var,
    float eps,
    cudaStream_t stream = 0
){
    // 1. im2col
    // output spatial size (OH, OW)
    int OH = (H + 2*pH - kH)/sH + 1;
    int OW = (W + 2*pW - kW)/sW + 1;

    int KCOL  = C*kH*kW;   // rows per output pixel
    int NCOL  = OH*OW;     // number of output pixels
    // d_colBuf: size KCOL * NCOL floats

    dim3 blkIm2(16,16);
    dim3 grdIm2((OW+blkIm2.x-1)/blkIm2.x,
                (OH+blkIm2.y-1)/blkIm2.y);

    im2col_nchw<<<grdIm2, blkIm2, 0, stream>>>(
        d_in,
        N,C,H,W,
        kH,kW,
        sH,sW,
        pH,pW,
        d_colBuf
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. GEMM
    // Y (Cout x NCOL) = W_col (Cout x KCOL) * Col (KCOL x NCOL)
    // 여기서는 weight를 미리 [Cout, C*kH*kW]로 펴서 줘야 한다.
    // => 호출하는 쪽에서 d_w는 이미 그런 형태로 준비돼 있다고 가정.
    // (너 Step2 코드에서 Wcol 따로 만들어서 dW에 넣었지? 그 방식을 그대로 쓸 거야)
    dim3 blkGemm(32,32);
    dim3 grdGemm((NCOL+blkGemm.x-1)/blkGemm.x,
                 (Cout+blkGemm.y-1)/blkGemm.y);

    sgemm_tiled<<<grdGemm, blkGemm, 0, stream>>>(
        d_w,        // [Cout, KCOL]
        d_colBuf,   // [KCOL, NCOL]
        d_out,      // [Cout, NCOL] => which we interpret as [N=1,Cout,OH,OW]
        Cout, NCOL, KCOL
    );
    CUDA_CHECK(cudaGetLastError());

    // 3. BN inference (in-place)
    //   bn_inference<<<grid,block>>>(d_out, gamma,beta,mean,var,eps, C, H, W)
    {
        int total_spatial = Cout*OH*OW;
        dim3 blkBN(256);
        dim3 grdBN((total_spatial+blkBN.x-1)/blkBN.x);

        bn_inference<<<grdBN, blkBN, 0, stream>>>(
            d_out,
            d_gamma,
            d_beta,
            d_mean,
            d_var,
            eps,
            Cout, OH, OW
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. ReLU (in-place)
    {
        int total_relu = Cout*OH*OW;
        dim3 blkReLU(256);
        dim3 grdReLU((total_relu+blkReLU.x-1)/blkReLU.x);

        relu_forward<<<grdReLU, blkReLU, 0, stream>>>(
            d_out,
            total_relu
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

// basic block 전체
// d_x      : [N,C,H,W]  (also skip input)
// d_out    : [N,C,H,W]
// d_tmp1   : [N,C,H,W] scratch (will hold y1 after first conv/bn/relu)
// d_tmp2   : [N,C,H,W] scratch (will hold y2 after second conv/bn (no relu yet))
// d_colBuf : workspace for im2col (size = C*kH*kW * H*W)  <-- we'll reuse same buf
//
// conv1 weights are assumed pre-flattened to [Cout, C*kH*kW] like Step2's Wcol
// conv2 weights same idea.
//
extern "C"
void basic_block_fp32_forward_identity(
    const float* d_x,
    float* d_out,
    float* d_tmp1,
    float* d_tmp2,
    float* d_colBuf, // workspace for im2col/GEMM

    // conv1 params
    const float* d_w1col,
    int N, int C, int H, int W,
    int C1out,
    int k1H,int k1W,
    int s1H,int s1W,
    int p1H,int p1W,
    const float* d_bn1_gamma,
    const float* d_bn1_beta,
    const float* d_bn1_mean,
    const float* d_bn1_var,
    float bn1_eps,

    // conv2 params
    const float* d_w2col,
    int C2out,
    int k2H,int k2W,
    int s2H,int s2W,
    int p2H,int p2W,
    const float* d_bn2_gamma,
    const float* d_bn2_beta,
    const float* d_bn2_mean,
    const float* d_bn2_var,
    float bn2_eps,

    cudaStream_t stream
){
    // 1. tmp1 = conv1+bn+relu(x)
    conv_bn_relu_once(
        d_x,
        d_w1col,
        d_tmp1,
        d_colBuf,
        N,C,H,W,
        C1out,
        k1H,k1W,
        s1H,s1W,
        p1H,p1W,
        d_bn1_gamma,
        d_bn1_beta,
        d_bn1_mean,
        d_bn1_var,
        bn1_eps,
        stream
    );

    // 2. tmp2 = conv2+bn(tmp1)  (여기서는 마지막 relu 안 함)
    {
        // conv2 + bn (no relu)
        // conv_bn_relu_once를 그대로 쓰면 relu까지 넣어버리니까
        // 여기서는 conv+bn만 수동으로 반복한다

        // 2-1. im2col(tmp1)
        int OH = H; // stride=1,pad=1,k=3 가정이라서 spatial 유지
        int OW = W;
        int KCOL = C1out * k2H * k2W;
        int NCOL = OH * OW;

        dim3 blkIm2(16,16);
        dim3 grdIm2((OW+blkIm2.x-1)/blkIm2.x,
                    (OH+blkIm2.y-1)/blkIm2.y);

        im2col_nchw<<<grdIm2, blkIm2, 0, stream>>>(
            d_tmp1,
            N, C1out, H, W,
            k2H, k2W,
            s2H, s2W,
            p2H, p2W,
            d_colBuf
        );
        CUDA_CHECK(cudaGetLastError());

        // 2-2. GEMM: tmp2 = w2col * colBuf
        dim3 blkGemm(32,32);
        dim3 grdGemm((NCOL+blkGemm.x-1)/blkGemm.x,
                     (C2out+blkGemm.y-1)/blkGemm.y);

        sgemm_tiled<<<grdGemm, blkGemm, 0, stream>>>(
            d_w2col,
            d_colBuf,
            d_tmp2,
            C2out, NCOL, KCOL
        );
        CUDA_CHECK(cudaGetLastError());

        // 2-3. bn_inference(tmp2)
        int total_spatial = C2out * OH * OW;
        dim3 blkBN(256);
        dim3 grdBN((total_spatial+blkBN.x-1)/blkBN.x);

        bn_inference<<<grdBN, blkBN, 0, stream>>>(
            d_tmp2,
            d_bn2_gamma,
            d_bn2_beta,
            d_bn2_mean,
            d_bn2_var,
            bn2_eps,
            C2out, OH, OW
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 3. out = tmp2 + x  (identity skip)
    {
        int total = C*H*W; // C == C2out == C1out == block channels
        dim3 blkAdd(256);
        dim3 grdAdd((total+blkAdd.x-1)/blkAdd.x);

        tensor_add_fp32_kernel<<<grdAdd, blkAdd, 0, stream>>>(
            d_tmp2,
            d_x,
            d_out,
            total
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // 4. relu(out)
    {
        int total_relu = C*H*W;
        dim3 blkReLU(256);
        dim3 grdReLU((total_relu+blkReLU.x-1)/blkReLU.x);

        relu_forward<<<grdReLU, blkReLU, 0, stream>>>(
            d_out,
            total_relu
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
