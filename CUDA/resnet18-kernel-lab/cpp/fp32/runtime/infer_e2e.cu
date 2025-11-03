// cpp/fp32/runtime/infer_e2e.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include "utils.hpp"
#include "../kernels/gap_global.cuh"
#include "../kernels/maxpool2d.cuh"

// ==== 외부 커널 선언 ====
extern "C" __global__
void im2col_nchw(const float* x, int N,int C,int H,int W,
                 int kH,int kW,int sH,int sW,int pH,int pW,
                 float* col); // (C*kH*kW, N*OH*OW)

extern "C" __global__
void sgemm_tiled(const float* A, const float* B, float* C,
                 int M,int N,int K); // C[MxN] = A[MxK] * B[KxN]

extern "C" __global__
void bn_inference(float* x, const float* g, const float* b,
                  const float* m, const float* v, float eps,
                  int C,int OH,int OW);

extern "C" __global__
void relu_forward(float* x, int n);

extern "C" __global__
void add_inplace(float* y, const float* x, int n);

// ==== 런처 ====
static inline void gemm_launch(const float* A, const float* B, float* C,
                               int M, int N, int K){
    dim3 blk(32,32);
    dim3 grd( (N+blk.x-1)/blk.x, (M+blk.y-1)/blk.y );
    sgemm_tiled<<<grd,blk>>>(A, B, C, M, N, K);
}

static inline void im2col_launch(const float* x,
                                 int N,int C,int H,int W,
                                 int kH,int kW,int sH,int sW,int pH,int pW,
                                 int OH,int OW,
                                 float* col){
    dim3 blk(16,16);
    dim3 grd( (OW+blk.x-1)/blk.x, (OH+blk.y-1)/blk.y );
    im2col_nchw<<<grd,blk>>>(x, N,C,H,W, kH,kW,sH,sW,pH,pW, col);
}

static inline void bn_launch(float* y,
                             const std::vector<float>& G,
                             const std::vector<float>& B,
                             const std::vector<float>& M,
                             const std::vector<float>& V,
                             int C, int OH, int OW,
                             float eps=1e-5f){
    auto dG = copy_to_device(G);
    auto dB = copy_to_device(B);
    auto dM = copy_to_device(M);
    auto dV = copy_to_device(V);
    int total = C*OH*OW;
    bn_inference<<< div_up(total,256), 256 >>>(y, dG.get(), dB.get(), dM.get(), dV.get(), eps, C, OH, OW);
}

// ===== 공통 유틸 =====
static inline std::vector<float>
conv_to_col_OCxKCOL(const std::vector<float>& W, int OC, int C, int kH, int kW){
    const int KCOL = C*kH*kW;
    std::vector<float> Wcol(OC*KCOL);
    for (int o=0;o<OC;o++){
        for (int c=0;c<C;c++){
            for (int kh=0;kh<kH;kh++){
                for (int kw=0;kw<kW;kw++){
                    int r   = c*kH*kW + kh*kW + kw;
                    int src = o*C*kH*kW + c*kH*kW + kh*kW + kw; // (OC,C,kH,kW) contiguous
                    Wcol[o*KCOL + r] = W[src];
                }
            }
        }
    }
    return Wcol;
}

static inline void conv3x3_bn_relu(const float* dX, // [N,C,H,W]
                                   int N,int C,int H,int W,
                                   int OC, int s, // stride (1 or 2)
                                   float* dY,     // [N,OC,OH,OW]
                                   const std::vector<float>& Wraw, // [OC,C,3,3]
                                   const std::vector<float>& G,    // BN gamma
                                   const std::vector<float>& B,    // BN beta
                                   const std::vector<float>& M,    // BN mean
                                   const std::vector<float>& V){   // BN var
    const int kH=3,kW=3,pH=1,pW=1;
    const int OH = (H + 2*pH - kH)/s + 1;
    const int OW = (W + 2*pW - kW)/s + 1;
    const int KCOL = C*kH*kW;

    auto Wcol = conv_to_col_OCxKCOL(Wraw, OC, C, kH, kW);
    auto dW   = copy_to_device(Wcol);
    auto dCol = make_device_f32(KCOL * OH * OW);

    // im2col + GEMM
    im2col_launch(dX, N,C,H,W, kH,kW, s,s, pH,pW, OH,OW, dCol.get());
    gemm_launch(dW.get(), dCol.get(), dY, OC, OH*OW, KCOL);

    // BN + ReLU
    bn_launch(dY, G,B,M,V, OC, OH, OW);
    relu_forward<<< div_up(OC*OH*OW,256), 256 >>>(dY, OC*OH*OW);
}

static inline void conv3x3_bn(const float* dX,  // [N,C,H,W]
                              int N,int C,int H,int W,
                              int OC, int s,
                              float* dY,        // [N,OC,OH,OW]
                              const std::vector<float>& Wraw, // [OC,C,3,3]
                              const std::vector<float>& G,
                              const std::vector<float>& B,
                              const std::vector<float>& M,
                              const std::vector<float>& V){
    const int kH=3,kW=3,pH=1,pW=1;
    const int OH = (H + 2*pH - kH)/s + 1;
    const int OW = (W + 2*pW - kW)/s + 1;
    const int KCOL = C*kH*kW;

    auto Wcol = conv_to_col_OCxKCOL(Wraw, OC, C, kH, kW);
    auto dW   = copy_to_device(Wcol);
    auto dCol = make_device_f32(KCOL * OH * OW);

    im2col_launch(dX, N,C,H,W, kH,kW, s,s, pH,pW, OH,OW, dCol.get());
    gemm_launch(dW.get(), dCol.get(), dY, OC, OH*OW, KCOL);
    bn_launch(dY, G,B,M,V, OC, OH, OW);
    // (relu 없는 변형; 필요한 곳에서 따로 relu)
}

static inline void conv1x1_bn(const float* dX,
                              int N,int C,int H,int W,
                              int OC,int s,
                              float* dY,
                              const std::vector<float>& Wraw, // [OC,C,1,1]
                              const std::vector<float>& G,
                              const std::vector<float>& B,
                              const std::vector<float>& M,
                              const std::vector<float>& V){
    const int kH=1,kW=1,pH=0,pW=0;
    const int OH = (H + 2*pH - kH)/s + 1;
    const int OW = (W + 2*pW - kW)/s + 1;
    const int KCOL = C*kH*kW;

    auto Wcol = conv_to_col_OCxKCOL(Wraw, OC, C, kH, kW);
    auto dW   = copy_to_device(Wcol);
    auto dCol = make_device_f32(KCOL * OH * OW);

    im2col_launch(dX, N,C,H,W, kH,kW, s,s, pH,pW, OH,OW, dCol.get());
    gemm_launch(dW.get(), dCol.get(), dY, OC, OH*OW, KCOL);
    bn_launch(dY, G,B,M,V, OC, OH, OW);
}

static inline void basic_block_identity( // stride=1, C==OC
        const float* dIn, int N,int C,int H,int W,
        float* dOut,
        // conv1
        const std::vector<float>& W1, const std::vector<float>& G1, const std::vector<float>& B1,
        const std::vector<float>& M1, const std::vector<float>& V1,
        // conv2 (no relu after)
        const std::vector<float>& W2, const std::vector<float>& G2, const std::vector<float>& B2,
        const std::vector<float>& M2, const std::vector<float>& V2){
    // tmp after conv1
    auto dTmp = make_device_f32(C*H*W);
    conv3x3_bn_relu(dIn, N,C,H,W, C, /*s=*/1, dTmp.get(), W1,G1,B1,M1,V1);
    // conv2 + bn (no relu)
    conv3x3_bn(dTmp.get(), N,C,H,W, C, /*s=*/1, dOut, W2,G2,B2,M2,V2);
    // add + relu
    add_inplace<<< div_up(C*H*W,256), 256 >>>(dOut, dIn, C*H*W);
    relu_forward<<< div_up(C*H*W,256), 256 >>>(dOut, C*H*W);
}

static inline void basic_block_downsample( // stride=2, C!=OC
        const float* dIn, int N,int C,int H,int W,
        int OC,
        float* dOut,
        // conv1 (stride=2)
        const std::vector<float>& W1, const std::vector<float>& G1, const std::vector<float>& B1,
        const std::vector<float>& M1, const std::vector<float>& V1,
        // conv2 (no relu)
        const std::vector<float>& W2, const std::vector<float>& G2, const std::vector<float>& B2,
        const std::vector<float>& M2, const std::vector<float>& V2,
        // downsample 1x1 s=2
        const std::vector<float>& Wd, const std::vector<float>& Gd, const std::vector<float>& Bd,
        const std::vector<float>& Md, const std::vector<float>& Vd){
    // conv1 + relu (s=2)
    const int OH = (H + 2*1 - 3)/2 + 1; // for 3x3 s=2, p=1
    const int OW = (W + 2*1 - 3)/2 + 1;
    auto dTmp = make_device_f32(OC*OH*OW);
    conv3x3_bn_relu(dIn, N,C,H,W, OC, /*s=*/2, dTmp.get(), W1,G1,B1,M1,V1);
    // conv2 + bn (no relu)
    conv3x3_bn(dTmp.get(), N,OC,OH,OW, OC, /*s=*/1, dOut, W2,G2,B2,M2,V2);
    // proj path
    auto dProj = make_device_f32(OC*OH*OW);
    conv1x1_bn(dIn, N,C,H,W, OC, /*s=*/2, dProj.get(), Wd,Gd,Bd,Md,Vd);
    // add + relu
    add_inplace<<< div_up(OC*OH*OW,256), 256 >>>(dOut, dProj.get(), OC*OH*OW);
    relu_forward<<< div_up(OC*OH*OW,256), 256 >>>(dOut, OC*OH*OW);
}

// ---- FC (GAP[512] -> 1000) ----
static inline void fc_forward(const float* gap,            // [512]
                              const std::vector<float>& W, // [1000,512]
                              const std::vector<float>& B, // [1000]
                              std::vector<float>& out){    // [1000]
    const int O = 1000, I = 512;
    auto dGap = copy_to_device(gap, I);
    auto dW   = copy_to_device(W);
    auto dOut = make_device_f32(O);
    gemm_launch(dW.get(), dGap.get(), dOut.get(), O, 1, I);
    out = copy_to_host(dOut, O);
    for (int o=0;o<O;o++) out[o] += B[o];
}

static void usage(){
    std::cout <<
      "Usage:\n"
      "  step8_e2e --manifest <dir> --input <input.bin>\n";
}

int main(int argc, char** argv){
    std::string mani, input_path;
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) mani = argv[++i];
        else if (a=="--input" && i+1<argc) input_path = argv[++i];
    }
    if (mani.empty() || input_path.empty()){ usage(); return 1; }

    // ========= 0) 입력 =========
    const int N=1, C0=3, H0=224, W0=224;
    auto X = load_bin_f32(input_path, N*C0*H0*W0);
    auto dX = copy_to_device(X);

    // ========= 1) Stem: conv1(7x7,s2,p3)->BN1->ReLU->MaxPool(3x3,s2,p1) =========
    const int OC1=64, KH=7, KW=7, SH=2, SW=2, PH=3, PW=3;
    const int H1 = (H0+2*PH-KH)/SH + 1; // 112
    const int W1 = (W0+2*PW-KW)/SW + 1; // 112
    const int KCOL1=C0*KH*KW;

    auto Wc1 = load_bin_f32(mani + "/conv1.weight.bin", OC1*C0*KH*KW);
    auto G1  = load_bin_f32(mani + "/bn1.weight.bin",       OC1);
    auto B1  = load_bin_f32(mani + "/bn1.bias.bin",         OC1);
    auto M1  = load_bin_f32(mani + "/bn1.running_mean.bin", OC1);
    auto V1  = load_bin_f32(mani + "/bn1.running_var.bin",  OC1);

    auto Wcol1 = conv_to_col_OCxKCOL(Wc1, OC1, C0, KH, KW);
    auto dW1   = copy_to_device(Wcol1);
    auto dCol1 = make_device_f32(KCOL1 * H1 * W1);
    auto dY1   = make_device_f32(OC1 * H1 * W1);

    im2col_launch(dX.get(), N,C0,H0,W0, KH,KW, SH,SW, PH,PW, H1,W1, dCol1.get());
    gemm_launch(dW1.get(), dCol1.get(), dY1.get(), OC1, H1*W1, KCOL1);
    bn_launch(dY1.get(), G1,B1,M1,V1, OC1, H1, W1);
    relu_forward<<< div_up(OC1*H1*W1,256), 256 >>>(dY1.get(), OC1*H1*W1);

    // MaxPool 3x3 s2 p1 -> 64x56x56
    const int MPH=3, MPW=3, MPS=2, MPP=1;
    const int H1p = (H1 + 2*MPP - MPH)/MPS + 1; // 56
    const int W1p = (W1 + 2*MPP - MPW)/MPS + 1; // 56
    auto dP1 = make_device_f32(OC1 * H1p * W1p);
    {
        dim3 blk(1,1,1);
        dim3 grd(H1p*W1p, OC1, N);
        maxpool2d_3x3_s2p1_nchw<<<grd, blk>>>(dY1.get(), N, OC1, H1, W1, dP1.get());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ========= 2) Layer1 (64) =========
    // block0 (stride=1, identity)
    {
        auto W10 = load_bin_f32(mani + "/layer1.0.conv1.weight.bin", 64*64*3*3);
        auto G10 = load_bin_f32(mani + "/layer1.0.bn1.weight.bin",   64);
        auto B10 = load_bin_f32(mani + "/layer1.0.bn1.bias.bin",     64);
        auto M10 = load_bin_f32(mani + "/layer1.0.bn1.running_mean.bin",64);
        auto V10 = load_bin_f32(mani + "/layer1.0.bn1.running_var.bin", 64);

        auto W11 = load_bin_f32(mani + "/layer1.0.conv2.weight.bin", 64*64*3*3);
        auto G11 = load_bin_f32(mani + "/layer1.0.bn2.weight.bin",   64);
        auto B11 = load_bin_f32(mani + "/layer1.0.bn2.bias.bin",     64);
        auto M11 = load_bin_f32(mani + "/layer1.0.bn2.running_mean.bin",64);
        auto V11 = load_bin_f32(mani + "/layer1.0.bn2.running_var.bin", 64);

        auto dL10 = make_device_f32(64*H1p*W1p);
        basic_block_identity(dP1.get(), N,64,H1p,W1p, dL10.get(),
                             W10,G10,B10,M10,V10,  W11,G11,B11,M11,V11);
        dP1 = std::move(dL10);
    }
    // block1 (stride=1, identity)
    {
        auto W10 = load_bin_f32(mani + "/layer1.1.conv1.weight.bin", 64*64*3*3);
        auto G10 = load_bin_f32(mani + "/layer1.1.bn1.weight.bin",   64);
        auto B10 = load_bin_f32(mani + "/layer1.1.bn1.bias.bin",     64);
        auto M10 = load_bin_f32(mani + "/layer1.1.bn1.running_mean.bin",64);
        auto V10 = load_bin_f32(mani + "/layer1.1.bn1.running_var.bin", 64);

        auto W11 = load_bin_f32(mani + "/layer1.1.conv2.weight.bin", 64*64*3*3);
        auto G11 = load_bin_f32(mani + "/layer1.1.bn2.weight.bin",   64);
        auto B11 = load_bin_f32(mani + "/layer1.1.bn2.bias.bin",     64);
        auto M11 = load_bin_f32(mani + "/layer1.1.bn2.running_mean.bin",64);
        auto V11 = load_bin_f32(mani + "/layer1.1.bn2.running_var.bin", 64);

        auto dL11 = make_device_f32(64*H1p*W1p);
        basic_block_identity(dP1.get(), N,64,H1p,W1p, dL11.get(),
                             W10,G10,B10,M10,V10,  W11,G11,B11,M11,V11);
        dP1 = std::move(dL11);
    }
    // 현재: 64x56x56

    // ========= 3) Layer2 (128, downsample) =========
    // block0 (stride=2 + proj)
    int H2 = (H1p + 2*1 - 3)/2 + 1; // 28
    int W2 = (W1p + 2*1 - 3)/2 + 1; // 28
    {
        auto W20 = load_bin_f32(mani + "/layer2.0.conv1.weight.bin", 128*64*3*3);
        auto G20 = load_bin_f32(mani + "/layer2.0.bn1.weight.bin",   128);
        auto B20 = load_bin_f32(mani + "/layer2.0.bn1.bias.bin",     128);
        auto M20 = load_bin_f32(mani + "/layer2.0.bn1.running_mean.bin",128);
        auto V20 = load_bin_f32(mani + "/layer2.0.bn1.running_var.bin", 128);

        auto W21 = load_bin_f32(mani + "/layer2.0.conv2.weight.bin", 128*128*3*3);
        auto G21 = load_bin_f32(mani + "/layer2.0.bn2.weight.bin",   128);
        auto B21 = load_bin_f32(mani + "/layer2.0.bn2.bias.bin",     128);
        auto M21 = load_bin_f32(mani + "/layer2.0.bn2.running_mean.bin",128);
        auto V21 = load_bin_f32(mani + "/layer2.0.bn2.running_var.bin", 128);

        auto W2d = load_bin_f32(mani + "/layer2.0.downsample.0.weight.bin", 128*64*1*1);
        auto G2d = load_bin_f32(mani + "/layer2.0.downsample.1.weight.bin", 128);
        auto B2d = load_bin_f32(mani + "/layer2.0.downsample.1.bias.bin",   128);
        auto M2d = load_bin_f32(mani + "/layer2.0.downsample.1.running_mean.bin",128);
        auto V2d = load_bin_f32(mani + "/layer2.0.downsample.1.running_var.bin", 128);

        auto dL20 = make_device_f32(128*H2*W2);
        basic_block_downsample(dP1.get(), N,64,H1p,W1p, 128, dL20.get(),
                               W20,G20,B20,M20,V20,   W21,G21,B21,M21,V21,
                               W2d,G2d,B2d,M2d,V2d);
        dP1 = std::move(dL20);
    }
    // block1 (stride=1, identity) — in:128x28x28
    {
        auto W20 = load_bin_f32(mani + "/layer2.1.conv1.weight.bin", 128*128*3*3);
        auto G20 = load_bin_f32(mani + "/layer2.1.bn1.weight.bin",   128);
        auto B20 = load_bin_f32(mani + "/layer2.1.bn1.bias.bin",     128);
        auto M20 = load_bin_f32(mani + "/layer2.1.bn1.running_mean.bin",128);
        auto V20 = load_bin_f32(mani + "/layer2.1.bn1.running_var.bin", 128);

        auto W21 = load_bin_f32(mani + "/layer2.1.conv2.weight.bin", 128*128*3*3);
        auto G21 = load_bin_f32(mani + "/layer2.1.bn2.weight.bin",   128);
        auto B21 = load_bin_f32(mani + "/layer2.1.bn2.bias.bin",     128);
        auto M21 = load_bin_f32(mani + "/layer2.1.bn2.running_mean.bin",128);
        auto V21 = load_bin_f32(mani + "/layer2.1.bn2.running_var.bin", 128);

        auto dL21 = make_device_f32(128*H2*W2);
        basic_block_identity(dP1.get(), N,128,H2,W2, dL21.get(),
                             W20,G20,B20,M20,V20,  W21,G21,B21,M21,V21);
        dP1 = std::move(dL21);
    }

    // ========= 4) Layer3 (256, downsample) =========
    int H3 = (H2 + 2*1 - 3)/2 + 1; // 14
    int W3 = (W2 + 2*1 - 3)/2 + 1; // 14
    {
        auto W30 = load_bin_f32(mani + "/layer3.0.conv1.weight.bin", 256*128*3*3);
        auto G30 = load_bin_f32(mani + "/layer3.0.bn1.weight.bin",   256);
        auto B30 = load_bin_f32(mani + "/layer3.0.bn1.bias.bin",     256);
        auto M30 = load_bin_f32(mani + "/layer3.0.bn1.running_mean.bin",256);
        auto V30 = load_bin_f32(mani + "/layer3.0.bn1.running_var.bin", 256);

        auto W31 = load_bin_f32(mani + "/layer3.0.conv2.weight.bin", 256*256*3*3);
        auto G31 = load_bin_f32(mani + "/layer3.0.bn2.weight.bin",   256);
        auto B31 = load_bin_f32(mani + "/layer3.0.bn2.bias.bin",     256);
        auto M31 = load_bin_f32(mani + "/layer3.0.bn2.running_mean.bin",256);
        auto V31 = load_bin_f32(mani + "/layer3.0.bn2.running_var.bin", 256);

        auto W3d = load_bin_f32(mani + "/layer3.0.downsample.0.weight.bin", 256*128*1*1);
        auto G3d = load_bin_f32(mani + "/layer3.0.downsample.1.weight.bin", 256);
        auto B3d = load_bin_f32(mani + "/layer3.0.downsample.1.bias.bin",   256);
        auto M3d = load_bin_f32(mani + "/layer3.0.downsample.1.running_mean.bin",256);
        auto V3d = load_bin_f32(mani + "/layer3.0.downsample.1.running_var.bin", 256);

        auto dL30 = make_device_f32(256*H3*W3);
        basic_block_downsample(dP1.get(), N,128,H2,W2, 256, dL30.get(),
                               W30,G30,B30,M30,V30,   W31,G31,B31,M31,V31,
                               W3d,G3d,B3d,M3d,V3d);
        dP1 = std::move(dL30);
    }
    // block1 (stride=1, identity) — in:256x14x14
    {
        auto W30 = load_bin_f32(mani + "/layer3.1.conv1.weight.bin", 256*256*3*3);
        auto G30 = load_bin_f32(mani + "/layer3.1.bn1.weight.bin",   256);
        auto B30 = load_bin_f32(mani + "/layer3.1.bn1.bias.bin",     256);
        auto M30 = load_bin_f32(mani + "/layer3.1.bn1.running_mean.bin",256);
        auto V30 = load_bin_f32(mani + "/layer3.1.bn1.running_var.bin", 256);

        auto W31 = load_bin_f32(mani + "/layer3.1.conv2.weight.bin", 256*256*3*3);
        auto G31 = load_bin_f32(mani + "/layer3.1.bn2.weight.bin",   256);
        auto B31 = load_bin_f32(mani + "/layer3.1.bn2.bias.bin",     256);
        auto M31 = load_bin_f32(mani + "/layer3.1.bn2.running_mean.bin",256);
        auto V31 = load_bin_f32(mani + "/layer3.1.bn2.running_var.bin", 256);

        auto dL31 = make_device_f32(256*H3*W3);
        basic_block_identity(dP1.get(), N,256,H3,W3, dL31.get(),
                             W30,G30,B30,M30,V30,  W31,G31,B31,M31,V31);
        dP1 = std::move(dL31);
    }

    // ========= 5) Layer4 (512, downsample) =========
    int H4 = (H3 + 2*1 - 3)/2 + 1; // 7
    int W4 = (W3 + 2*1 - 3)/2 + 1; // 7
    {
        auto W40 = load_bin_f32(mani + "/layer4.0.conv1.weight.bin", 512*256*3*3);
        auto G40 = load_bin_f32(mani + "/layer4.0.bn1.weight.bin",   512);
        auto B40 = load_bin_f32(mani + "/layer4.0.bn1.bias.bin",     512);
        auto M40 = load_bin_f32(mani + "/layer4.0.bn1.running_mean.bin",512);
        auto V40 = load_bin_f32(mani + "/layer4.0.bn1.running_var.bin", 512);

        auto W41 = load_bin_f32(mani + "/layer4.0.conv2.weight.bin", 512*512*3*3);
        auto G41 = load_bin_f32(mani + "/layer4.0.bn2.weight.bin",   512);
        auto B41 = load_bin_f32(mani + "/layer4.0.bn2.bias.bin",     512);
        auto M41 = load_bin_f32(mani + "/layer4.0.bn2.running_mean.bin",512);
        auto V41 = load_bin_f32(mani + "/layer4.0.bn2.running_var.bin", 512);

        auto W4d = load_bin_f32(mani + "/layer4.0.downsample.0.weight.bin", 512*256*1*1);
        auto G4d = load_bin_f32(mani + "/layer4.0.downsample.1.weight.bin", 512);
        auto B4d = load_bin_f32(mani + "/layer4.0.downsample.1.bias.bin",   512);
        auto M4d = load_bin_f32(mani + "/layer4.0.downsample.1.running_mean.bin",512);
        auto V4d = load_bin_f32(mani + "/layer4.0.downsample.1.running_var.bin", 512);

        auto dL40 = make_device_f32(512*H4*W4);
        basic_block_downsample(dP1.get(), N,256,H3,W3, 512, dL40.get(),
                               W40,G40,B40,M40,V40,   W41,G41,B41,M41,V41,
                               W4d,G4d,B4d,M4d,V4d);
        dP1 = std::move(dL40);
    }
    // block1 (stride=1, identity) — in:512x7x7
    {
        auto W40 = load_bin_f32(mani + "/layer4.1.conv1.weight.bin", 512*512*3*3);
        auto G40 = load_bin_f32(mani + "/layer4.1.bn1.weight.bin",   512);
        auto B40 = load_bin_f32(mani + "/layer4.1.bn1.bias.bin",     512);
        auto M40 = load_bin_f32(mani + "/layer4.1.bn1.running_mean.bin",512);
        auto V40 = load_bin_f32(mani + "/layer4.1.bn1.running_var.bin", 512);

        auto W41 = load_bin_f32(mani + "/layer4.1.conv2.weight.bin", 512*512*3*3);
        auto G41 = load_bin_f32(mani + "/layer4.1.bn2.weight.bin",   512);
        auto B41 = load_bin_f32(mani + "/layer4.1.bn2.bias.bin",     512);
        auto M41 = load_bin_f32(mani + "/layer4.1.bn2.running_mean.bin",512);
        auto V41 = load_bin_f32(mani + "/layer4.1.bn2.running_var.bin", 512);

        auto dL41 = make_device_f32(512*H4*W4);
        basic_block_identity(dP1.get(), N,512,H4,W4, dL41.get(),
                             W40,G40,B40,M40,V40,  W41,G41,B41,M41,V41);
        dP1 = std::move(dL41);
    }

    // ========= 6) GAP + FC =========
    auto dGAP = make_device_f32(512);
    {
        dim3 blk(256);
        dim3 grd(div_up(512, (int)blk.x));
        gap_global<<<grd, blk>>>(dP1.get(), 512, 7, 7, dGAP.get());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto hGAP = copy_to_host(dGAP, 512);

    auto Wfc = load_bin_f32(mani + "/fc.weight.bin", 1000*512);
    auto Bfc = load_bin_f32(mani + "/fc.bias.bin",   1000);

    std::vector<float> logits;
    fc_forward(hGAP.data(), Wfc, Bfc, logits);

    // 출력
    int top=-1; float best=-1e30f;
    for (int i=0;i<1000;i++){ if (logits[i] > best){ best=logits[i]; top=i; } }
    std::cout<<"[E2E] top-1 class index = "<<top<<", logit="<<best<<"\n";
    std::cout<<"(주의: synset 매핑은 별도)\n";
    return 0;
}
