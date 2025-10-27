/*
infer_stem_block0.cu

Step3 runner:
    1) conv1 -> bn1 -> relu
    2) maxpool (3x3, stride=2, pad=1)
    3) basic_block (layer1[0], identity skip)
    4) compare against PyTorch dump (sample_block0_out.bin)

Inputs:
    --manifest <dir>   exports/resnet18/fp32
    --input    <bin>   sample_input.bin          # N=1,C=3,H=224,W=224
    --expect   <bin>   sample_block0_out.bin     # after stem+block0

Assumptions:
    - N=1 always
    - conv1: 7x7, stride=2, pad=3, inC=3, outC=64
    - maxpool: 3x3, stride=2, pad=1
    - basic_block: C=64, H=W=56, identity skip (layer1[0])
*/

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cuda_runtime.h>
#include "utils.hpp"

// ===== device kernels (must match definitions in kernels/*.cu) =====

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
void maxpool2d_kernel_fp32(
    const float* in,
    float* out,
    int N, int C, int H, int W,
    int kernel_h, int kernel_w,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int H_out, int W_out);

extern "C" __global__
void tensor_add_fp32_kernel(const float* a,
                            const float* b,
                            float* out,
                            int total);

// ===== basic block host API (from kernels/basic_block.cu) =====
extern "C"
void basic_block_fp32_forward_identity(
    const float* d_x,        // input / identity skip
    float* d_out,            // final output
    float* d_tmp1,           // scratch after conv1/bn/relu
    float* d_tmp2,           // scratch after conv2/bn
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
);

// ---------------------------------------------------------------
// stem config for conv1/bn1/relu
struct StemCfg {
    // input  : [N=1, C=3, H=224, W=224]
    // conv1  : 7x7 stride2 pad3, outC=64
    // output : [1, 64, 112, 112]
    int N = 1;
    int C = 3;
    int H = 224;
    int W = 224;

    int OC = 64;
    int KH = 7;
    int KW = 7;
    int SH = 2;
    int SW = 2;
    int PH = 3;
    int PW = 3;

    int OH = (H + 2*PH - KH)/SH + 1; // 112
    int OW = (W + 2*PW - KW)/SW + 1; // 112

    // im2col dims
    int KCOL = C * KH * KW;   // rows per output pixel
    int NCOL = OH * OW;       // number of output pixels (N=1 assumed)
};

// after maxpool(3x3,s2,p1):
// shape -> [1, 64, 56, 56]
struct PoolCfg {
    int N = 1;
    int C = 64;
    int H_in = 112;
    int W_in = 112;

    int KH = 3;
    int KW = 3;
    int SH = 2;
    int SW = 2;
    int PH = 1;
    int PW = 1;

    int H_out = (H_in + 2*PH - KH)/SH + 1; // 56
    int W_out = (W_in + 2*PW - KW)/SW + 1; // 56
};

// basic block cfg for layer1[0]:
// keeps C=64, spatial 56x56
struct Block0Cfg {
    int N = 1;
    int C = 64;
    int H = 56;
    int W = 56;

    // convs in block0 are 3x3 stride1 pad1, so spatial unchanged
    int KH = 3;
    int KW = 3;
    int SH = 1;
    int SW = 1;
    int PH = 1;
    int PW = 1;

    int OH = 56;
    int OW = 56;

    int KCOL = C * KH * KW;    // 64 * 3 * 3 = 576
    int NCOL = OH * OW;        // 3136
};

// ----------------------------------------------------------------
// helper: flatten OIHW weight (OC,IC,KH,KW) -> [OC, IC*KH*KW]
static void flatten_OIHW_to_OxKCOL(
    const std::vector<float> &w, // size OC*IC*KH*KW
    int OC, int IC, int KH, int KW,
    std::vector<float> &Wcol    // size OC * (IC*KH*KW)
){
    int KCOL = IC * KH * KW;
    Wcol.resize(OC * KCOL);
    for (int o=0;o<OC;o++){
        for (int c=0;c<IC;c++){
            for (int kh=0;kh<KH;kh++){
                for (int kw=0;kw<KW;kw++){
                    int r   = c*KH*KW + kh*KW + kw; // row idx within KCOL
                    int src = o*IC*KH*KW + c*KH*KW + kh*KW + kw;
                    Wcol[o*KCOL + r] = w[src];
                }
            }
        }
    }
}

// usage info
static void usage(){
    std::cout
        << "--manifest exports/resnet18/fp32 "
        << "--input exports/resnet18/fp32/sample_input.bin "
        << "--expect exports/resnet18/fp32/sample_block0_out.bin\n";
}

// ===============================================================

int main(int argc, char** argv){
    std::string maniDir;
    std::string inputPath;
    std::string expectPath;

    // parse CLI
    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest" && i+1<argc) {
            maniDir = argv[++i];
        } else if (a=="--input" && i+1<argc) {
            inputPath = argv[++i];
        } else if (a=="--expect" && i+1<argc) {
            expectPath = argv[++i];
        }
    }
    if (maniDir.empty()||inputPath.empty()||expectPath.empty()){
        usage(); return 1;
    }

    StemCfg stem;
    PoolCfg pool;
    Block0Cfg blk0;

    // ---------------------------
    // 1) Load weights / params from disk (host)
    // stem conv1/bn1
    auto conv1_w  = load_bin_f32(maniDir + "/conv1.weight.bin",
                                 stem.OC * stem.C * stem.KH * stem.KW);
    auto bn1_w    = load_bin_f32(maniDir + "/bn1.weight.bin", stem.OC);
    auto bn1_b    = load_bin_f32(maniDir + "/bn1.bias.bin",   stem.OC);
    auto bn1_m    = load_bin_f32(maniDir + "/bn1.running_mean.bin", stem.OC);
    auto bn1_v    = load_bin_f32(maniDir + "/bn1.running_var.bin",  stem.OC);

    // first basic block (layer1.0)
    // conv1
    auto l10c1_w  = load_bin_f32(maniDir + "/layer1.0.conv1.weight.bin",
                                 blk0.C * blk0.C * blk0.KH * blk0.KW);
    auto l10bn1_w = load_bin_f32(maniDir + "/layer1.0.bn1.weight.bin", blk0.C);
    auto l10bn1_b = load_bin_f32(maniDir + "/layer1.0.bn1.bias.bin",   blk0.C);
    auto l10bn1_m = load_bin_f32(maniDir + "/layer1.0.bn1.running_mean.bin", blk0.C);
    auto l10bn1_v = load_bin_f32(maniDir + "/layer1.0.bn1.running_var.bin",  blk0.C);

    // conv2
    auto l10c2_w  = load_bin_f32(maniDir + "/layer1.0.conv2.weight.bin",
                                 blk0.C * blk0.C * blk0.KH * blk0.KW);
    auto l10bn2_w = load_bin_f32(maniDir + "/layer1.0.bn2.weight.bin", blk0.C);
    auto l10bn2_b = load_bin_f32(maniDir + "/layer1.0.bn2.bias.bin",   blk0.C);
    auto l10bn2_m = load_bin_f32(maniDir + "/layer1.0.bn2.running_mean.bin", blk0.C);
    auto l10bn2_v = load_bin_f32(maniDir + "/layer1.0.bn2.running_var.bin",  blk0.C);

    // ---------------------------
    // 2) Load input / expected reference
    auto x_in     = load_bin_f32(inputPath,
                                 stem.N * stem.C * stem.H * stem.W);
    auto y_expect = load_bin_f32(expectPath,
                                 blk0.N * blk0.C * blk0.H * blk0.W);

    // ---------------------------
    // 3) Prepare flattened weights for GEMM form
    std::vector<float> conv1_Wcol;
    flatten_OIHW_to_OxKCOL(conv1_w,
                           stem.OC, stem.C,
                           stem.KH, stem.KW,
                           conv1_Wcol);

    std::vector<float> l10c1_Wcol;
    flatten_OIHW_to_OxKCOL(l10c1_w,
                           blk0.C, blk0.C,
                           blk0.KH, blk0.KW,
                           l10c1_Wcol);

    std::vector<float> l10c2_Wcol;
    flatten_OIHW_to_OxKCOL(l10c2_w,
                           blk0.C, blk0.C,
                           blk0.KH, blk0.KW,
                           l10c2_Wcol);

    // ---------------------------
    // 4) Allocate device buffers

    // stem conv1 output: [1,64,112,112]
    size_t stem_out_elems = stem.OC * stem.OH * stem.OW;
    // after maxpool: [1,64,56,56]
    size_t pool_out_elems = pool.N * pool.C * pool.H_out * pool.W_out;
    // block0 final out: [1,64,56,56]
    size_t blk0_elems     = blk0.N * blk0.C * blk0.H * blk0.W;

    // im2col workspace sizes
    size_t stem_col_elems = stem.KCOL * stem.NCOL;   // (3*7*7=147 * 112*112)
    size_t blk0_col_elems = blk0.KCOL * blk0.NCOL;   // (64*3*3=576 * 56*56=3136)

    // device ptrs
    float *dInput      = nullptr; // original input
    float *dCol_stem   = nullptr;
    float *dW_stem     = nullptr;
    float *dStemOut    = nullptr;
    float *dPoolOut    = nullptr;

    float *dBn1_w=nullptr,*dBn1_b=nullptr,*dBn1_m=nullptr,*dBn1_v=nullptr;

    // block0 buffers
    float *dCol_blk0   = nullptr;
    float *dW_l10c1    = nullptr;
    float *dW_l10c2    = nullptr;

    float *dBlkTmp1    = nullptr; // after block conv1/bn/relu
    float *dBlkTmp2    = nullptr; // after block conv2/bn
    float *dBlkOut     = nullptr; // final block output

    float *dBn_l10bn1_w=nullptr,*dBn_l10bn1_b=nullptr,*dBn_l10bn1_m=nullptr,*dBn_l10bn1_v=nullptr;
    float *dBn_l10bn2_w=nullptr,*dBn_l10bn2_b=nullptr,*dBn_l10bn2_m=nullptr,*dBn_l10bn2_v=nullptr;

    CUDA_CHECK(cudaMalloc(&dInput,    x_in.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dCol_stem, stem_col_elems*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_stem,   conv1_Wcol.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dStemOut,  stem_out_elems*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dPoolOut,  pool_out_elems*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBn1_w, stem.OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1_b, stem.OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1_m, stem.OC*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn1_v, stem.OC*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dCol_blk0, blk0_col_elems*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_l10c1,  l10c1_Wcol.size()*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dW_l10c2,  l10c2_Wcol.size()*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBlkTmp1,  blk0_elems*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlkTmp2,  blk0_elems*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBlkOut,   blk0_elems*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBn_l10bn1_w, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn1_b, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn1_m, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn1_v, blk0.C*sizeof(float)));

    CUDA_CHECK(cudaMalloc(&dBn_l10bn2_w, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn2_b, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn2_m, blk0.C*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dBn_l10bn2_v, blk0.C*sizeof(float)));

    // copy host->device
    CUDA_CHECK(cudaMemcpy(dInput,   x_in.data(), x_in.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW_stem,  conv1_Wcol.data(), conv1_Wcol.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1_w,   bn1_w.data(),     stem.OC*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1_b,   bn1_b.data(),     stem.OC*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1_m,   bn1_m.data(),     stem.OC*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn1_v,   bn1_v.data(),     stem.OC*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dW_l10c1, l10c1_Wcol.data(), l10c1_Wcol.size()*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW_l10c2, l10c2_Wcol.data(), l10c2_Wcol.size()*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dBn_l10bn1_w, l10bn1_w.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn1_b, l10bn1_b.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn1_m, l10bn1_m.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn1_v, l10bn1_v.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(dBn_l10bn2_w, l10bn2_w.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn2_b, l10bn2_b.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn2_m, l10bn2_m.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dBn_l10bn2_v, l10bn2_v.data(), blk0.C*sizeof(float), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------------
    // 5) Run stem: conv1 -> bn1 -> relu
    Timer T;
    float ms_im2col, ms_gemm, ms_bn, ms_relu, ms_pool, ms_block0;

    cudaStream_t stream = nullptr; // default stream (0)

    // (a) im2col for stem conv1
    {
        dim3 blk(16,16);
        dim3 grd((stem.OW + blk.x -1)/blk.x,
                 (stem.OH + blk.y -1)/blk.y);

        T.start();
        im2col_nchw<<<grd, blk, 0, stream>>>(
            dInput,
            stem.N, stem.C, stem.H, stem.W,
            stem.KH, stem.KW,
            stem.SH, stem.SW,
            stem.PH, stem.PW,
            dCol_stem
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_im2col = T.stop();
    }

    // (b) GEMM: stem conv1
    {
        dim3 blk(32,32);
        dim3 grd((stem.NCOL + blk.x -1)/blk.x,
                 (stem.OC   + blk.y -1)/blk.y);

        T.start();
        sgemm_tiled<<<grd, blk, 0, stream>>>(
            dW_stem,      // [64, 3*7*7]
            dCol_stem,    // [3*7*7, OH*OW]
            dStemOut,     // [64, OH*OW] -> interpreted as [1,64,OH,OW]
            stem.OC,      // M
            stem.NCOL,    // N_
            stem.KCOL     // K
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_gemm = T.stop();
    }

    // (c) BN stem
    {
        int OH = stem.OH;
        int OW = stem.OW;
        int total_spatial = stem.OC * OH * OW;

        dim3 blk(256);
        dim3 grd((total_spatial + blk.x -1)/blk.x);

        T.start();
        bn_inference<<<grd, blk, 0, stream>>>(
            dStemOut,
            dBn1_w,
            dBn1_b,
            dBn1_m,
            dBn1_v,
            1e-5f,
            stem.OC, OH, OW
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_bn = T.stop();
    }

    // (d) ReLU stem
    {
        int total_relu = stem.OC * stem.OH * stem.OW;
        dim3 blk(256);
        dim3 grd((total_relu + blk.x -1)/blk.x);

        T.start();
        relu_forward<<<grd, blk, 0, stream>>>(
            dStemOut,
            total_relu
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_relu = T.stop();
    }

    // ----------------------------------------------------------------
    // 6) MaxPool 3x3 s2 p1 -> output 56x56
    {
        int N = pool.N;
        int C = pool.C;
        int H_in = pool.H_in;
        int W_in = pool.W_in;
        int H_out = pool.H_out;
        int W_out = pool.W_out;

        int total_pool = N*C*H_out*W_out;
        dim3 blk(256);
        dim3 grd((total_pool + blk.x -1)/blk.x);

        T.start();
        maxpool2d_kernel_fp32<<<grd, blk, 0, stream>>>(
            dStemOut,   // in: [1,64,112,112]
            dPoolOut,   // out:[1,64,56,56]
            N, C, H_in, W_in,
            pool.KH, pool.KW,
            pool.SH, pool.SW,
            pool.PH, pool.PW,
            H_out, W_out
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_pool = T.stop();
    }

    // 디버그: stem after maxpool (shape [1,64,56,56])
    {
        std::vector<float> stem_after_pool(pool_out_elems);
        CUDA_CHECK(cudaMemcpy(
            stem_after_pool.data(),
            dPoolOut,
            pool_out_elems * sizeof(float),
            cudaMemcpyDeviceToHost
        ));

        // 임시로 파일로 떨어뜨리자
        FILE* f = fopen("debug_stem_after_pool.bin", "wb");
        fwrite(stem_after_pool.data(), sizeof(float), stem_after_pool.size(), f);
        fclose(f);
    }

    // ----------------------------------------------------------------
    // 7) BasicBlock layer1[0]
    {
        // basic_block_fp32_forward_identity(
        //     d_x, d_out, d_tmp1, d_tmp2, d_colBuf,
        //     d_w1col, bn1..., d_w2col, bn2..., N,C,H,W, stream)

        T.start();
        basic_block_fp32_forward_identity(
            dPoolOut,
            dBlkOut,
            dBlkTmp1,
            dBlkTmp2,
            dCol_blk0,
            dW_l10c1,
            dBn_l10bn1_w,
            dBn_l10bn1_b,
            dBn_l10bn1_m,
            dBn_l10bn1_v,
            1e-5f,
            dW_l10c2,
            dBn_l10bn2_w,
            dBn_l10bn2_b,
            dBn_l10bn2_m,
            dBn_l10bn2_v,
            1e-5f,
            blk0.N, blk0.C, blk0.H, blk0.W,
            stream
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        ms_block0 = T.stop();
    }

    // ----------------------------------------------------------------
    // 8) Copy result back & compare
    std::vector<float> y_out(blk0_elems);
    CUDA_CHECK(cudaMemcpy(y_out.data(), dBlkOut,
                          blk0_elems*sizeof(float),
                          cudaMemcpyDeviceToHost));

    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i=0;i<y_out.size();++i){
        double diff = std::fabs((double)y_out[i] - (double)y_expect[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
    }
    mean_abs /= y_out.size();

    std::cout << "Step3 stem+block0 forward done\n";
    std::cout << "  im2col (stem conv1): " << ms_im2col << " ms\n";
    std::cout << "  gemm   (stem conv1): " << ms_gemm   << " ms\n";
    std::cout << "  bn1+relu (stem)    : " << (ms_bn+ms_relu) << " ms\n";
    std::cout << "  maxpool            : " << ms_pool   << " ms\n";
    std::cout << "  block0 (layer1[0]) : " << ms_block0 << " ms\n";
    std::cout << "Diff vs PyTorch     : max_abs=" << max_abs
              << " mean_abs=" << mean_abs << "\n";

    if (max_abs <= 1e-4) {
        std::cout << "[OK] within atol 1e-4\n";
    } else {
        std::cerr << "[FAIL] exceed atol 1e-4\n";
    }

    // ----------------------------------------------------------------
    // 9) cleanup
    cudaFree(dInput);
    cudaFree(dCol_stem);
    cudaFree(dW_stem);
    cudaFree(dStemOut);
    cudaFree(dPoolOut);

    cudaFree(dBn1_w);
    cudaFree(dBn1_b);
    cudaFree(dBn1_m);
    cudaFree(dBn1_v);

    cudaFree(dCol_blk0);
    cudaFree(dW_l10c1);
    cudaFree(dW_l10c2);

    cudaFree(dBlkTmp1);
    cudaFree(dBlkTmp2);
    cudaFree(dBlkOut);

    cudaFree(dBn_l10bn1_w);
    cudaFree(dBn_l10bn1_b);
    cudaFree(dBn_l10bn1_m);
    cudaFree(dBn_l10bn1_v);

    cudaFree(dBn_l10bn2_w);
    cudaFree(dBn_l10bn2_b);
    cudaFree(dBn_l10bn2_m);
    cudaFree(dBn_l10bn2_v);

    return 0;
}
