// cpp/fp32/runtime/infer_e2e.cu
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sys/stat.h>  // mkdir
#include "utils.hpp"
#include "../kernels/gap_global.cuh"

// ==== 외부 커널 선언 (이미 다른 .cu에서 구현됨) ====
extern "C" __global__
void im2col_nchw(const float* x, int N,int C,int H,int W,
                 int kH,int kW,int sH,int sW,int pH,int pW,
                 float* col);

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
void add_inplace(float* y, const float* x, int n); // (필요시)

// ---- 로컬 유틸 ----
static inline void ensure_out_dir() {
#ifdef _WIN32
    _mkdir("out");
#else
    ::mkdir("out", 0755);
#endif
}

static inline void save_bin_f32_local(const std::string& path,
                                      const std::vector<float>& v) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) { std::perror(("fopen "+path).c_str()); std::exit(1); }
    std::fwrite(v.data(), sizeof(float), v.size(), fp);
    std::fclose(fp);
}

// ==== GEMM 런처 ====
static inline void gemm_launch(const float* A, const float* B, float* C,
                               int M, int N, int K)
{
    dim3 blk(32,32);
    dim3 grd( (N+blk.x-1)/blk.x, (M+blk.y-1)/blk.y );
    sgemm_tiled<<<grd,blk>>>(A, B, C, M, N, K);
}

// ==== BN 런처 ====
static inline void bn_launch(float* y,
                             const std::vector<float>& G,
                             const std::vector<float>& B,
                             const std::vector<float>& M,
                             const std::vector<float>& V,
                             int C, int OH, int OW,
                             float eps=1e-5f)
{
    auto dG = copy_to_device(G);
    auto dB = copy_to_device(B);
    auto dM = copy_to_device(M);
    auto dV = copy_to_device(V);
    int total = C*OH*OW;
    bn_inference<<< div_up(total,256), 256 >>>(y, dG.get(), dB.get(), dM.get(), dV.get(), eps, C, OH, OW);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ==== FC: 512 -> 1000 ====
static inline void fc_forward(const float* gap,            // [512]
                              const std::vector<float>& W, // [1000,512] (row-major, OI)
                              const std::vector<float>& B, // [1000]
                              std::vector<float>& out)     // [1000]
{
    const int O = 1000, I = 512;
    auto dGap = copy_to_device(std::vector<float>(gap, gap + I));
    auto dW   = copy_to_device(W);
    auto dOut = make_device_f32(O);

    // GEMM: (O x I) * (I x 1) -> (O x 1)
    gemm_launch(dW.get(), dGap.get(), dOut.get(), O, 1, I);
    out = copy_to_host(dOut, O);

    // bias add on host
    for (int o=0;o<O;o++) out[o] += B[o];
}
// ... 기존 include 동일 ...
static void usage(){
    std::cout <<
      "Usage:\n"
      "  step8_e2e --manifest <dir> "
      "[--layer4 <l4.bin>] [--gap <gap.bin>] "
      "[--fc_weight <w.bin>] [--fc_bias <b.bin>]\n";
}

int main(int argc, char** argv)
{
    std::string mani, layer4_path, gap_path, fcw_path, fcb_path;

    for (int i=1;i<argc;i++){
        std::string a = argv[i];
        if (a=="--manifest"   && i+1<argc) mani = argv[++i];
        else if (a=="--layer4" && i+1<argc) layer4_path = argv[++i];
        else if (a=="--gap"    && i+1<argc) gap_path    = argv[++i];  // NEW
        else if (a=="--fc_weight" && i+1<argc) fcw_path = argv[++i];
        else if (a=="--fc_bias"   && i+1<argc) fcb_path = argv[++i];
    }
    if (mani.empty()) { usage(); return 1; }

    ensure_out_dir();  // utils.hpp에 이미 구현한 mkdir -p 헬퍼

    // ---- FC 가중치 선택 로드 ----
    std::vector<float> Wfc = fcw_path.empty() ? load_bin_f32(mani + "/fc.weight.bin", 1000*512)
                                              : load_bin_f32(fcw_path,                1000*512);
    std::vector<float> Bfc = fcb_path.empty() ? load_bin_f32(mani + "/fc.bias.bin",   1000)
                                              : load_bin_f32(fcb_path,                1000);

    // ---- 입력 벡터: gap(512) 우선 사용 ----
    std::vector<float> gap512;
    if (!gap_path.empty()) {
        gap512 = load_bin_f32(gap_path, 512);           // Torch에서 만든 GAP 그대로 사용
    } else {
        // fallback: layer4 -> CUDA GAP (이전과 동일)
        const int C=512, H=7, W=7;
        std::vector<float> L4 = layer4_path.empty()
          ? load_bin_f32(mani + "/fixtures_step8/layer4_out.bin", C*H*W)
          : load_bin_f32(layer4_path, C*H*W);

        auto dL4  = copy_to_device(L4);
        auto dGAP = make_device_f32(C);
        dim3 blk(256), grd(div_up(C, (int)blk.x));
        gap_global<<<grd, blk>>>(dL4.get(), C, H, W, dGAP.get());
        CUDA_CHECK(cudaDeviceSynchronize());
        gap512 = copy_to_host(dGAP, C);
    }

    // ---- FC만 수행 ----
    std::vector<float> logits;
    fc_forward(gap512.data(), Wfc, Bfc, logits);      // y[1000] = W[1000x512] @ x[512] + b

    save_bin_f32_local("out/step8_logits.bin", logits);

    // top-1
    int top=-1; float best=-1e30f;
    for (int i=0;i<1000;i++) if (logits[i] > best){ best=logits[i]; top=i; }
    std::cout<<"[E2E] top-1 class index = "<<top<<", logit="<<best<<"\n";
    std::cout<<"(주의: synset 매핑은 별도)\n";
    return 0;
}
