// cpp/fp32/runtime/infer_e2e.cu
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "utils.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// ===== 외부 커널 선언 (정의는 kernels/*.cu 에 존재) =====
extern "C" __global__
void gap_global(const float* x, int C, int H, int W, float* out); // out[C]

extern "C" __global__
void sgemm_tiled(const float* A, const float* B, float* C,
                 int M, int N, int K); // C[MxN] = A[MxK] * B[KxN]

extern "C" __global__
void add_inplace(float* y, const float* x, int n); // y[i] += x[i]

// (필요시) ReLU/BN 커널도 extern 선언 가능
// extern "C" __global__ void relu_forward(float* x, int n);
// extern "C" __global__ void bn_inference(...);

// ===== 작은 런처 유틸 =====
static inline void launch_gemm(const float* A, const float* B, float* C,
                               int M, int N, int K)
{
    dim3 blk(32, 32);
    dim3 grd(div_up(N, (int)blk.x), div_up(M, (int)blk.y));
    sgemm_tiled<<<grd, blk>>>(A, B, C, M, N, K);
}

static inline void usage() {
    std::cout << "Usage: --manifest <exports/resnet18/fp32>\n";
}

int main(int argc, char** argv) {
    // ---- 인자 파싱 ----
    CmdArgs args = parse_args(argc, argv);
    if (args.manifest.empty()) { usage(); return 1; }

    // ============================================================
    // 1) 입력: Layer4 마지막 출력(512x7x7)을 fixture에서 로드
    //    (tools/make_step8_fixture.py 가 생성)
    // ============================================================
    const int C = 512, H = 7, W = 7;
    const size_t L4_ELEMS = (size_t)C * H * W;

    const std::string f_layer4 = args.manifest + "/fixtures_step8/layer4_out.bin";
    auto hL4 = load_bin_f32(f_layer4, L4_ELEMS);     // host [512*7*7]
    auto dL4 = copy_to_device(hL4);                  // device

    // ============================================================
    // 2) Global Average Pooling: (C,H,W) -> (C)
    // ============================================================
    auto dGAP = make_device_f32(C);                  // device [512]
    {
        dim3 blk(256);
        dim3 grd(div_up(C, (int)blk.x));
        gap_global<<<grd, blk>>>(dL4.get(), C, H, W, dGAP.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ============================================================
    // 3) FC (512 -> 1000): logits = W(1000x512) * gap(512) + b(1000)
    // ============================================================
    const int I = 512;    // in-dim
    const int O = 1000;   // out-dim

    // 가중치/바이어스 로드 (host)
    auto Wfc_h = load_bin_f32(args.manifest + "/fc.weight.bin", (size_t)O * I); // row-major [O,I]
    auto Bfc_h = load_bin_f32(args.manifest + "/fc.bias.bin",   (size_t)O);

    // 디바이스 메모리 준비
    auto dWfc    = copy_to_device(Wfc_h);  // [O*I]
    auto dBfc    = copy_to_device(Bfc_h);  // [O]
    auto dLogits = make_device_f32(O);     // [O]

    // GEMM: (O x I) * (I x 1) = (O x 1)
    // dGAP는 [I] 벡터이므로 (I x 1)로 해석하여 N=1로 곱함
    {
        dim3 blk(32, 32);
        dim3 grd(div_up(1, (int)blk.x), div_up(O, (int)blk.y));
        sgemm_tiled<<<grd, blk>>>(dWfc.get(), dGAP.get(), dLogits.get(), O, 1, I);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // bias add: logits += b  (디바이스에서 in-place)
    {
        dim3 blk(256);
        dim3 grd(div_up(O, (int)blk.x));
        add_inplace<<<grd, blk>>>(dLogits.get(), dBfc.get(), O);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ============================================================
    // 4) Host로 복사/저장 + Top-1 출력 + (선택) 기준과 diff
    // ============================================================
    auto Hlogits = copy_to_host(dLogits, O);

    // 결과 저장
    fs::create_directories("out"); 
    save_bin_f32("out/step8_logits.bin", Hlogits);

    // Top-1
    int top1 = int(std::max_element(Hlogits.begin(), Hlogits.end()) - Hlogits.begin());
    std::cout << "[E2E] top-1 class index = " << top1
              << ", logit=" << Hlogits[top1] << "\n";

    // (선택) Torch 기준과 비교
    const std::string f_ref = args.manifest + "/fixtures_step8/logits.bin";
    std::ifstream chk(f_ref, std::ios::binary);
    if (chk.good()) {
        chk.close();
        auto Href = load_bin_f32(f_ref, (size_t)O);
        auto [mx, mn] = diff_max_mean(Hlogits, Href);
        std::cout << "[E2E] diff vs torch  : max_abs=" << mx
                  << " mean_abs=" << mn << "\n";
    }

    return 0;
}
