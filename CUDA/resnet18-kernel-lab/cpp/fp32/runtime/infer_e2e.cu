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

static void usage(){
  std::cout <<
    "step8_e2e --manifest <dir> "
    "[--gap <gap.bin> | --gap_list <list.txt>] "
    "[--fc_weight <w.bin>] [--fc_bias <b.bin>] "
    "[--save_logits]\n";
}

int main(int argc, char** argv){
  std::string mani, gap_path, gap_list, fcw_path, fcb_path;
  bool save_logits=false;

  for (int i=1;i<argc;i++){
    std::string a=argv[i];
    if(a=="--manifest" && i+1<argc) mani=argv[++i];
    else if(a=="--gap" && i+1<argc) gap_path=argv[++i];
    else if(a=="--gap_list" && i+1<argc) gap_list=argv[++i];          // NEW
    else if(a=="--fc_weight" && i+1<argc) fcw_path=argv[++i];
    else if(a=="--fc_bias"   && i+1<argc) fcb_path=argv[++i];
    else if(a=="--save_logits") save_logits=true;
  }
  if(mani.empty() || (gap_path.empty() && gap_list.empty())){ usage(); return 1; }
  ensure_out_dir();

  // FC weight/bias 로드
  auto Wfc = fcw_path.empty()? load_bin_f32(mani + "/fc.weight.bin", 1000*512)
                             : load_bin_f32(fcw_path,                1000*512);
  auto Bfc = fcb_path.empty()? load_bin_f32(mani + "/fc.bias.bin",   1000)
                             : load_bin_f32(fcb_path,                1000);

  // 디바이스 메모리 재사용(FC용)
  auto dW   = copy_to_device(Wfc);
  auto dGap = make_device_f32(512);
  auto dOut = make_device_f32(1000);

  // 타이머(순수 CUDA 커널 시간만)
  cudaEvent_t evA, evB; CUDA_CHECK(cudaEventCreate(&evA)); CUDA_CHECK(cudaEventCreate(&evB));
  float total_ms=0.0f; int cnt=0;

  auto run_one = [&](const std::string& gpath){
    auto gap = load_bin_f32(gpath, 512);
    CUDA_CHECK(cudaMemcpy(dGap.get(), gap.data(), 512*sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(evA));
    // y[1000x1] = W[1000x512] @ x[512x1]
    {
      dim3 blk(32,32);
      dim3 grd( (1+blk.x-1)/blk.x, (1000+blk.y-1)/blk.y );
      sgemm_tiled<<<grd,blk>>>(dW.get(), dGap.get(), dOut.get(), 1000, 1, 512);
    }
    CUDA_CHECK(cudaEventRecord(evB));
    CUDA_CHECK(cudaEventSynchronize(evB));
    float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms, evA, evB));
    total_ms += ms; cnt++;

    auto host = copy_to_host(dOut, 1000);
    for(int i=0;i<1000;i++) host[i] += Bfc[i];

    if(save_logits){
      char fn[256]; std::snprintf(fn, sizeof(fn), "out/step8_logits_%06d.bin", cnt);
      save_bin_f32_local(fn, host);
    }
    int top=-1; float best=-1e30f;
    for(int i=0;i<1000;i++) if(host[i]>best){ best=host[i]; top=i; }
    // (원하면 per-sample 로그 출력)
    // std::cout<<"top1="<<top<<" logit="<<best<<"\n";
  };

  if(!gap_list.empty()){
    std::ifstream ifs(gap_list);
    if(!ifs){ std::cerr<<"open fail: "<<gap_list<<"\n"; return 2; }
    std::string line;
    while(std::getline(ifs, line)){
      if(line.empty()) continue;
      run_one(line);
    }
  } else {
    run_one(gap_path);
  }

  if(cnt>0) std::cout<<"[CUDA] avg FC time: "<<(total_ms/cnt)<<" ms over "<<cnt<<" samples\n";

  CUDA_CHECK(cudaEventDestroy(evA)); CUDA_CHECK(cudaEventDestroy(evB));
  return 0;
}
