#include <cstdio>
#include <vector>
#include "utils.hpp"
#include "../kernels/kernels.cuh"   // im2col, sgemm, bn, relu 선언 모음
// add 커널 선언
extern "C" __global__ void add_inplace(float*, const float*, int);

struct Cfg {
  int N=1, C=64, H=56, W=56;   // stem+maxpool 이후
  int KH=3, KW=3, SH=1, SW=1, PH=1, PW=1;
  int OC=64;                   // layer1 출력 채널
  int OH=56, OW=56;
  int KCOL() const { return C*KH*KW; }
  int NCOL() const { return OH*OW; }
  int OUT_SIZE() const { return OC*OH*OW; }
  int IN_SIZE() const  { return C*H*W; }
};

int main(int argc, char** argv){
  CmdArgs args = parse_args(argc, argv); // --manifest, --input, --expect(최종)
  Manifest mani(args.manifest);

  // 1) 입력 로드: stem_after_maxpool.bin (1×64×56×56)
  std::vector<float> hX = load_bin(args.manifest+"/fixtures_step3/stem_after_maxpool.bin");
  // 준비: weights/bn for layer1.0 & layer1.1
  // 이름 예: layer1.0.conv1.weight, layer1.0.bn1.{weight,bias,running_mean,running_var} ...
  Tensor W10 = mani.load("layer1.0.conv1.weight"); // (64,64,3,3)
  Tensor G10 = mani.load("layer1.0.bn1.weight");
  Tensor B10 = mani.load("layer1.0.bn1.bias");
  Tensor M10 = mani.load("layer1.0.bn1.running_mean");
  Tensor V10 = mani.load("layer1.0.bn1.running_var");
  Tensor W11 = mani.load("layer1.0.conv2.weight");
  Tensor G11 = mani.load("layer1.0.bn2.weight");
  Tensor B11 = mani.load("layer1.0.bn2.bias");
  Tensor M11 = mani.load("layer1.0.bn2.running_mean");
  Tensor V11 = mani.load("layer1.0.bn2.running_var");

  Tensor W20 = mani.load("layer1.1.conv1.weight");
  Tensor G20 = mani.load("layer1.1.bn1.weight");
  Tensor B20 = mani.load("layer1.1.bn1.bias");
  Tensor M20 = mani.load("layer1.1.bn1.running_mean");
  Tensor V20 = mani.load("layer1.1.bn1.running_var");
  Tensor W21 = mani.load("layer1.1.conv2.weight");
  Tensor G21 = mani.load("layer1.1.bn2.weight");
  Tensor B21 = mani.load("layer1.1.bn2.bias");
  Tensor M21 = mani.load("layer1.1.bn2.running_mean");
  Tensor V21 = mani.load("layer1.1.bn2.running_var");

  Cfg cfg;
  // device buffers
  float *dX, *dCol, *dY, *dTmp;
  CUDA_CHECK(cudaMalloc(&dX,   cfg.IN_SIZE()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dCol, cfg.KCOL()*cfg.NCOL()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dY,   cfg.OUT_SIZE()*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&dTmp, cfg.OUT_SIZE()*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(dX, hX.data(), cfg.IN_SIZE()*sizeof(float), cudaMemcpyHostToDevice));

  // helper
  dim3 blk(32,32);
  dim3 grd((cfg.OW+blk.x-1)/blk.x, (cfg.OH+blk.y-1)/blk.y);

  GpuTimer tIm2Col, tGemm, tBn, tRelu, tAdd;
  auto run_conv_bn_relu = [&](const Tensor& W, const Tensor& G,const Tensor& B,const Tensor& M,const Tensor& V){
    // im2col
    tIm2Col.tic();
    im2col_nchw<<<grd, blk>>>(dX, cfg.N,cfg.C,cfg.H,cfg.W, cfg.KH,cfg.KW,cfg.SH,cfg.SW,cfg.PH,cfg.PW, dCol);
    CUDA_CHECK(cudaDeviceSynchronize());
    tIm2Col.toc();

    // W_col (OC × KCOL) 는 파일 저장이 OIHW 이므로 host에서 OIHW→(OC,KCOL)로 전개해
    // utils에 준비된 get_weight_col(W) 사용 가정 (또는 로더가 W_col로 저장)
    const float* dWcol = W.d_col(); // 가정: 로더가 준비 (없다면 memcpy로 준비)

    // gemm: (OC×KCOL) × (KCOL×NCOL) = (OC×NCOL)
    dim3 blk2(32,32), grd2((cfg.NCOL()+31)/32, (cfg.OC+31)/32);
    tGemm.tic();
    sgemm_tiled<<<grd2, blk2>>>(dWcol, dCol, dY, cfg.OC, cfg.NCOL(), cfg.KCOL());
    CUDA_CHECK(cudaDeviceSynchronize());
    tGemm.toc();

    // bn
    tBn.tic();
    int nthreads = cfg.OUT_SIZE();
    bn_inference<<< (nthreads+255)/256, 256 >>>(dY, G.d(), B.d(), M.d(), V.d(), 1e-5f, cfg.OC, cfg.OH, cfg.OW);
    CUDA_CHECK(cudaDeviceSynchronize());
    tBn.toc();

    // relu
    tRelu.tic();
    relu_forward<<< (nthreads+255)/256, 256 >>>(dY, nthreads);
    CUDA_CHECK(cudaDeviceSynchronize());
    tRelu.toc();
    // dY 가 결과, 필요 시 dX로 교체
    CUDA_CHECK(cudaMemcpy(dX, dY, cfg.OUT_SIZE()*sizeof(float), cudaMemcpyDeviceToDevice));
  };

  // === layer1.0 ===
  run_conv_bn_relu(W10,G10,B10,M10,V10);                // conv1→bn1→relu
  // conv2→bn2 (relu 없음)
  // im2col
  tIm2Col.tic();
  im2col_nchw<<<grd, blk>>>(dX, cfg.N,cfg.C,cfg.H,cfg.W, cfg.KH,cfg.KW,cfg.SH,cfg.SW,cfg.PH,cfg.PW, dCol);
  CUDA_CHECK(cudaDeviceSynchronize());
  tIm2Col.toc();
  // gemm
  {
    dim3 blk2(32,32), grd2((cfg.NCOL()+31)/32, (cfg.OC+31)/32);
    tGemm.tic();
    sgemm_tiled<<<grd2, blk2>>>(W11.d_col(), dCol, dY, cfg.OC, cfg.NCOL(), cfg.KCOL());
    CUDA_CHECK(cudaDeviceSynchronize());
    tGemm.toc();
  }
  // bn
  tBn.tic();
  bn_inference<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, G11.d(), B11.d(), M11.d(), V11.d(), 1e-5f, cfg.OC, cfg.OH, cfg.OW);
  CUDA_CHECK(cudaDeviceSynchronize());
  tBn.toc();
  // skip add + relu (입력은 layer1 시작 전 x; 현재 dTmp에 백업 가정)
  CUDA_CHECK(cudaMemcpy(dTmp, hX.data(), cfg.OUT_SIZE()*sizeof(float), cudaMemcpyHostToDevice)); // stem의 동일 shape 입력을 dTmp에 (최적화X)
  tAdd.tic();
  add_inplace<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, dTmp, cfg.OUT_SIZE());
  CUDA_CHECK(cudaDeviceSynchronize());
  tAdd.toc();
  tRelu.tic();
  relu_forward<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, cfg.OUT_SIZE());
  CUDA_CHECK(cudaDeviceSynchronize());
  tRelu.toc();
  CUDA_CHECK(cudaMemcpy(dX, dY, cfg.OUT_SIZE()*sizeof(float), cudaMemcpyDeviceToDevice)); // block0 output → dX

  // 정답과 비교(layer1_block0_out.bin)
  compare_with_expect(args.manifest+"/fixtures_step3/layer1_block0_out.bin", dX, cfg.OUT_SIZE());

  // === layer1.1 === (구조 동일: conv1→bn1→relu → conv2→bn2 → add+relu)
  // block1 입력 백업
  CUDA_CHECK(cudaMemcpy(dTmp, dX, cfg.OUT_SIZE()*sizeof(float), cudaMemcpyDeviceToDevice));
  run_conv_bn_relu(W20,G20,B20,M20,V20);
  // conv2→bn2
  tIm2Col.tic();
  im2col_nchw<<<grd, blk>>>(dX, cfg.N,cfg.C,cfg.H,cfg.W, cfg.KH,cfg.KW,cfg.SH,cfg.SW,cfg.PH,cfg.PW, dCol);
  CUDA_CHECK(cudaDeviceSynchronize()); tIm2Col.toc();
  {
    dim3 blk2(32,32), grd2((cfg.NCOL()+31)/32, (cfg.OC+31)/32);
    tGemm.tic();
    sgemm_tiled<<<grd2, blk2>>>(W21.d_col(), dCol, dY, cfg.OC, cfg.NCOL(), cfg.KCOL());
    CUDA_CHECK(cudaDeviceSynchronize()); tGemm.toc();
  }
  tBn.tic();
  bn_inference<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, G21.d(), B21.d(), M21.d(), V21.d(), 1e-5f, cfg.OC, cfg.OH, cfg.OW);
  CUDA_CHECK(cudaDeviceSynchronize()); tBn.toc();
  tAdd.tic();
  add_inplace<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, dTmp, cfg.OUT_SIZE());
  CUDA_CHECK(cudaDeviceSynchronize()); tAdd.toc();
  tRelu.tic();
  relu_forward<<< (cfg.OUT_SIZE()+255)/256, 256 >>>(dY, cfg.OUT_SIZE());
  CUDA_CHECK(cudaDeviceSynchronize()); tRelu.toc();

  // 최종 비교(layer1_block1_out.bin)
  compare_with_expect(args.manifest+"/fixtures_step3/layer1_block1_out.bin", dY, cfg.OUT_SIZE());

  // 요약 타이밍
  printf("Step3 layer1 done\n  im2col: %.3f ms  gemm: %.3f ms  bn: %.3f ms  add: %.3f ms  relu: %.3f ms\n",
    tIm2Col.ms(), tGemm.ms(), tBn.ms(), tAdd.ms(), tRelu.ms());

  cudaFree(dX); cudaFree(dCol); cudaFree(dY); cudaFree(dTmp);
  return 0;
}
