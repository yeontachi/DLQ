// fc_api.cu : FC(512->1000) 배치용 C API (ctypes/pybind에서 호출)
#include "utils.hpp"

// 커널 선언
extern "C" __global__
void transpose_bi_to_ib(float*, const float*, int B, int I);

extern "C" __global__
void add_bias_OB(float*, const float*, int O, int B);

extern "C" __global__
void sgemm_tiled(const float*, const float*, float*, int M, int N, int K);

// 전역(프로세스 내)으로 FC 파라미터를 상주시킴
namespace {
    int gI = 0, gO = 0;          // I=512, O=1000
    float *gW = nullptr;         // device: [O*I]
    float *gB = nullptr;         // device: [O]
    float *gXT = nullptr;        // device: scratch [I*B_max]
    int    gXT_capB = 0;
}

extern "C" void fc_cleanup()
{
    if (gW) { cudaFree(gW); gW = nullptr; }
    if (gB) { cudaFree(gB); gB = nullptr; }
    if (gXT){ cudaFree(gXT); gXT = nullptr; gXT_capB = 0; }
    gI = gO = 0;
}

extern "C" int fc_init(const float* hW, const float* hB, int O, int I)
{
    // O=1000, I=512 예상
    fc_cleanup();
    gI = I; gO = O;
    CUDA_CHECK(cudaMalloc(&gW, sizeof(float)*O*I));
    CUDA_CHECK(cudaMalloc(&gB, sizeof(float)*O));
    CUDA_CHECK(cudaMemcpy(gW, hW, sizeof(float)*O*I, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gB, hB, sizeof(float)*O,   cudaMemcpyHostToDevice));
    return 0;
}

// 내부 스크래치(I*B) 확보/확장
static void ensure_xt(int B)
{
    if (gXT_capB >= B) return;
    if (gXT) cudaFree(gXT);
    gXT_capB = B;
    CUDA_CHECK(cudaMalloc(&gXT, sizeof(float)*gI*B));
}

// dX_BI : device ptr to GAP [B×I] (row-major)
// dOut_BO: device ptr to output [B×O] (row-major) <- 원하는 최종 레이아웃
extern "C" int fc_forward_batched(const float* dX_BI, int B, float* dOut_BO)
{
    if (!gW || !gB || gI<=0 || gO<=0) return 1;

    // 1) transpose: X^T [I×B]
    ensure_xt(B);
    dim3 blk(32, 32);
    dim3 grd((gI + blk.x - 1)/blk.x, (B + blk.y - 1)/blk.y);
    transpose_bi_to_ib<<<grd, blk>>>(gXT, dX_BI, B, gI);
    CUDA_CHECK(cudaGetLastError());

    // 2) GEMM: C_OB = W_OI * X^T_IB  => C [O×B]
    //    our sgemm_tiled: C[M×N] = A[M×K] * B[K×N]
    dim3 blkG(32, 32);
    dim3 grdG((B + blkG.x - 1)/blkG.x, (gO + blkG.y - 1)/blkG.y);
    sgemm_tiled<<<grdG, blkG>>>(gW, gXT, /*C=*/(float*)dOut_BO, gO, B, gI);
    CUDA_CHECK(cudaGetLastError());

    // 3) bias add: C[O×B] += b[O] (broadcast)
    add_bias_OB<<<grd, blk>>>((float*)dOut_BO, gB, gO, B);
    CUDA_CHECK(cudaGetLastError());

    // 4) (선택) 여기서 C[O×B] → Y[B×O] transpose가 필요 없다면 그대로 사용
    //     Python에서 [B,O]로 본다면, 후단 로직만 맞춰주면 됨.

    return 0;
}
