// cpp/fp32/runtime/libfc.cu
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

// ---- sgemm 커널은 .cu만 있으므로 시그니처만 선언 ----
extern "C" __global__
void sgemm_tiled(const float* A, const float* B, float* C,
                 int M, int N, int K); // C[MxN] = A[MxK] * B[KxN]

// ---- 에러 체크 ----
#define CUDA_CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  return -1; }}while(0)

static float* g_dWt = nullptr; // (I x O)  전치 보관
static float* g_dB  = nullptr; // (O)
static int gO = 0, gI = 0;

// host W(OxI)을 Wt(IxO)로 전치하여 올림
extern "C" int fc_init(void* hW, void* hB, int O, int I)
{
    gO = O; gI = I;

    // host transpose
    std::vector<float> Wt((size_t)I * O);
    const float* W = reinterpret_cast<const float*>(hW);
    const float* B = reinterpret_cast<const float*>(hB);
    for (int o=0; o<O; ++o){
        for (int i=0; i<I; ++i){
            Wt[(size_t)i*O + o] = W[(size_t)o*I + i];
        }
    }

    if (g_dWt) cudaFree(g_dWt);
    if (g_dB)  cudaFree(g_dB);
    CUDA_CHECK(cudaMalloc(&g_dWt, (size_t)I*O*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&g_dB,  O*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(g_dWt, Wt.data(), (size_t)I*O*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dB,  B,         (size_t)O*sizeof(float),   cudaMemcpyHostToDevice));
    return 0;
}

extern "C" void fc_cleanup()
{
    if (g_dWt){ cudaFree(g_dWt); g_dWt=nullptr; }
    if (g_dB ){ cudaFree(g_dB ); g_dB =nullptr; }
    gO = gI = 0;
}

// bias add: out[b, o] += bias[o]
__global__ void add_bias_rowwise(float* out, const float* bias, int B, int O)
{
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (o >= O || b >= B) return;
    out[(size_t)b*O + o] += bias[o];
}

// GEMM 런처
static inline void launch_sgemm(const float* A, const float* B, float* C,
                                int M, int N, int K)
{
    dim3 blk(32,32);
    dim3 grd((N+blk.x-1)/blk.x, (M+blk.y-1)/blk.y);
    sgemm_tiled<<<grd, blk>>>(A, B, C, M, N, K);
}

extern "C" int fc_forward_batched(void* dX_ptr, int B, void* dOut_ptr)
{
    // dX: (B x I), g_dWt: (I x O)  => Y = X * Wt  -> (B x O)
    if (!g_dWt || !g_dB || gO<=0 || gI<=0) return -1;
    float* dX   = reinterpret_cast<float*>(dX_ptr);
    float* dOut = reinterpret_cast<float*>(dOut_ptr);

    // GEMM
    launch_sgemm(/*A*/dX, /*B*/g_dWt, /*C*/dOut, /*M*/B, /*N*/gO, /*K*/gI);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr,"GEMM launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // bias add
    dim3 blk(32,8);
    dim3 grd((gO+blk.x-1)/blk.x, (B+blk.y-1)/blk.y);
    add_bias_rowwise<<<grd, blk>>>(dOut, g_dB, B, gO);
    err = cudaGetLastError();
    if (err != cudaSuccess){
        fprintf(stderr,"bias launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }
    return 0;
}
