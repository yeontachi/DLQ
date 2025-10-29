#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <memory>
#include <utility>   // std::pair
#include <cassert>

static inline int div_up(int a, int b) { return (a + b - 1) / b; }
static inline int max3(int a, int b, int c){ return std::max(a, std::max(b,c)); }

#define CUDA_LAUNCH(kernel, grid, block, ...) \
do { \
  kernel<<<grid, block>>>(__VA_ARGS__); \
  cudaError_t __e = cudaGetLastError(); \
  if (__e != cudaSuccess){ \
    fprintf(stderr, "CUDA launch error %s @ %s:%d\n", cudaGetErrorString(__e), __FILE__, __LINE__); \
    return 3; \
  } \
} while(0)


inline void cudaCheck(cudaError_t e, const char* f, int l){
    if (e!=cudaSuccess){ std::cerr<<"CUDA "<<cudaGetErrorString(e)
        <<" @ "<<f<<":"<<l<<"\n"; std::exit(1); }
}
#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

inline std::vector<float> load_bin_f32(const std::string& path, size_t expected = 0){
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) { std::cerr<<"open fail: "<<path<<"\n"; std::exit(1); }
    ifs.seekg(0, std::ios::end); size_t bytes = ifs.tellg(); ifs.seekg(0);
    if (bytes % 4){ std::cerr<<"size not float-aligned: "<<path<<"\n"; std::exit(1); }
    size_t n = bytes/4;
    std::vector<float> v(n); ifs.read((char*)v.data(), bytes);
    if (expected && n!=expected){
        std::cerr<<"unexpected size: "<<path<<" got "<<n<<" expected "<<expected<<"\n"; std::exit(1);
    }
    return v;
}

// 커널 선언
extern "C" __global__
void im2col_nchw(const float*,int,int,int,int,int,int,int,int,int,int,float*);

extern "C" __global__
void sgemm_tiled(const float*,const float*,float*,int,int,int);

extern "C" __global__
void bn_inference(float*,const float*,const float*,const float*,const float*,float,int,int,int);

extern "C" __global__
void relu_forward(float*,int);

struct Timer {
    cudaEvent_t a,b;
    Timer(){ CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(){ CUDA_CHECK(cudaEventRecord(a)); }
    float stop(){ CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
                  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};

// utils.hpp 끝부분에 추가 (선택)
// 간단한 호환 래퍼: 실제로는 load_bin_f32를 감싸서 반환만 맞춤
struct CmdArgs { std::string manifest; };
inline CmdArgs parse_args(int argc, char** argv) {
    CmdArgs a;
    for (int i=1;i<argc;i++){
        std::string s = argv[i];
        if (s=="--manifest" && i+1<argc) a.manifest = argv[++i];
    }
    return a;
}
struct Tensor {
    std::vector<float> buf;
    float* data() { return buf.data(); }
    const float* data() const { return buf.data(); }
    size_t size() const { return buf.size(); }
};

inline Tensor load_bin(const std::string& path, size_t n_elems) {
    Tensor t; t.buf = load_bin_f32(path, n_elems); return t;
}
struct Manifest {
    std::string root;
    explicit Manifest(std::string r): root(std::move(r)){}
    // 주의: 실제 manifest.json 파싱은 안 함. 호출부에서 개수를 제공해야 함.
    Tensor load(const std::string& name, size_t n_elems) {
        return load_bin(root + "/" + name + ".bin", n_elems);
    }
};

// CUDA 에러 체크 매크로가 이미 있다면 중복 정의 금지
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s @ %s:%d\n",                       \
                    cudaGetErrorString(_err), __FILE__, __LINE__);           \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)
#endif

// 디바이스 메모리 unique_ptr 삭제자
struct DeviceDeleter {
    void operator()(float* p) const noexcept {
        if (p) cudaFree(p);
    }
};

// host vector<float> -> device float* (unique_ptr로 관리)
inline std::unique_ptr<float, DeviceDeleter>
copy_to_device(const std::vector<float>& host)
{
    float* d = nullptr;
    size_t bytes = host.size() * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d, bytes));
    if (!host.empty()) {
        CUDA_CHECK(cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice));
    }
    return std::unique_ptr<float, DeviceDeleter>(d);
}

// 두 벡터 차이의 (max_abs, mean_abs) 계산
inline std::pair<double,double>
diff_max_mean(const std::vector<float>& a, const std::vector<float>& b)
{
    assert(a.size() == b.size());
    const size_t n = a.size();
    double max_abs = 0.0;
    double mean_abs = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::abs((double)a[i] - (double)b[i]);
        if (d > max_abs) max_abs = d;
        mean_abs += d;
    }
    if (n) mean_abs /= (double)n;
    return {max_abs, mean_abs};
}