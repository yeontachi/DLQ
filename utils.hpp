#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>

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

struct Timer {
    cudaEvent_t a,b;
    Timer(){ CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~Timer(){ cudaEventDestroy(a); cudaEventDestroy(b); }
    void start(){ CUDA_CHECK(cudaEventRecord(a)); }
    float stop(){ CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
                  float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); return ms; }
};
