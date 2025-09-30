// This is for the Test CUDA
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add1(int*x){
    x[0]+=1;
}

int main() {
    int h = 41, *d = nullptr;
    cudaError_t err = cudaMalloc(&d, sizeof(int));
    if (err != cudaSuccess) { printf("cudaMalloc error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(d, &h, sizeof(int), cudaMemcpyHostToDevice);
    add1<<<1,10>>>(d);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("kernel error: %s\n", cudaGetErrorString(err)); return 1; }

    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d);

    printf("answer = %d\n", h);  // 42 기대
    return 0;
}