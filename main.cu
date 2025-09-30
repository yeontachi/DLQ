#include <iostream>
#include <cuda_runtime.h>

// CUDA 커널: 100개의 스레드가 각자 1을 더하고 블록 내에서 합산
__global__ void sumKernel(int *result){
    __shared__ int shared[100]; // 공유 메모리 (스레드 100개용)

    int tid = threadIdx.x;
    shared[tid] = 1;  // 각 스레드가 1을 넣음

    __syncthreads();  // 모든 스레드가 shared[]에 값을 넣을 때까지 대기

    // 병렬 reduction (반으로 나누어 합치기)
    for
}

int main(void){
    const int NUM_THREADS=100;
    const int NUM_BLOCKS=1;

    int h_result = 0; // CPU 메모리
    int *d_result;    // GPU 메모리

    // GPU 메모리 할당 및 초기화
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);

    // 커널 실행: 1개의 블록, 100개의 스레드
    sumKernel<<<NUM_BLOCKS, NUM_THREADS>>>(d_result);

    // 결과를  CPU로 복사
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    // GPU 메모리 해제
    cudaFree(d_result);

    std::cout<<"누적합 결과 = "<<h_result<< std::endl;
    return 0;
}