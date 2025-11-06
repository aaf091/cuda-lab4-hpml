#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "matmultKernel.h"
#include "timer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    const int N = atoi(argv[1]);
    const size_t bytes = (size_t)N * (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    float *hC = (float*)malloc(bytes);

    for (int i = 0; i < N * N; ++i) { hA[i] = 1.f; hB[i] = 1.f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

#ifndef TILE
#define TILE 32
#endif
#ifdef OPTIMIZED
    dim3 block(TILE, TILE);
#else
    dim3 block(16, 16);
#endif
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    start_timer();
#ifdef OPTIMIZED
    matmultKernel01<<<grid, block>>>(dA, dB, dC, N);
#else
    matmultKernel00<<<grid, block>>>(dA, dB, dC, N);
#endif
    cudaDeviceSynchronize();
    float ms = stop_timer();

    cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);

    printf("MatMul %s: N=%d, time=%.3f ms\n",
#ifdef OPTIMIZED
           "Optimized(TILE=32)",
#else
           "Naive(16x16)",
#endif
           N, ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}
