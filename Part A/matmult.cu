#include <stdio.h>
#include <stdlib.h>
#include "matmultKernel.h"
#include "timer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int SIZE = N * N;
    size_t bytes = SIZE * sizeof(float);

    float *A = (float*)malloc(bytes);
    float *B = (float*)malloc(bytes);
    float *C = (float*)malloc(bytes);

    for (int i = 0; i < SIZE; i++) {
        A[i] = 1.0f;
        B[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    start_timer();
#ifdef OPTIMIZED
    matmultKernel01<<<grid, block>>>(dA, dB, dC, N);
#else
    matmultKernel00<<<grid, block>>>(dA, dB, dC, N);
#endif
    cudaDeviceSynchronize();
    float elapsed = stop_timer();

    printf("MatrixMul %s: N=%d, time=%f ms\n",
#ifdef OPTIMIZED
           "Optimized",
#else
           "Naive",
#endif
           N, elapsed);

    cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
}