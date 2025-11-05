#include <stdio.h>
#include <stdlib.h>
#include "vecaddKernel.h"
#include "timer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <num_values_per_thread>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]) * 1024;
    float *A, *B, *C;
    A = (float*)malloc(N * sizeof(float));
    B = (float*)malloc(N * sizeof(float));
    C = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = i * 0.5f;
        B[i] = i * 0.25f;
    }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));
    cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    start_timer();
#ifdef COALESCED
    vecAddKernel01<<<grid, block>>>(dA, dB, dC, N);
#else
    vecAddKernel00<<<grid, block>>>(dA, dB, dC, N);
#endif
    cudaDeviceSynchronize();
    float elapsed = stop_timer();

    cudaMemcpy(C, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("VectorAdd %s: N=%d, time=%f ms\n",
#ifdef COALESCED
           "Coalesced", 
#else
           "Uncoalesced",
#endif
           N, elapsed);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}