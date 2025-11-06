#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "vecaddKernel.h"
#include "timer.h"

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <num_values_per_thread>\n", argv[0]);
        return 1;
    }
    // Turn the lab's "values per thread" arg into a realistic N
    const int vals_per_thread = atoi(argv[1]);
    const int threads = 256;
    const int blocks  = 1024;
    const int N = vals_per_thread * threads * blocks; // big enough to see timing

    float *A = (float*)malloc(N * sizeof(float));
    float *B = (float*)malloc(N * sizeof(float));
    float *C = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) { A[i] = i * 0.5f; B[i] = i * 0.25f; }

    float *dA, *dB, *dC;
    cudaMalloc(&dA, N * sizeof(float));
    cudaMalloc(&dB, N * sizeof(float));
    cudaMalloc(&dC, N * sizeof(float));
    cudaMemcpy(dA, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(threads);
    dim3 grid(blocks);

    start_timer();
#ifdef COALESCED
    vecaddKernel01<<<grid, block>>>(dA, dB, dC, N);
#else
    vecaddKernel00<<<grid, block>>>(dA, dB, dC, N);
#endif
    cudaDeviceSynchronize();
    float elapsed = stop_timer();

    cudaMemcpy(C, dC, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("VectorAdd %s: vals/thread=%d, N=%d, time=%.3f ms\n",
#ifdef COALESCED
           "Coalesced", 
#else
           "Uncoalesced",
#endif
           vals_per_thread, N, elapsed);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(A); free(B); free(C);
    return 0;
}
