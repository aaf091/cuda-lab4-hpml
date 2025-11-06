#include "vecaddKernel.h"

// Intentionally strided to be less friendly (still correct)
__global__ void vecaddKernel00(float *A, float *B, float *C, int N) {
    int tid    = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x * 2; // bigger step hurts coalescing
    for (int i = tid; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}
