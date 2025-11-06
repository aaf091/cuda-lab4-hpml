#include "vecaddKernel.h"

__global__ void vecaddKernel01(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i]; // contiguous access across a warp
    }
}
