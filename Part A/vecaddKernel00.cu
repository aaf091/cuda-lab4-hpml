#include "vecaddKernel.h"

__global__ void vecAddKernel00(float *A, float *B, float *C, int N) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += stride * 2)  // stride too big → uncoalesced
        if (i < N) C[i] = A[i] + B[i];
}