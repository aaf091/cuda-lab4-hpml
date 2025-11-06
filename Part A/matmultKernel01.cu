#include "matmultKernel.h"

#ifndef TILE
#define TILE 32
#endif

__global__ void matmultKernel01(const float *A, const float *B, float *C, int width) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float sum = 0.f;
    // Assuming width % TILE == 0 per assignment (FOOTPRINT-based sizing)
    for (int t = 0; t < width / TILE; ++t) {
        As[threadIdx.y][threadIdx.x] = A[row * width + (t * TILE + threadIdx.x)];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * width + col];
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < width && col < width) {
        C[row * width + col] = sum;
    }
}
