#include "matmultKernel.h"

__global__ void matmultKernel00(float *A, float *B, float *C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++)
            sum += A[row * width + k] * B[k * width + col];
        C[row * width + col] = sum;
    }
}