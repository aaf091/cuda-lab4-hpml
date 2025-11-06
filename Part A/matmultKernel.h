#ifndef MATMULT_KERNEL_H
#define MATMULT_KERNEL_H
__global__ void matmultKernel00(const float *A, const float *B, float *C, int width);
__global__ void matmultKernel01(const float *A, const float *B, float *C, int width);
#endif
