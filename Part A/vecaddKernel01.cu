// vecAddKernel01.cu
// ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Zehra Sura and Robert Kingan
// Based on code from the CUDA Programming Guide
//
// This kernel adds two vectors A and B into C on the GPU
// using a COALESCED access pattern across threads.
//
// Each block processes (blockDim.x * N) elements.
// On iteration j, threads in a warp access consecutive indices:
//   idx = blockIdx.x * blockDim.x * N + j * blockDim.x + threadIdx.x
// which is coalesced across the warp.

__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
    int blockStart = blockIdx.x * blockDim.x * N;

    // Iterate over ValuesPerThread (N) with unit-stride across threads
    for (int j = 0; j < N; ++j) {
        int i = blockStart + j * blockDim.x + threadIdx.x;
        C[i] = A[i] + B[i];
    }
}