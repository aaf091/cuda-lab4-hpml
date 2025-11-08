// matmultKernel01.cu
// For ECE-GY 9143 - High Performance Computing for Machine Learning
// Instructor: Zehra Sura and Robert Kingan
// Based on code from the CUDA Programming Guide
//
// Multiplies two matrices using CUDA: A x B = C
// Block computes a FOOTPRINT_SIZE x FOOTPRINT_SIZE tile of C,
// with BLOCK_SIZE x BLOCK_SIZE threads. When FOOTPRINT_SIZE > BLOCK_SIZE,
// each thread computes (RATIO x RATIO) outputs, where
//   RATIO = FOOTPRINT_SIZE / BLOCK_SIZE.
//
// Build with: -DFOOTPRINT_SIZE=32  (BLOCK_SIZE remains 16 by default)

#include "matmultKernel.h"

#ifndef FOOTPRINT_SIZE
#define FOOTPRINT_SIZE BLOCK_SIZE
#endif

// Ensure integer ratio at compile time (assumes power-of-two sizes in the lab)
#define RATIO (FOOTPRINT_SIZE / BLOCK_SIZE)

__global__ void MatMulKernel(const Matrix A, const Matrix B, Matrix C)
{
  // Tile index in the output space
  const int tile_row = blockIdx.y;
  const int tile_col = blockIdx.x;

  // Thread coordinates inside the BLOCK_SIZE x BLOCK_SIZE thread block
  const int trow = threadIdx.y;
  const int tcol = threadIdx.x;

  // Accumulators for this thread's (RATIO x RATIO) output sub-tile
  float cval[RATIO][RATIO];
#pragma unroll
  for (int rr = 0; rr < RATIO; ++rr)
#pragma unroll
    for (int cc = 0; cc < RATIO; ++cc)
      cval[rr][cc] = 0.0f;

  // Shared tiles:
  //  A_tile: FOOTPRINT_SIZE x BLOCK_SIZE
  //  B_tile: BLOCK_SIZE x FOOTPRINT_SIZE
  __shared__ float As[FOOTPRINT_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][FOOTPRINT_SIZE];

  // Loop over K dimension in steps of BLOCK_SIZE
  for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

    // Cooperative load into shared memory.
    // Each thread loads RATIO rows from A and RATIO cols from B.
#pragma unroll
    for (int rr = 0; rr < RATIO; ++rr) {
      int a_row = tile_row * FOOTPRINT_SIZE + (trow * RATIO + rr);
      int a_col = m * BLOCK_SIZE + tcol;
      As[trow * RATIO + rr][tcol] = A.elements[a_row * A.stride + a_col];
    }

#pragma unroll
    for (int cc = 0; cc < RATIO; ++cc) {
      int b_row = m * BLOCK_SIZE + trow;
      int b_col = tile_col * FOOTPRINT_SIZE + (tcol * RATIO + cc);
      Bs[trow][tcol * RATIO + cc] = B.elements[b_row * B.stride + b_col];
    }

    __syncthreads();

    // Multiply the tiles: (FOOTPRINT_SIZE x BLOCK_SIZE) * (BLOCK_SIZE x FOOTPRINT_SIZE)
#pragma unroll
    for (int e = 0; e < BLOCK_SIZE; ++e) {
      // Fetch needed B values once per e for this thread
      float b_reg[RATIO];
#pragma unroll
      for (int cc = 0; cc < RATIO; ++cc)
        b_reg[cc] = Bs[e][tcol * RATIO + cc];

#pragma unroll
      for (int rr = 0; rr < RATIO; ++rr) {
        float a_reg = As[trow * RATIO + rr][e];
#pragma unroll
        for (int cc = 0; cc < RATIO; ++cc) {
          cval[rr][cc] += a_reg * b_reg[cc];
        }
      }
    }

    __syncthreads();
  }

  // Write results back to global memory
#pragma unroll
  for (int rr = 0; rr < RATIO; ++rr) {
    int row = tile_row * FOOTPRINT_SIZE + (trow * RATIO + rr);
#pragma unroll
    for (int cc = 0; cc < RATIO; ++cc) {
      int col = tile_col * FOOTPRINT_SIZE + (tcol * RATIO + cc);
      C.elements[row * C.stride + col] = cval[rr][cc];
    }
  }
}