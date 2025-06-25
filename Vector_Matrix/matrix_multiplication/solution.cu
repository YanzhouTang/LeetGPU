#include "solve.h"
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Naive implementation: each thread computes one result element
__global__ void matrix_multiplication_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < K) {
        float sum = 0.0f;
        
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * K + col];
        }
        
        C[row * K + col] = sum;
    }
}

// Shared memory optimized implementation using tiling
__global__ void matrix_multiplication_kernel_sharemem(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
    
    // Calculate the position of result element that this thread is responsible for
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float c_value = 0.0f;
    
    // Process all tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile of matrix A from global memory to shared memory
        if (row < M && (tile * TILE_SIZE + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of matrix B from global memory to shared memory
        if (col < K && (tile * TILE_SIZE + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * K + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Ensure all threads have completed loading data
        __syncthreads();
        
        // Compute contribution of current tile
        for (int k = 0; k < TILE_SIZE; k++) {
            c_value += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
        }
        
        // Ensure all threads have completed computation before next round
        __syncthreads();
    }
    
    // Write result back to global memory
    if (row < M && col < K) {
        C[row * K + col] = c_value;
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((K + TILE_SIZE - 1) / TILE_SIZE,
                       (M + TILE_SIZE - 1) / TILE_SIZE);
    
    matrix_multiplication_kernel_sharemem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    cudaDeviceSynchronize();
}
