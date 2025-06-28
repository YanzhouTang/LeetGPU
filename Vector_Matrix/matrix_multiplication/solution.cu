#include <cuda_runtime.h>

#define TILE_SIZE 64
#define THREAD_TILE_SIZE 8  // Each thread computes a 2x2 result tile
#define PREFETCH_TILE_SIZE 32
#define PREFETCH_THREAD_TILE_SIZE 4
#define SHARED_TILE_SIZE 16  // Separate tile size for shared memory version

extern "C" {

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
    __shared__ float tile_A[SHARED_TILE_SIZE][SHARED_TILE_SIZE];
    __shared__ float tile_B[SHARED_TILE_SIZE][SHARED_TILE_SIZE];
    
    // Calculate the position of result element that this thread is responsible for
    int row = blockIdx.y * SHARED_TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * SHARED_TILE_SIZE + threadIdx.x;
    
    float c_value = 0.0f;
    
    // Process all tiles
    for (int tile = 0; tile < (N + SHARED_TILE_SIZE - 1) / SHARED_TILE_SIZE; tile++) {
        // Load tile of matrix A from global memory to shared memory
        if (row < M && (tile * SHARED_TILE_SIZE + threadIdx.x) < N) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * N + tile * SHARED_TILE_SIZE + threadIdx.x];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load tile of matrix B from global memory to shared memory
        if (col < K && (tile * SHARED_TILE_SIZE + threadIdx.y) < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[(tile * SHARED_TILE_SIZE + threadIdx.y) * K + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Ensure all threads have completed loading data
        __syncthreads();
        
        // Compute contribution of current tile
        for (int k = 0; k < SHARED_TILE_SIZE; k++) {
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

// Register blocking + shared memory implementation
// Each thread computes THREAD_TILE_SIZE x THREAD_TILE_SIZE result elements
__global__ void matrix_multiplication_kernel_register_blocking(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];
    
    // Register arrays to store intermediate results (4x4 per thread)
    float c_reg[THREAD_TILE_SIZE][THREAD_TILE_SIZE] = {0.0f};
    
    // Calculate thread's starting position in result matrix
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    int thread_row = threadIdx.y * THREAD_TILE_SIZE;
    int thread_col = threadIdx.x * THREAD_TILE_SIZE;
    
    // Thread indices for loading - use 2D mapping for better coalescing
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // Loop over all tiles along the K dimension
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load A tile: each thread loads a 2x2 sub-block
        for (int i = 0; i < THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                int load_row = ty + i * TILE_SIZE / THREAD_TILE_SIZE ;
                int load_col = tx + j * TILE_SIZE / THREAD_TILE_SIZE ;
                int global_row = block_row + load_row;
                int global_col = tile * TILE_SIZE + load_col;
                
                if (global_row < M && global_col < N) {
                    tile_A[load_row][load_col] = A[global_row * N + global_col];
                } else {
                    tile_A[load_row][load_col] = 0.0f;
                }
            }
        }
        
        // Load B tile: each thread loads a 2x2 sub-block
        for (int i = 0; i < THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                int load_row = ty * THREAD_TILE_SIZE + i;
                int load_col = tx * THREAD_TILE_SIZE + j;
                int global_row = tile * TILE_SIZE + load_row;
                int global_col = block_col + load_col;
                
                if (global_row < N && global_col < K) {
                    tile_B[load_row][load_col] = B[global_row * K + global_col];
                } else {
                    tile_B[load_row][load_col] = 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        // Compute 2x2 thread tile using register blocking
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load A and B values into registers
            float a_reg[THREAD_TILE_SIZE];
            float b_reg[THREAD_TILE_SIZE];
            
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                a_reg[i] = tile_A[thread_row + i][k];
            }
            
            for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                b_reg[j] = tile_B[k][thread_col + j];
            }
            
            // Compute outer product and accumulate
            for (int i = 0; i < THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < THREAD_TILE_SIZE; j++) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results from registers to global memory
    for (int i = 0; i < THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < THREAD_TILE_SIZE; j++) {
            int global_row = block_row + thread_row + i;
            int global_col = block_col + thread_col + j;
            
            if (global_row < M && global_col < K) {
                C[global_row * K + global_col] = c_reg[i][j];
            }
        }
    }
}

__global__ void matrix_multiplication_kernel_prefetch(const float* A, const float* B, float* C, int M, int N, int K) {
    // Double buffering: use two sets of shared memory tiles
    __shared__ float tile_A[2][PREFETCH_TILE_SIZE][PREFETCH_TILE_SIZE + 1];
    __shared__ float tile_B[2][PREFETCH_TILE_SIZE][PREFETCH_TILE_SIZE + 1];
    
    // Register arrays to store intermediate results (2x2 per thread)
    float c_reg[PREFETCH_THREAD_TILE_SIZE][PREFETCH_THREAD_TILE_SIZE] = {0.0f};
    
    // Calculate thread's starting position in result matrix
    int block_row = blockIdx.y * PREFETCH_TILE_SIZE;
    int block_col = blockIdx.x * PREFETCH_TILE_SIZE;
    int thread_row = threadIdx.y * PREFETCH_THREAD_TILE_SIZE;
    int thread_col = threadIdx.x * PREFETCH_THREAD_TILE_SIZE;
    
    // Thread indices for loading
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    int num_tiles = (N + PREFETCH_TILE_SIZE - 1) / PREFETCH_TILE_SIZE;
    
    // Load first tile (tile 0) into buffer 0
    int buffer_idx = 0;
    if (num_tiles > 0) {
        // Load A tile: each thread loads a 2x2 sub-block
        for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                int load_row = ty * PREFETCH_THREAD_TILE_SIZE + i;
                int load_col = tx * PREFETCH_THREAD_TILE_SIZE + j;
                int global_row = block_row + load_row;
                int global_col = 0 * PREFETCH_TILE_SIZE + load_col;  // tile = 0
                
                if (global_row < M && global_col < N) {
                    tile_A[buffer_idx][load_row][load_col] = A[global_row * N + global_col];
                } else {
                    tile_A[buffer_idx][load_row][load_col] = 0.0f;
                }
            }
        }
        
        // Load B tile: each thread loads a 2x2 sub-block
        for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
            for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                int load_row = ty * PREFETCH_THREAD_TILE_SIZE + i;
                int load_col = tx * PREFETCH_THREAD_TILE_SIZE + j;
                int global_row = 0 * PREFETCH_TILE_SIZE + load_row;  // tile = 0
                int global_col = block_col + load_col;
                
                if (global_row < N && global_col < K) {
                    tile_B[buffer_idx][load_row][load_col] = B[global_row * K + global_col];
                } else {
                    tile_B[buffer_idx][load_row][load_col] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Main loop with prefetching
    for (int tile = 0; tile < num_tiles; tile++) {
        int compute_buffer = buffer_idx;
        int prefetch_buffer = 1 - buffer_idx;  // Toggle between 0 and 1
        
        // Prefetch next tile while computing current tile
        if (tile + 1 < num_tiles) {
            // Load next A tile: each thread loads a 2x2 sub-block
            for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                    int load_row = ty * PREFETCH_THREAD_TILE_SIZE + i;
                    int load_col = tx * PREFETCH_THREAD_TILE_SIZE + j;
                    int global_row = block_row + load_row;
                    int global_col = (tile + 1) * PREFETCH_TILE_SIZE + load_col;
                    
                    if (global_row < M && global_col < N) {
                        tile_A[prefetch_buffer][load_row][load_col] = A[global_row * N + global_col];
                    } else {
                        tile_A[prefetch_buffer][load_row][load_col] = 0.0f;
                    }
                }
            }
            
            // Load next B tile: each thread loads a 2x2 sub-block
            for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                    int load_row = ty * PREFETCH_THREAD_TILE_SIZE + i;
                    int load_col = tx * PREFETCH_THREAD_TILE_SIZE + j;
                    int global_row = (tile + 1) * PREFETCH_TILE_SIZE + load_row;
                    int global_col = block_col + load_col;
                    
                    if (global_row < N && global_col < K) {
                        tile_B[prefetch_buffer][load_row][load_col] = B[global_row * K + global_col];
                    } else {
                        tile_B[prefetch_buffer][load_row][load_col] = 0.0f;
                    }
                }
            }
        }
        
        // Compute using current tile (overlap with prefetch)
        for (int k = 0; k < PREFETCH_TILE_SIZE; k++) {
            // Load A and B values into registers from compute buffer
            float a_reg[PREFETCH_THREAD_TILE_SIZE];
            float b_reg[PREFETCH_THREAD_TILE_SIZE];
            
            for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
                a_reg[i] = tile_A[compute_buffer][thread_row + i][k];
            }
            
            for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                b_reg[j] = tile_B[compute_buffer][k][thread_col + j];
            }
            
            // Compute outer product and accumulate
            for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
                for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
                    c_reg[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }
        
        // Toggle buffer for next iteration
        buffer_idx = prefetch_buffer;
        
        __syncthreads();
    }
    
    // Write results from registers to global memory
    for (int i = 0; i < PREFETCH_THREAD_TILE_SIZE; i++) {
        for (int j = 0; j < PREFETCH_THREAD_TILE_SIZE; j++) {
            int global_row = block_row + thread_row + i;
            int global_col = block_col + thread_col + j;
            
            if (global_row < M && global_col < K) {
                C[global_row * K + global_col] = c_reg[i][j];
            }
        }
    }
}

} // extern "C"

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // For register blocking: 8x8 threads, each thread computes 2x2 elements
    dim3 threadsPerBlock(PREFETCH_TILE_SIZE / PREFETCH_THREAD_TILE_SIZE, PREFETCH_TILE_SIZE / PREFETCH_THREAD_TILE_SIZE);  // 8x8 threads per block
    dim3 blocksPerGrid((K + PREFETCH_TILE_SIZE - 1) / PREFETCH_TILE_SIZE, (M + PREFETCH_TILE_SIZE - 1) / PREFETCH_TILE_SIZE);
    
    // Use register blocking optimized version
    matrix_multiplication_kernel_prefetch<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);
    
    cudaDeviceSynchronize();
}
