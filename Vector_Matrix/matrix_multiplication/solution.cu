#include <cuda_runtime.h>

#define TILE_SIZE 64
#define THREAD_TILE_SIZE 4  // Each thread computes a 2x2 result tile
#define PREFETCH_TILE_SIZE 32
#define PREFETCH_THREAD_TILE_SIZE 4
#define SHARED_TILE_SIZE 16  // Separate tile size for shared memory version

const int BM = 128;
const int BN = 128;
const int BK = 8;
const int M_WARP_TILE = BM/2;
const int N_WARP_TILE = BN/4;
const int TM = 8;
const int TN = 8;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

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


__global__ void gemm_kernel(
    const float * a, const float * b, float * c,
     int M,  int N,  int K) {



    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;   //relative id in this block

    const int warp_num = tid/32;
    const int warp_n = warp_num%4;
    const int warp_m = warp_num/4;
    const int mma_num = tid%32;
    const int mma_n = mma_num%4;
    const int mma_m = mma_num/4;



    //use for one block
    __shared__ float s_a[BK][BM];       //128*8
    __shared__ float s_b[BK][BN];       //8*128

    float r_c[8][8] = {0.0};          //use only for one thread
    float a_fragment[4*2];
    float b_fragment[4*2];

    //relative address
    //deal with data_per_thread data a thread, so that a block can complete part of gemm
    int load_a_smem_m = tid % 32;  
    int load_a_smem_k = tid / 32;  
    int load_b_smem_n = tid % 32;   
    int load_b_smem_k = tid/32;  

    int load_a_gmem_m = by * BM + load_a_smem_m;  // global row of a
    int load_b_gmem_n = bx * BN + load_b_smem_n;  // global col of b


    //k is devided by dk to (K+BK-1)/BK part
    for (int bk = 0; bk < (N + BK - 1) / BK; bk++) {
        //load data from global mem to share mem
        int load_a_gmem_k = bk * BK + load_a_smem_k;   // global col of a
        // int load_a_gmem_addr = OFFSET(load_a_gmem_k, load_a_gmem_m, M);         //row col row_length
        int load_a_gmem_addr = OFFSET(load_a_gmem_m, load_a_gmem_k, N);         //row col row_length
        int load_a_gmem_addr2 = OFFSET((load_a_gmem_m+32), load_a_gmem_k, N);         //row col row_length
        int load_a_gmem_addr3 = OFFSET((load_a_gmem_m+32*2), load_a_gmem_k, N);         //row col row_length
        int load_a_gmem_addr4 = OFFSET((load_a_gmem_m+32*3), load_a_gmem_k, N);         //row col row_length
        //float tmp[4];
        // FLOAT4(tmp)=FLOAT4(a[load_a_gmem_addr]);
        // #pragma unroll
        // for (int i=0;i<data_per_thread;i++) {
        //     s_a[load_a_smem_k+i][load_a_smem_m]= tmp[i];
        //     //s_a[load_a_smem_k+i][load_a_smem_m]= a[load_a_gmem_addr+i];
        // }        
        //another way
        s_a[load_a_smem_k][load_a_smem_m]= a[load_a_gmem_addr];
        s_a[load_a_smem_k][load_a_smem_m+32]= a[load_a_gmem_addr2];
        s_a[load_a_smem_k][load_a_smem_m+32*2]= a[load_a_gmem_addr3];
        s_a[load_a_smem_k][load_a_smem_m+32*3]= a[load_a_gmem_addr4];

        int load_b_gmem_k = bk * BK + load_b_smem_k;   // global row of b
        int load_b_gmem_addr = OFFSET(load_b_gmem_k, load_b_gmem_n, K);         //row col row_length
        // FLOAT4(tmp)=FLOAT4(b[load_b_gmem_addr]);
        // #pragma unroll
        // for (int i=0;i<data_per_thread;i++) {
        //     s_b[load_b_smem_k+i][load_b_smem_n] = tmp[i];
        //     //s_b[load_b_smem_k][load_b_smem_n+i] = b[load_b_gmem_addr+i];
        // }
        s_b[load_b_smem_k][load_b_smem_n] = b[load_b_gmem_addr];
        s_b[load_b_smem_k][load_b_smem_n+32] = b[load_b_gmem_addr+32];
        s_b[load_b_smem_k][load_b_smem_n+32*2] = b[load_b_gmem_addr+32*2];
        s_b[load_b_smem_k][load_b_smem_n+32*3] = b[load_b_gmem_addr+32*3];
        

        __syncthreads();

        //mma part
        //wrap tile first, N:4, M:2
        //register to restore fragment
        //thread tile
        #pragma unroll
        for(int warp_k = 0; warp_k < BK; warp_k++) {
            FLOAT4(a_fragment[0])=FLOAT4(s_a[warp_k][warp_m*(M_WARP_TILE)+mma_m*4]);
            FLOAT4(a_fragment[4])=FLOAT4(s_a[warp_k][warp_m*(M_WARP_TILE)+mma_m*4+M_WARP_TILE/2]);
            FLOAT4(b_fragment[0])=FLOAT4(s_b[warp_k][warp_n*(N_WARP_TILE)+mma_n*4]);
            FLOAT4(b_fragment[4])=FLOAT4(s_b[warp_k][warp_n*(N_WARP_TILE)+mma_n*4+N_WARP_TILE/2]);
            
            #pragma unroll
            for(int n=0; n< 4 ; n++) {
                for(int m=0; m < 4 ; m++) {
                    r_c[m][n]                   += a_fragment[m] * b_fragment[n];
                    r_c[m][n + 4]               += a_fragment[m] * b_fragment[n + 4];
                    r_c[m+4][n]                 += a_fragment[m + 4] * b_fragment[n];
                    r_c[m+4][n+4]               += a_fragment[m + 4] * b_fragment[n + 4];
                }
            }
            // #pragma unroll
            // for(int i=0;i<4;i++) {
            //     a_fragment[i]=s_a[warp_k][warp_m*(M_WARP_TILE)+mma_m*4+i];
            //     a_fragment[i+4]=s_a[warp_k][warp_m*(M_WARP_TILE)+mma_m*4+i+M_WARP_TILE/2];
            // }
            // #pragma unroll
            // for(int i=0;i<4;i++) {
            //     b_fragment[i]=s_b[warp_k][warp_n*(N_WARP_TILE)+mma_n*4+i];
            //     b_fragment[i+4]=s_b[warp_k][warp_n*(N_WARP_TILE)+mma_n*4+i+N_WARP_TILE/2];
            // }
            

        }

        __syncthreads();
    }

    //write to mem
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int store_c_gmem_m = by * BM + warp_m * M_WARP_TILE + mma_m * 4 + i;                 //global rol of output
        int store_c_gmem_n = bx * BN + warp_n * N_WARP_TILE + mma_n * 4;             //global col of output
        int store_c_gmem_addr = OFFSET(store_c_gmem_m, store_c_gmem_n, K);
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr+M_WARP_TILE/2*K]) = FLOAT4(r_c[i+4][0]);
        FLOAT4(c[store_c_gmem_addr+N_WARP_TILE/2]) = FLOAT4(r_c[i][4]);
        FLOAT4(c[store_c_gmem_addr+M_WARP_TILE/2*K+N_WARP_TILE/2]) = FLOAT4(r_c[i+4][4]);

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
