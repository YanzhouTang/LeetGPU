#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>

// Include kernel definitions
#define TILE_SIZE 32
#define THREAD_TILE_SIZE 4
#define SHARED_TILE_SIZE 16

// Forward declarations of kernels from solution.cu
extern "C" {
    __global__ void matrix_multiplication_kernel_naive(const float* A, const float* B, float* C, int M, int N, int K);
    __global__ void matrix_multiplication_kernel_sharemem(const float* A, const float* B, float* C, int M, int N, int K);
    __global__ void matrix_multiplication_kernel_register_blocking(const float* A, const float* B, float* C, int M, int N, int K);
    __global__ void matrix_multiplication_kernel_prefetch(const float* A, const float* B, float* C, int M, int N, int K);
}

// Utility functions
void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

void checkCublasError(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS Error: " << message << " - Status: " << status << std::endl;
        exit(1);
    }
}

// Initialize matrix with random values
void initializeMatrix(std::vector<float>& matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

// Verify correctness by comparing with cuBLAS
bool verifyResult(const std::vector<float>& result, const std::vector<float>& reference, 
                  int M, int K, float tolerance = 1e-3f) {
    int errors = 0;
    float max_error = 0.0f;
    
    for (int i = 0; i < M * K; i++) {
        float error = std::abs(result[i] - reference[i]);
        max_error = std::max(max_error, error);
        
        if (error > tolerance) {
            errors++;
            if (errors <= 10) {  // Print first 10 errors
                std::cout << "Error at index " << i << ": got " << result[i] 
                         << ", expected " << reference[i] << ", diff " << error << std::endl;
            }
        }
    }
    
    std::cout << "Max error: " << max_error << ", Total errors: " << errors 
              << " out of " << M * K << " elements" << std::endl;
    
    return errors == 0;
}

// cuBLAS reference implementation
void runCuBLAS(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
    
    const float alpha = 1.0f, beta = 0.0f;
    
    // cuBLAS uses column-major, our matrices are row-major
    // C = A * B becomes C^T = B^T * A^T in column-major
    checkCublasError(
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
                   &alpha, d_B, K, d_A, N, &beta, d_C, K),
        "cuBLAS SGEMM failed"
    );
    
    checkCublasError(cublasDestroy(handle), "Failed to destroy cuBLAS handle");
}

// Benchmark kernel function
struct BenchmarkResult {
    std::string name;
    float time_ms;
    float gflops;
    bool correct;
};

// cuBLAS benchmark function
BenchmarkResult benchmarkCuBLAS(const float* d_A, const float* d_B, float* d_C,
                                int M, int N, int K, int warmup_runs = 10, int benchmark_runs = 20) {
    cublasHandle_t handle;
    checkCublasError(cublasCreate(&handle), "Failed to create cuBLAS handle");
    
    const float alpha = 1.0f, beta = 0.0f;
    
    std::cout << "Testing cuBLAS SGEMM" << std::endl;
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; i++) {
        checkCublasError(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
                       &alpha, d_B, K, d_A, N, &beta, d_C, K),
            "cuBLAS SGEMM warmup failed"
        );
    }
    checkCudaError(cudaDeviceSynchronize(), "cuBLAS warmup synchronization failed");
    
    // Benchmark runs
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    for (int i = 0; i < benchmark_runs; i++) {
        checkCublasError(
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N,
                       &alpha, d_B, K, d_A, N, &beta, d_C, K),
            "cuBLAS SGEMM benchmark failed"
        );
    }
    
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    float total_time_ms;
    checkCudaError(cudaEventElapsedTime(&total_time_ms, start, stop), "Failed to get elapsed time");
    
    float avg_time_ms = total_time_ms / benchmark_runs;
    
    // Calculate GFLOPS (2*M*N*K operations)
    double flops = 2.0 * M * N * K;
    float gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    
    checkCudaError(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Failed to destroy stop event");
    checkCublasError(cublasDestroy(handle), "Failed to destroy cuBLAS handle");
    
    return {"cuBLAS", avg_time_ms, gflops, true};  // cuBLAS is always "correct" as reference
}

BenchmarkResult benchmarkKernel(const std::string& name,
                               void (*kernel)(const float*, const float*, float*, int, int, int),
                               const float* d_A, const float* d_B, float* d_C,
                               const std::vector<float>& reference,
                               int M, int N, int K, int warmup_runs = 10, int benchmark_runs = 20) {
    
    // Determine grid and block dimensions
    dim3 threadsPerBlock, blocksPerGrid;
    
    if (name.find("naive") != std::string::npos) {
        threadsPerBlock = dim3(16, 16);
        blocksPerGrid = dim3((K + 15) / 16, (M + 15) / 16);
    } else if (name.find("shared_memory") != std::string::npos) {
        threadsPerBlock = dim3(SHARED_TILE_SIZE, SHARED_TILE_SIZE);
        blocksPerGrid = dim3((K + SHARED_TILE_SIZE - 1) / SHARED_TILE_SIZE, (M + SHARED_TILE_SIZE - 1) / SHARED_TILE_SIZE);
    } else {  // register_blocking or prefetch
        threadsPerBlock = dim3(TILE_SIZE / THREAD_TILE_SIZE, TILE_SIZE / THREAD_TILE_SIZE);
        blocksPerGrid = dim3((K + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    }
    
    std::cout << "Testing " << name << " with grid(" << blocksPerGrid.x << "," << blocksPerGrid.y 
              << ") block(" << threadsPerBlock.x << "," << threadsPerBlock.y << ")" << std::endl;
    
    // Warmup runs
    for (int i = 0; i < warmup_runs; i++) {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    checkCudaError(cudaDeviceSynchronize(), "Warmup synchronization failed");
    
    // Benchmark runs
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Failed to create start event");
    checkCudaError(cudaEventCreate(&stop), "Failed to create stop event");
    
    checkCudaError(cudaEventRecord(start), "Failed to record start event");
    
    for (int i = 0; i < benchmark_runs; i++) {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    }
    
    checkCudaError(cudaEventRecord(stop), "Failed to record stop event");
    checkCudaError(cudaEventSynchronize(stop), "Failed to synchronize stop event");
    
    float total_time_ms;
    checkCudaError(cudaEventElapsedTime(&total_time_ms, start, stop), "Failed to get elapsed time");
    
    float avg_time_ms = total_time_ms / benchmark_runs;
    
    // Calculate GFLOPS (2*M*N*K operations)
    double flops = 2.0 * M * N * K;
    float gflops = (flops / (avg_time_ms * 1e-3)) / 1e9;
    
    // Verify correctness
    std::vector<float> result(M * K);
    checkCudaError(cudaMemcpy(result.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy result from device");
    
    bool correct = verifyResult(result, reference, M, K);
    
    checkCudaError(cudaEventDestroy(start), "Failed to destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Failed to destroy stop event");
    
    return {name, avg_time_ms, gflops, correct};
}

int main(int argc, char** argv) {
    // Default matrix dimensions
    int M = 8192, N = 6144, K = 4096;
    
    // Parse command line arguments
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }
    
    std::cout << "Matrix dimensions: A(" << M << "x" << N << ") * B(" << N << "x" << K 
              << ") = C(" << M << "x" << K << ")" << std::endl;
    
    // Initialize host matrices
    std::vector<float> h_A(M * N), h_B(N * K), h_C(M * K);
    std::vector<float> reference(M * K);
    
    initializeMatrix(h_A, M * N);
    initializeMatrix(h_B, N * K);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, M * N * sizeof(float)), "Failed to allocate d_A");
    checkCudaError(cudaMalloc(&d_B, N * K * sizeof(float)), "Failed to allocate d_B");
    checkCudaError(cudaMalloc(&d_C, M * K * sizeof(float)), "Failed to allocate d_C");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy A to device");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice),
                   "Failed to copy B to device");
    
    // Get cuBLAS reference result
    std::cout << "Computing cuBLAS reference..." << std::endl;
    runCuBLAS(d_A, d_B, d_C, M, N, K);
    checkCudaError(cudaMemcpy(reference.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost),
                   "Failed to copy reference result");
    
    // Clear result matrix for benchmarking
    checkCudaError(cudaMemset(d_C, 0, M * K * sizeof(float)), "Failed to clear result matrix");
    
    // Benchmark all kernels
    std::vector<BenchmarkResult> results;
    
    std::cout << "\n=== Starting Benchmarks ===" << std::endl;
    
    // First benchmark cuBLAS as reference
    results.push_back(benchmarkCuBLAS(d_A, d_B, d_C, M, N, K));
    
    // Test each kernel
    results.push_back(benchmarkKernel("naive", matrix_multiplication_kernel_naive, 
                                     d_A, d_B, d_C, reference, M, N, K));
    
    results.push_back(benchmarkKernel("shared_memory", matrix_multiplication_kernel_sharemem,
                                     d_A, d_B, d_C, reference, M, N, K));
    
    results.push_back(benchmarkKernel("register_blocking", matrix_multiplication_kernel_register_blocking,
                                     d_A, d_B, d_C, reference, M, N, K));
    
    results.push_back(benchmarkKernel("prefetch", matrix_multiplication_kernel_prefetch,
                                     d_A, d_B, d_C, reference, M, N, K));
    
    // Print results
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << std::setw(20) << "Kernel" << std::setw(12) << "Time (ms)" 
              << std::setw(12) << "GFLOPS" << std::setw(12) << "Correct" << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    for (const auto& result : results) {
        std::cout << std::setw(20) << result.name 
                  << std::setw(12) << std::fixed << std::setprecision(3) << result.time_ms
                  << std::setw(12) << std::fixed << std::setprecision(1) << result.gflops
                  << std::setw(12) << (result.correct ? "✓" : "✗") << std::endl;
    }
    
    // Cleanup
    checkCudaError(cudaFree(d_A), "Failed to free d_A");
    checkCudaError(cudaFree(d_B), "Failed to free d_B");
    checkCudaError(cudaFree(d_C), "Failed to free d_C");
    
    std::cout << "\nBenchmark completed successfully!" << std::endl;
    std::cout << "\nTo profile with NCU, run:" << std::endl;
    std::cout << "ncu --set full -o profile_naive ./benchmark" << std::endl;
    std::cout << "ncu --set full -o profile_shared ./benchmark" << std::endl;
    std::cout << "ncu --set full -o profile_register ./benchmark" << std::endl;
    std::cout << "ncu --set full -o profile_prefetch ./benchmark" << std::endl;
    
    return 0;
} 