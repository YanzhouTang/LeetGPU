# Makefile for Matrix Multiplication Benchmark

# CUDA compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_70 -lcublas

# Source files
SOLUTION_SRC = solution.cu
BENCHMARK_SRC = benchmark.cu

# Object files
SOLUTION_OBJ = solution.o
BENCHMARK_OBJ = benchmark.o

# Executable
TARGET = benchmark

# Profile output directory
PROFILE_DIR = profiles

# Matrix dimensions for profiling
M ?= 8192
N ?= 6144  
K ?= 4096

# Comprehensive metrics
COMPUTE_METRICS = smsp__cycles_elapsed.avg,smsp__cycles_elapsed.sum,smsp__inst_executed.avg.per_cycle_elapsed,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
MEMORY_METRICS = dram__bytes_read.sum,dram__bytes_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__data_bank_reads.sum,l1tex__data_bank_writes.sum,l1tex__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed
OCCUPANCY_METRICS = smsp__warps_active.avg.pct_of_peak_sustained_elapsed,smsp__maximum_warps_per_active_cycle_pct,sm__warps_active.avg.pct_of_peak_sustained_elapsed
CACHE_METRICS = l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sector_hit_rate.pct,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio

# Default target
all: $(TARGET)

# Compile solution.cu to object file
$(SOLUTION_OBJ): $(SOLUTION_SRC)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Compile benchmark.cu to object file
$(BENCHMARK_OBJ): $(BENCHMARK_SRC)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Link object files to create executable
$(TARGET): $(SOLUTION_OBJ) $(BENCHMARK_OBJ)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@

# Create profile directory
$(PROFILE_DIR):
	mkdir -p $(PROFILE_DIR)

# Clean build artifacts
clean:
	rm -f $(SOLUTION_OBJ) $(BENCHMARK_OBJ) $(TARGET) *.ncu-rep
	rm -rf $(PROFILE_DIR)

# Run benchmark
run: $(TARGET)
	./$(TARGET)

# Run with specific matrix size
run-size: $(TARGET)
	./$(TARGET) 2048 2048 2048

# Run with custom size
run-custom: $(TARGET)
	./$(TARGET) $(M) $(N) $(K)

# Basic profiling (single file per kernel)
profile-naive: $(TARGET) $(PROFILE_DIR)
	ncu --set full -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive ./$(TARGET) $(M) $(N) $(K)

profile-shared: $(TARGET) $(PROFILE_DIR)
	ncu --set full -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared ./$(TARGET) $(M) $(N) $(K)

profile-register: $(TARGET) $(PROFILE_DIR)
	ncu --set full -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register ./$(TARGET) $(M) $(N) $(K)

profile-prefetch: $(TARGET) $(PROFILE_DIR)
	ncu --set full -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch ./$(TARGET) $(M) $(N) $(K)

# Comprehensive profiling (multiple analysis files per kernel)
profile-naive-full: $(TARGET) $(PROFILE_DIR)
	@echo "Profiling naive kernel with comprehensive metrics..."
	ncu --set full -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_full ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(COMPUTE_METRICS)" -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_compute ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(MEMORY_METRICS)" -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_memory ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(OCCUPANCY_METRICS)" -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_occupancy ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(CACHE_METRICS)" -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_cache ./$(TARGET) $(M) $(N) $(K)

profile-shared-full: $(TARGET) $(PROFILE_DIR)
	@echo "Profiling shared memory kernel with comprehensive metrics..."
	ncu --set full -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_full ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(COMPUTE_METRICS)" -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_compute ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(MEMORY_METRICS)" -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_memory ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(OCCUPANCY_METRICS)" -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_occupancy ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(CACHE_METRICS)" -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_cache ./$(TARGET) $(M) $(N) $(K)

profile-register-full: $(TARGET) $(PROFILE_DIR)
	@echo "Profiling register blocking kernel with comprehensive metrics..."
	ncu --set full -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_full ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(COMPUTE_METRICS)" -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_compute ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(MEMORY_METRICS)" -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_memory ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(OCCUPANCY_METRICS)" -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_occupancy ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(CACHE_METRICS)" -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_cache ./$(TARGET) $(M) $(N) $(K)

profile-prefetch-full: $(TARGET) $(PROFILE_DIR)
	@echo "Profiling prefetch kernel with comprehensive metrics..."
	ncu --set full -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_full ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(COMPUTE_METRICS)" -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_compute ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(MEMORY_METRICS)" -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_memory ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(OCCUPANCY_METRICS)" -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_occupancy ./$(TARGET) $(M) $(N) $(K)
	ncu --metrics "$(CACHE_METRICS)" -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_cache ./$(TARGET) $(M) $(N) $(K)

# Profile all kernels (basic)
profile-all: profile-naive profile-shared profile-register profile-prefetch

# Profile all kernels (comprehensive)
profile-all-full: profile-naive-full profile-shared-full profile-register-full profile-prefetch-full

# Quick profiling with basic metrics
profile-quick: $(TARGET) $(PROFILE_DIR)
	ncu --metrics smsp__cycles_elapsed.avg,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum -o $(PROFILE_DIR)/profile_quick ./$(TARGET) $(M) $(N) $(K)

# Fast profiling for large matrices (minimal metrics)
profile-fast: $(TARGET) $(PROFILE_DIR)
	@echo "Fast profiling with minimal metrics for large matrices..."
	ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__warps_active.avg.pct_of_peak_sustained_elapsed -o $(PROFILE_DIR)/profile_fast ./$(TARGET) $(M) $(N) $(K)

# Timing-only profiling (fastest)
profile-timing: $(TARGET) $(PROFILE_DIR)
	@echo "Timing-only profiling (fastest)..."
	ncu --metrics gpu__time_duration.sum -o $(PROFILE_DIR)/profile_timing ./$(TARGET) $(M) $(N) $(K)

# Individual kernel fast profiling
profile-naive-fast: $(TARGET) $(PROFILE_DIR)
	ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum -k matrix_multiplication_kernel_naive -o $(PROFILE_DIR)/profile_naive_fast ./$(TARGET) $(M) $(N) $(K)

profile-shared-fast: $(TARGET) $(PROFILE_DIR)
	ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum -k matrix_multiplication_kernel_sharemem -o $(PROFILE_DIR)/profile_shared_fast ./$(TARGET) $(M) $(N) $(K)

profile-register-fast: $(TARGET) $(PROFILE_DIR)
	ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum -k matrix_multiplication_kernel_register_blocking -o $(PROFILE_DIR)/profile_register_fast ./$(TARGET) $(M) $(N) $(K)

profile-prefetch-fast: $(TARGET) $(PROFILE_DIR)
	ncu --metrics gpu__time_duration.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,dram__bytes_read.sum -k matrix_multiplication_kernel_prefetch -o $(PROFILE_DIR)/profile_prefetch_fast ./$(TARGET) $(M) $(N) $(K)

# Memory-focused profiling
profile-memory: $(TARGET) $(PROFILE_DIR)
	ncu --metrics "$(MEMORY_METRICS)" -o $(PROFILE_DIR)/profile_memory ./$(TARGET) $(M) $(N) $(K)

# View profiles in NCU GUI
view-naive:
	ncu-ui $(PROFILE_DIR)/profile_naive_full.ncu-rep

view-shared:
	ncu-ui $(PROFILE_DIR)/profile_shared_full.ncu-rep

view-register:
	ncu-ui $(PROFILE_DIR)/profile_register_full.ncu-rep

view-prefetch:
	ncu-ui $(PROFILE_DIR)/profile_prefetch_full.ncu-rep

# Generate text reports
report-naive:
	ncu --import $(PROFILE_DIR)/profile_naive_full.ncu-rep --print-summary

report-shared:
	ncu --import $(PROFILE_DIR)/profile_shared_full.ncu-rep --print-summary

report-register:
	ncu --import $(PROFILE_DIR)/profile_register_full.ncu-rep --print-summary

report-prefetch:
	ncu --import $(PROFILE_DIR)/profile_prefetch_full.ncu-rep --print-summary

# Help target
help:
	@echo "Matrix Multiplication Benchmark Makefile"
	@echo "========================================"
	@echo ""
	@echo "Build targets:"
	@echo "  all                 - Build benchmark executable"
	@echo "  clean               - Remove build artifacts and profiles"
	@echo ""
	@echo "Run targets:"
	@echo "  run                 - Run benchmark with default size (8192x6144x4096)"
	@echo "  run-size            - Run benchmark with 2048x2048x2048"
	@echo "  run-custom          - Run with custom size: make run-custom M=1024 N=1024 K=1024"
	@echo ""
	@echo "Basic profiling (single file per kernel):"
	@echo "  profile-naive       - Profile naive kernel"
	@echo "  profile-shared      - Profile shared memory kernel"
	@echo "  profile-register    - Profile register blocking kernel"
	@echo "  profile-prefetch    - Profile prefetch kernel"
	@echo "  profile-all         - Profile all kernels (basic)"
	@echo ""
	@echo "Comprehensive profiling (multiple analysis files per kernel):"
	@echo "  profile-naive-full  - Comprehensive profiling of naive kernel"
	@echo "  profile-shared-full - Comprehensive profiling of shared memory kernel"
	@echo "  profile-register-full - Comprehensive profiling of register blocking kernel"
	@echo "  profile-prefetch-full - Comprehensive profiling of prefetch kernel"
	@echo "  profile-all-full    - Comprehensive profiling of all kernels"
	@echo ""
	@echo "Quick profiling:"
	@echo "  profile-quick       - Quick profiling with basic metrics"
	@echo "  profile-fast        - Fast profiling for large matrices (minimal metrics)"
	@echo "  profile-timing      - Timing-only profiling (fastest)"
	@echo "  profile-memory      - Memory-focused profiling"
	@echo ""
	@echo "Fast individual kernel profiling:"
	@echo "  profile-naive-fast  - Fast profiling of naive kernel only"
	@echo "  profile-shared-fast - Fast profiling of shared memory kernel only"
	@echo "  profile-register-fast - Fast profiling of register blocking kernel only"
	@echo "  profile-prefetch-fast - Fast profiling of prefetch kernel only"
	@echo ""
	@echo "View results:"
	@echo "  view-naive          - Open naive kernel profile in NCU GUI"
	@echo "  view-shared         - Open shared memory kernel profile in NCU GUI"
	@echo "  view-register       - Open register blocking kernel profile in NCU GUI"
	@echo "  view-prefetch       - Open prefetch kernel profile in NCU GUI"
	@echo ""
	@echo "Generate reports:"
	@echo "  report-naive        - Generate text summary for naive kernel"
	@echo "  report-shared       - Generate text summary for shared memory kernel"
	@echo "  report-register     - Generate text summary for register blocking kernel"
	@echo "  report-prefetch     - Generate text summary for prefetch kernel"
	@echo ""
	@echo "Examples:"
	@echo "  make profile-all-full M=2048 N=2048 K=2048"
	@echo "  make profile-prefetch-full"
	@echo "  make view-prefetch"

.PHONY: all clean run run-size run-custom profile-naive profile-shared profile-register profile-prefetch profile-naive-full profile-shared-full profile-register-full profile-prefetch-full profile-all profile-all-full profile-quick profile-fast profile-timing profile-naive-fast profile-shared-fast profile-register-fast profile-prefetch-fast profile-memory view-naive view-shared view-register view-prefetch report-naive report-shared report-register report-prefetch help 