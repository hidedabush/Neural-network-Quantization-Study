cmake_minimum_required(VERSION 3.20)
project(quant_study LANGUAGES CXX CUDA)

# ── C++ / CUDA standards ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# ── GPU architecture ───────────────────────────────────────────────────────────
# RTX 4060 Laptop = Ada Lovelace = sm_89
# Change this if you switch GPUs. Find yours with: nvidia-smi --query-gpu=compute_cap
set(CMAKE_CUDA_ARCHITECTURES 89)

# ── Compiler flags ─────────────────────────────────────────────────────────────
# --generate-line-info : embeds source line info for Nsight Compute profiling
# --ptxas-options=-v   : print register/smem usage per kernel (useful for occupancy tuning)
# -O3                  : full optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info --ptxas-options=-v -O3")

# ── Include directories ────────────────────────────────────────────────────────
include_directories(cuda_kernels/include)

# ── Library: quantization kernels ─────────────────────────────────────────────
# All .cu files in cuda_kernels/src/ are compiled into a shared library
# that can be linked by tests and (later) Python bindings.
add_library(quant_kernels STATIC
    cuda_kernels/src/quantize.cu
    cuda_kernels/src/gemm_int8.cu
)

target_include_directories(quant_kernels PUBLIC cuda_kernels/include)

# ── Executable: standalone benchmark ──────────────────────────────────────────
add_executable(benchmark cuda_kernels/src/benchmark.cu)
target_link_libraries(benchmark quant_kernels)

# ── Test executables ───────────────────────────────────────────────────────────
enable_testing()

add_executable(test_quantize cuda_kernels/tests/test_quantize.cu)
target_link_libraries(test_quantize quant_kernels)
add_test(NAME test_quantize COMMAND test_quantize)

add_executable(test_gemm cuda_kernels/tests/test_gemm.cu)
target_link_libraries(test_gemm quant_kernels)
add_test(NAME test_gemm COMMAND test_gemm)
