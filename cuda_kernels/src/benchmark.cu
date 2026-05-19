// ═══════════════════════════════════════════════════════════════════════════════
// benchmark.cu  —  Standalone latency and throughput benchmarks
//
// Run this after completing each kernel to log before/after performance.
// Copy the output into your daily_log and results/benchmarks/.
//
// Usage:  ./build/benchmark [N]
//         N = number of elements (default 1<<24 = 16M elements)
//
// What to record from each run (paste into day_NNN.md):
//   - Kernel name
//   - N (problem size)
//   - Elapsed time (ms)
//   - Throughput (GB/s)
//   - Comparison vs theoretical peak
//
// RTX 4060 Laptop theoretical peak bandwidth: ~272 GB/s (GDDR6)
// If your kernel is hitting < 50% of that, there is optimization opportunity.
// ═══════════════════════════════════════════════════════════════════════════════

#include "quantize.cuh"
#include "gemm_int8.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

// ── Timing helper ──────────────────────────────────────────────────────────────
// cudaEvent timing is the standard way to measure GPU kernel duration.
// It's more accurate than host-side timing (which includes kernel launch overhead)
// and works even for async kernels.
static float time_kernel_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0.f;
    CUDA_CHECK( cudaEventSynchronize(stop) );
    CUDA_CHECK( cudaEventElapsedTime(&ms, start, stop) );
    return ms;
}

// ── Print a benchmark result row ──────────────────────────────────────────────
static void print_result(const char* name, long n, float ms, size_t bytes_moved)
{
    double gb_s = (bytes_moved / 1e9) / (ms / 1e3);
    printf("  %-42s  N=%-10ld  %.3f ms  %.1f GB/s\n", name, n, ms, gb_s);
}

int main(int argc, char** argv)
{
    long N = (argc > 1) ? atol(argv[1]) : (1L << 24);  // 16M elements default

    printf("\n══════════════════════════════════════════════════════════\n");
    printf("  quant_study benchmark  |  N = %ld elements\n", N);
    printf("══════════════════════════════════════════════════════════\n\n");

    // Print GPU info for the log
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK( cudaGetDevice(&device) );
    CUDA_CHECK( cudaGetDeviceProperties(&prop, device) );
    printf("GPU: %s  |  SM count: %d  |  Compute: %d.%d\n\n",
           prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    // ── Allocate host memory ────────────────────────────────────────────────
    float*   h_fp32_in  = new float[N];
    float*   h_fp32_out = new float[N];
    int8_t*  h_int8_out = new int8_t[N];

    // Fill with random values in [-1, 1] (typical for normalized weights)
    srand(42);
    for (long i = 0; i < N; ++i)
        h_fp32_in[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f;

    float scale = compute_symmetric_scale(h_fp32_in, static_cast<int>(N));
    printf("Computed scale factor: %.6f\n\n", scale);

    // ── Allocate device memory ──────────────────────────────────────────────
    float*   d_fp32_in;
    float*   d_fp32_out;
    int8_t*  d_int8;

    CUDA_CHECK( cudaMalloc(&d_fp32_in,  N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_fp32_out, N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_int8,     N * sizeof(int8_t)) );

    CUDA_CHECK( cudaMemcpy(d_fp32_in, h_fp32_in, N * sizeof(float), cudaMemcpyHostToDevice) );

    // ── CUDA events for timing ──────────────────────────────────────────────
    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop) );

    const int BLOCK = 256;
    int grid = static_cast<int>((N + BLOCK - 1) / BLOCK);

    // Number of warm-up runs (don't time these) + timed runs
    const int WARMUP = 3;
    const int RUNS   = 10;

    printf("Benchmarks (avg of %d runs, %d warm-up):\n\n", RUNS, WARMUP);

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 1: Naive quantize (FP32 → INT8)
    // Bytes moved: N*4 (read) + N*1 (write) = 5N bytes
    // ══════════════════════════════════════════════════════════════════════
    for (int r = 0; r < WARMUP; ++r)
        quantize_symmetric_int8<<<grid, BLOCK>>>(d_fp32_in, d_int8, N, scale);
    cudaDeviceSynchronize();

    float total_ms = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK( cudaEventRecord(start) );
        quantize_symmetric_int8<<<grid, BLOCK>>>(d_fp32_in, d_int8, N, scale);
        CUDA_CHECK( cudaEventRecord(stop) );
        total_ms += time_kernel_ms(start, stop);
    }
    print_result("quantize_symmetric_int8 (naive)", N, total_ms / RUNS,
                 N * (sizeof(float) + sizeof(int8_t)));

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 2: Tiled quantize (shared memory version)
    // Same bytes moved — compare latency. Document difference in journal.
    // ══════════════════════════════════════════════════════════════════════
    for (int r = 0; r < WARMUP; ++r)
        quantize_symmetric_int8_tiled<<<grid, BLOCK>>>(d_fp32_in, d_int8, N, scale);
    cudaDeviceSynchronize();

    total_ms = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK( cudaEventRecord(start) );
        quantize_symmetric_int8_tiled<<<grid, BLOCK>>>(d_fp32_in, d_int8, N, scale);
        CUDA_CHECK( cudaEventRecord(stop) );
        total_ms += time_kernel_ms(start, stop);
    }
    print_result("quantize_symmetric_int8 (tiled smem)", N, total_ms / RUNS,
                 N * (sizeof(float) + sizeof(int8_t)));

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 3: Dequantize (INT8 → FP32)
    // ══════════════════════════════════════════════════════════════════════
    total_ms = 0.f;
    for (int r = 0; r < RUNS; ++r) {
        CUDA_CHECK( cudaEventRecord(start) );
        dequantize_symmetric_int8<<<grid, BLOCK>>>(d_int8, d_fp32_out, N, scale);
        CUDA_CHECK( cudaEventRecord(stop) );
        total_ms += time_kernel_ms(start, stop);
    }
    print_result("dequantize_symmetric_int8", N, total_ms / RUNS,
                 N * (sizeof(int8_t) + sizeof(float)));

    // ══════════════════════════════════════════════════════════════════════
    // Benchmark 4: Round-trip accuracy check
    // Copy back to host and compute max absolute error vs original.
    // Log this number — it should be at most 0.5 * scale.
    // ══════════════════════════════════════════════════════════════════════
    CUDA_CHECK( cudaMemcpy(h_fp32_out, d_fp32_out, N * sizeof(float), cudaMemcpyDeviceToHost) );

    float max_err = 0.f, mean_err = 0.f;
    for (long i = 0; i < N; ++i) {
        float err = fabsf(h_fp32_in[i] - h_fp32_out[i]);
        if (err > max_err) max_err = err;
        mean_err += err;
    }
    mean_err /= static_cast<float>(N);
    printf("\n  Round-trip accuracy (quantize → dequantize):\n");
    printf("    Max absolute error  : %.6f  (theoretical max: %.6f)\n",
           max_err, 0.5f * scale);
    printf("    Mean absolute error : %.6f\n\n", mean_err);

    // ── Cleanup ────────────────────────────────────────────────────────────
    CUDA_CHECK( cudaEventDestroy(start) );
    CUDA_CHECK( cudaEventDestroy(stop) );
    CUDA_CHECK( cudaFree(d_fp32_in) );
    CUDA_CHECK( cudaFree(d_fp32_out) );
    CUDA_CHECK( cudaFree(d_int8) );

    delete[] h_fp32_in;
    delete[] h_fp32_out;
    delete[] h_int8_out;

    printf("══════════════════════════════════════════════════════════\n");
    printf("  Done. Copy these numbers into results/benchmarks/ and\n");
    printf("  your daily log.\n");
    printf("══════════════════════════════════════════════════════════\n\n");

    return 0;
}
