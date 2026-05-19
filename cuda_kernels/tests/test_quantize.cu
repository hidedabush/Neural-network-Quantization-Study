// ═══════════════════════════════════════════════════════════════════════════════
// test_quantize.cu  —  Correctness tests for quantize/dequantize kernels
//
// These are unit tests, not benchmarks. They verify:
//   1. Quantized values fall in the expected integer range
//   2. Round-trip error is bounded by theory
//   3. Edge cases: all-zero input, very large values, negative values
//
// Run with: ./build/tests/test_quantize
// All tests print PASS or FAIL. Any FAIL means a kernel bug.
// ═══════════════════════════════════════════════════════════════════════════════

#include "quantize.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>

static int tests_run    = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond, msg)                          \
    do {                                                \
        ++tests_run;                                    \
        if (!(cond)) {                                  \
            printf("  FAIL: %s  (line %d)\n", msg, __LINE__); \
        } else {                                        \
            ++tests_passed;                             \
            printf("  PASS: %s\n", msg);                \
        }                                               \
    } while(0)

#define ASSERT_NEAR(a, b, tol, msg)   ASSERT_TRUE(fabsf((a)-(b)) <= (tol), msg)


// ── Test 1: Basic round-trip ──────────────────────────────────────────────────
void test_roundtrip_basic()
{
    printf("\n[Test 1] Basic symmetric round-trip\n");

    const int N = 1024;
    float h_in[N], h_out[N];
    int8_t h_int8[N];

    // Uniform values in [-0.5, 0.5]
    for (int i = 0; i < N; ++i)
        h_in[i] = (static_cast<float>(i) / N) - 0.5f;

    float scale = compute_symmetric_scale(h_in, N);

    float *d_in, *d_out;
    int8_t *d_int8;
    CUDA_CHECK( cudaMalloc(&d_in,   N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_out,  N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_int8, N * sizeof(int8_t)) );

    CUDA_CHECK( cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice) );

    int grid = (N + 255) / 256;
    quantize_symmetric_int8  <<<grid, 256>>>(d_in,   d_int8, N, scale);
    dequantize_symmetric_int8<<<grid, 256>>>(d_int8, d_out,  N, scale);

    CUDA_CHECK( cudaMemcpy(h_int8, d_int8, N * sizeof(int8_t), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(h_out,  d_out,  N * sizeof(float),  cudaMemcpyDeviceToHost) );

    // Check INT8 values stay in [-127, 127]
    bool range_ok = true;
    for (int i = 0; i < N; ++i)
        if (h_int8[i] < -127 || h_int8[i] > 127) { range_ok = false; break; }
    ASSERT_TRUE(range_ok, "All INT8 values in [-127, 127]");

    // Check round-trip error bounded by 0.5 * scale
    float max_err = 0.f;
    for (int i = 0; i < N; ++i)
        max_err = fmaxf(max_err, fabsf(h_in[i] - h_out[i]));
    ASSERT_TRUE(max_err <= 0.5f * scale + 1e-6f, "Round-trip error <= 0.5 * scale");

    CUDA_CHECK( cudaFree(d_in) );
    CUDA_CHECK( cudaFree(d_out) );
    CUDA_CHECK( cudaFree(d_int8) );
}


// ── Test 2: All-zero input ────────────────────────────────────────────────────
void test_all_zeros()
{
    printf("\n[Test 2] All-zero input\n");

    const int N = 256;
    float h_in[N] = {};    // all zeros
    int8_t h_int8[N];

    float scale = compute_symmetric_scale(h_in, N);
    ASSERT_NEAR(scale, 1.0f, 1e-6f, "Scale for all-zeros is 1.0 (safe fallback)");

    float *d_in;
    int8_t *d_int8;
    CUDA_CHECK( cudaMalloc(&d_in,   N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_int8, N * sizeof(int8_t)) );
    CUDA_CHECK( cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice) );

    quantize_symmetric_int8<<<1, 256>>>(d_in, d_int8, N, scale);
    CUDA_CHECK( cudaMemcpy(h_int8, d_int8, N * sizeof(int8_t), cudaMemcpyDeviceToHost) );

    bool all_zero = true;
    for (int i = 0; i < N; ++i)
        if (h_int8[i] != 0) { all_zero = false; break; }
    ASSERT_TRUE(all_zero, "Quantized all-zero input gives all INT8 zeros");

    CUDA_CHECK( cudaFree(d_in) );
    CUDA_CHECK( cudaFree(d_int8) );
}


// ── Test 3: Extreme values clamped to ±127 ───────────────────────────────────
void test_clamping()
{
    printf("\n[Test 3] Clamping extreme values\n");

    const int N = 4;
    float h_in[4] = { 1.0f, -1.0f, 100.0f, -100.0f };
    int8_t h_int8[4];

    // scale = 100 / 127
    float scale = compute_symmetric_scale(h_in, N);

    float *d_in;
    int8_t *d_int8;
    CUDA_CHECK( cudaMalloc(&d_in,   N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_int8, N * sizeof(int8_t)) );
    CUDA_CHECK( cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice) );

    quantize_symmetric_int8<<<1, 32>>>(d_in, d_int8, N, scale);
    CUDA_CHECK( cudaMemcpy(h_int8, d_int8, N * sizeof(int8_t), cudaMemcpyDeviceToHost) );

    // 100.0 / (100/127) = 127.0 → should clamp to 127
    ASSERT_TRUE(h_int8[2] == 127,  "Max value maps to 127");
    ASSERT_TRUE(h_int8[3] == -127, "Min value maps to -127");

    CUDA_CHECK( cudaFree(d_in) );
    CUDA_CHECK( cudaFree(d_int8) );
}


// ── Test 4: Tiled kernel matches naive kernel ─────────────────────────────────
void test_tiled_matches_naive()
{
    printf("\n[Test 4] Tiled kernel output matches naive kernel\n");

    const int N = 4096;
    float h_in[N];
    int8_t h_naive[N], h_tiled[N];

    srand(123);
    for (int i = 0; i < N; ++i)
        h_in[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.f - 1.f;

    float scale = compute_symmetric_scale(h_in, N);

    float *d_in;
    int8_t *d_naive, *d_tiled;
    CUDA_CHECK( cudaMalloc(&d_in,     N * sizeof(float)) );
    CUDA_CHECK( cudaMalloc(&d_naive,  N * sizeof(int8_t)) );
    CUDA_CHECK( cudaMalloc(&d_tiled,  N * sizeof(int8_t)) );
    CUDA_CHECK( cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice) );

    int grid = (N + 255) / 256;
    quantize_symmetric_int8      <<<grid, 256>>>(d_in, d_naive, N, scale);
    quantize_symmetric_int8_tiled<<<grid, 256>>>(d_in, d_tiled, N, scale);

    CUDA_CHECK( cudaMemcpy(h_naive, d_naive, N * sizeof(int8_t), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(h_tiled, d_tiled, N * sizeof(int8_t), cudaMemcpyDeviceToHost) );

    bool match = true;
    for (int i = 0; i < N; ++i)
        if (h_naive[i] != h_tiled[i]) { match = false; break; }
    ASSERT_TRUE(match, "Tiled output bit-identical to naive output");

    CUDA_CHECK( cudaFree(d_in) );
    CUDA_CHECK( cudaFree(d_naive) );
    CUDA_CHECK( cudaFree(d_tiled) );
}


int main()
{
    printf("═══════════════════════════════════════\n");
    printf("  Quantization kernel correctness tests\n");
    printf("═══════════════════════════════════════\n");

    test_roundtrip_basic();
    test_all_zeros();
    test_clamping();
    test_tiled_matches_naive();

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d / %d tests passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
