// ═══════════════════════════════════════════════════════════════════════════════
// test_gemm.cu  —  Correctness tests for INT8 GEMM kernel
//
// Validates that our INT8 GEMM produces results consistent with a naive FP32
// reference GEMM computed on the CPU.
//
// Run with: ./build/tests/test_gemm
// ═══════════════════════════════════════════════════════════════════════════════

#include "gemm_int8.cuh"
#include "quantize.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

static int tests_run = 0, tests_passed = 0;

#define ASSERT_TRUE(cond, msg)                                      \
    do {                                                             \
        ++tests_run;                                                 \
        if (!(cond)) { printf("  FAIL: %s (line %d)\n", msg, __LINE__); } \
        else         { ++tests_passed; printf("  PASS: %s\n", msg); }  \
    } while(0)

// CPU reference GEMM in INT32 (ground truth)
static void cpu_gemm_int8(
    const int8_t* A, const int8_t* B, int32_t* C,
    int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < K; ++k)
                acc += static_cast<int32_t>(A[i*K+k]) * static_cast<int32_t>(B[k*N+j]);
            C[i*N+j] = acc;
        }
}

void test_gemm_small()
{
    printf("\n[Test 1] Small GEMM (M=16, N=16, K=16) vs CPU reference\n");

    const int M=16, N=16, K=16;
    int8_t  h_A[M*K], h_B[K*N];
    int32_t h_C_gpu[M*N], h_C_cpu[M*N];

    srand(7);
    for (int i = 0; i < M*K; ++i) h_A[i] = static_cast<int8_t>((rand() % 254) - 127);
    for (int i = 0; i < K*N; ++i) h_B[i] = static_cast<int8_t>((rand() % 254) - 127);

    cpu_gemm_int8(h_A, h_B, h_C_cpu, M, N, K);

    int8_t  *d_A, *d_B;
    int32_t *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(int8_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(int8_t), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N+BLOCK_N-1)/BLOCK_N, (M+BLOCK_M-1)/BLOCK_M);
    gemm_int8<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaMemcpy(h_C_gpu, d_C, M*N*sizeof(int32_t), cudaMemcpyDeviceToHost));

    bool match = true;
    for (int i = 0; i < M*N; ++i)
        if (h_C_gpu[i] != h_C_cpu[i]) { match = false; break; }

    ASSERT_TRUE(match, "INT8 GEMM output matches CPU INT32 reference exactly");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

void test_fused_gemm_accuracy()
{
    printf("\n[Test 2] Fused FP32→INT8→FP32 GEMM round-trip error\n");

    const int M=32, N=32, K=32;
    float h_A[M*K], h_B[K*N], h_C[M*N], h_C_ref[M*N];

    srand(42);
    for (int i = 0; i < M*K; ++i) h_A[i] = (rand()/(float)RAND_MAX)*2.f-1.f;
    for (int i = 0; i < K*N; ++i) h_B[i] = (rand()/(float)RAND_MAX)*2.f-1.f;

    // FP32 reference on CPU
    for (int i=0;i<M;++i)
        for (int j=0;j<N;++j) {
            float acc=0;
            for (int k=0;k<K;++k) acc+=h_A[i*K+k]*h_B[k*N+j];
            h_C_ref[i*N+j]=acc;
        }

    float scale_A = compute_symmetric_scale(h_A, M*K);
    float scale_B = compute_symmetric_scale(h_B, K*N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M*K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M*N*sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_N, BLOCK_M);
    dim3 grid((N+BLOCK_N-1)/BLOCK_N, (M+BLOCK_M-1)/BLOCK_M);
    gemm_int8_fused<<<grid, block>>>(d_A, d_B, d_C, M, N, K, scale_A, scale_B);
    CUDA_CHECK(cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost));

    float max_err=0, mean_err=0;
    for (int i=0;i<M*N;++i) {
        float e=fabsf(h_C[i]-h_C_ref[i]);
        if(e>max_err) max_err=e;
        mean_err+=e;
    }
    mean_err/=(M*N);
    printf("    Max error: %.5f  Mean error: %.5f\n", max_err, mean_err);

    // Allow ~1% relative error given INT8 quantization of both A and B
    float max_ref = 0;
    for (int i=0;i<M*N;++i) if(fabsf(h_C_ref[i])>max_ref) max_ref=fabsf(h_C_ref[i]);
    float rel_err = (max_ref > 0) ? max_err / max_ref : max_err;
    ASSERT_TRUE(rel_err < 0.05f, "Fused GEMM relative error < 5% (INT8 quantization budget)");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main()
{
    printf("═══════════════════════════════════════\n");
    printf("  INT8 GEMM kernel correctness tests\n");
    printf("═══════════════════════════════════════\n");

    test_gemm_small();
    test_fused_gemm_accuracy();

    printf("\n═══════════════════════════════════════\n");
    printf("  Results: %d / %d tests passed\n", tests_passed, tests_run);
    printf("═══════════════════════════════════════\n\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
