// ═══════════════════════════════════════════════════════════════════════════════
// gemm_int8.cu  —  INT8 GEMM kernel implementation
//
// Implement after completing Days 9–14 (quantize kernels + shared memory).
// Read this file together with the tiling diagram in docs/paper/report_draft.md.
// ═══════════════════════════════════════════════════════════════════════════════

#include "gemm_int8.cuh"
#include "quantize.cuh"
#include <cuda_runtime.h>

// ══════════════════════════════════════════════════════════════════════════════
// Tiled INT8 GEMM
// ══════════════════════════════════════════════════════════════════════════════
//
// ALGORITHM (tiled matrix multiply):
//
//   The output matrix C has M×N elements. We assign one thread block to compute
//   a BLOCK_M × BLOCK_N tile of C.
//
//   Within a block, threads cooperate to load tiles of A and B into shared memory,
//   then each thread accumulates its partial dot products.
//
//   Pseudocode for thread (row, col) computing C[row_global, col_global]:
//
//     accumulator = 0  (INT32 — must be 32-bit to avoid overflow of 127*127*K products)
//     for tile_k in range(0, K, BLOCK_K):
//       load A[row_global, tile_k .. tile_k+BLOCK_K] → shared_A[threadIdx.y, 0..BLOCK_K]
//       load B[tile_k .. tile_k+BLOCK_K, col_global] → shared_B[0..BLOCK_K, threadIdx.x]
//       __syncthreads()
//       for k in range(BLOCK_K):
//         accumulator += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x]
//       __syncthreads()   ← barrier before loading next tile
//     C[row_global, col_global] = accumulator
//
// WHY DOES THIS BEAT THE NAIVE KERNEL?
//   Naive: each thread loads K elements of A and K elements of B from global mem.
//   Total global loads = M*N*2*K  (one per thread per element of the inner loop)
//
//   Tiled: each tile of A is loaded once and shared by BLOCK_N threads.
//          each tile of B is loaded once and shared by BLOCK_M threads.
//   Total global loads = M*N*2*K / BLOCK_SIZE  → BLOCK_SIZE× fewer DRAM accesses.
//   For BLOCK_SIZE=16: 16× fewer loads. For BLOCK_SIZE=32: 32× fewer.
//
// OVERFLOW ANALYSIS:
//   Each multiply: INT8 * INT8 = product fits in INT16 (max 127*127 = 16129).
//   Accumulated over K steps: max = 16129 * K.
//   For K=512 (typical hidden dim): max accumulation = 8,258,048 → needs INT32.
//   We use int32_t for shared_A dot products and the accumulator.
//   (Some hardware uses int16 accumulators; Ada INT8 Tensor Cores use INT32.)

__global__ void gemm_int8(
    const int8_t* A,
    const int8_t* B,
    int32_t*      C,
    int           M,
    int           N,
    int           K)
{
    // Shared memory tiles — loaded cooperatively by the block
    __shared__ int8_t shared_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t shared_B[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;  // this thread's row in C
    int col = blockIdx.x * BLOCK_N + threadIdx.x;  // this thread's col in C

    int32_t accumulator = 0;

    // Loop over tiles along the K dimension
    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {

        // ── Cooperative load: tile of A ──────────────────────────────────────
        // Thread (ty, tx) loads A[row, tile_k + tx] into shared_A[ty][tx].
        // If out of bounds, pad with 0 (contributes nothing to the dot product).
        if (row < M && (tile_k + threadIdx.x) < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + tile_k + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        // ── Cooperative load: tile of B ──────────────────────────────────────
        // Thread (ty, tx) loads B[tile_k + ty, col] into shared_B[ty][tx].
        if ((tile_k + threadIdx.y) < K && col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile_k + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        // Barrier: all threads must finish loading before any thread reads smem
        __syncthreads();

        // ── Compute partial dot product for this tile ────────────────────────
        // Each thread accumulates BLOCK_K multiply-adds.
        // Cast to int32 before multiply to prevent INT8 overflow in the product.
        for (int k = 0; k < BLOCK_K; ++k) {
            accumulator += static_cast<int32_t>(shared_A[threadIdx.y][k])
                         * static_cast<int32_t>(shared_B[k][threadIdx.x]);
        }

        // Barrier: all threads must finish computing before next tile is loaded
        // (otherwise a fast thread could overwrite shared_A before a slow one reads it)
        __syncthreads();
    }

    // Write result — only for valid output positions
    if (row < M && col < N) {
        C[row * N + col] = accumulator;
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Fused: FP32 input → quantize → INT8 GEMM → dequantize → FP32 output
// ══════════════════════════════════════════════════════════════════════════════
//
// MOTIVATION:
//   Doing this in three separate kernel launches requires:
//     1. Write INT8(A) to global memory
//     2. Write INT8(B) to global memory
//     3. Read INT8(A), INT8(B) from global memory for GEMM
//     4. Write INT32(C) to global memory
//     5. Read INT32(C), write FP32(C) to global memory
//   That's 5 global memory passes for what could be 2 (read FP32 inputs, write FP32 output).
//
//   Fusion eliminates the intermediate INT8 and INT32 buffers, keeping them in registers.
//   This is exactly what TensorRT does internally when it fuses quantize+GEMM+dequantize.
//
// NOTE: This naive fusion still has a tile load from global memory. A fully fused
// Tensor Core kernel would use PTX-level wmma intrinsics. That's an optional
// extension for Day 20 if you want to go deeper.
__global__ void gemm_int8_fused(
    const float* A_fp32,
    const float* B_fp32,
    float*       C_fp32,
    int          M,
    int          N,
    int          K,
    float        scale_A,
    float        scale_B)
{
    __shared__ int8_t shared_A[BLOCK_M][BLOCK_K];
    __shared__ int8_t shared_B[BLOCK_K][BLOCK_N];

    int row = blockIdx.y * BLOCK_M + threadIdx.y;
    int col = blockIdx.x * BLOCK_N + threadIdx.x;

    int32_t accumulator = 0;

    for (int tile_k = 0; tile_k < K; tile_k += BLOCK_K) {

        // Quantize on the fly during the load — no separate quantize pass needed
        if (row < M && (tile_k + threadIdx.x) < K) {
            float val = A_fp32[row * K + tile_k + threadIdx.x] / scale_A;
            shared_A[threadIdx.y][threadIdx.x] =
                static_cast<int8_t>(fmaxf(-127.f, fminf(127.f, rintf(val))));
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0;
        }

        if ((tile_k + threadIdx.y) < K && col < N) {
            float val = B_fp32[(tile_k + threadIdx.y) * N + col] / scale_B;
            shared_B[threadIdx.y][threadIdx.x] =
                static_cast<int8_t>(fmaxf(-127.f, fminf(127.f, rintf(val))));
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        for (int k = 0; k < BLOCK_K; ++k) {
            accumulator += static_cast<int32_t>(shared_A[threadIdx.y][k])
                         * static_cast<int32_t>(shared_B[k][threadIdx.x]);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        // Dequantize the INT32 accumulator back to FP32
        // The combined scale is scale_A * scale_B because:
        //   A_int * B_int = (A_fp/scale_A) * (B_fp/scale_B)
        //   → A_fp * B_fp ≈ (A_int * B_int) * scale_A * scale_B
        C_fp32[row * N + col] = static_cast<float>(accumulator) * scale_A * scale_B;
    }
}
