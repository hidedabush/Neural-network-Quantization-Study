#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// gemm_int8.cuh  —  Declarations for INT8 General Matrix Multiply kernel
// ═══════════════════════════════════════════════════════════════════════════════
//
// LEARNING NOTE (Days 15–17):
// ─────────────────────────────
// GEMM computes: C = A * B  where A is [M×K] and B is [K×N], result C is [M×N].
// Every linear (fully-connected) layer in a neural network IS a GEMM:
//   output [batch × out_features] = input [batch × in_features] * weight^T [in × out]
// Every convolution can be expressed as a GEMM via im2col.
// So if you want to speed up neural network inference, GEMM is the target.
//
// INT8 GEMM on Ada Tensor Cores:
//   - Each Tensor Core MMA instruction: D = A*B + C where A,B are INT8 matrices
//     and the accumulator D,C are INT32 (to avoid overflow from accumulated products)
//   - Ada has 4th-gen Tensor Cores supporting INT8 at ~2× the throughput of FP16
//   - Our implementation: a tiled GEMM using shared memory for A and B tiles,
//     accumulating in INT32. This is the core of what TensorRT's INT8 path does.
//
// For a production GEMM you'd use cuBLAS (cublasGemmEx with CUDA_R_8I).
// We write our own to understand the algorithm — then benchmark both.
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>

// Tile dimensions for shared memory tiling.
// Each thread block computes a BLOCK_M × BLOCK_N tile of C.
// Each tile loads BLOCK_K columns of A and rows of B at a time.
//
// Tuning guide (Day 30 — occupancy):
//   Shared memory per block = BLOCK_M * BLOCK_K + BLOCK_K * BLOCK_N (in bytes for INT8)
//   Ada SM has 100KB shared mem. With BLOCK_M=BLOCK_N=BLOCK_K=16:
//   smem = 16*16 + 16*16 = 512 bytes per block — very small, many blocks can coexist.
//   Try increasing to 32 and check Nsight for occupancy change.
#define BLOCK_M 16
#define BLOCK_N 16
#define BLOCK_K 16


/// INT8 GEMM: C_int32 = A_int8 * B_int8  (accumulates in INT32 to avoid overflow)
///
/// @param A      Device pointer, INT8, shape [M × K], row-major
/// @param B      Device pointer, INT8, shape [K × N], row-major
/// @param C      Device pointer, INT32, shape [M × N], row-major (output)
/// @param M      Rows of A / rows of C
/// @param N      Cols of B / cols of C
/// @param K      Cols of A / rows of B (inner dimension)
///
/// Launch config: <<< dim3(ceil(N/BLOCK_N), ceil(M/BLOCK_M)), dim3(BLOCK_N, BLOCK_M) >>>
__global__ void gemm_int8(
    const int8_t*  A,
    const int8_t*  B,
    int32_t*       C,
    int            M,
    int            N,
    int            K
);


/// Fused: quantize A and B, run INT8 GEMM, dequantize C back to FP32.
/// This is a single-kernel version of the full quantize → compute → dequantize pipeline.
///
/// @param scale_A   Scale factor for A (from compute_symmetric_scale)
/// @param scale_B   Scale factor for B
/// @param C_fp32    FP32 output: C_fp32[i,j] ≈ (sum_k A[i,k] * B[k,j]) * scale_A * scale_B
__global__ void gemm_int8_fused(
    const float*   A_fp32,
    const float*   B_fp32,
    float*         C_fp32,
    int            M,
    int            N,
    int            K,
    float          scale_A,
    float          scale_B
);
