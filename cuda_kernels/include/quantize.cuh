#pragma once

// ═══════════════════════════════════════════════════════════════════════════════
// quantize.cuh  —  Declarations for INT8 quantization/dequantization kernels
// ═══════════════════════════════════════════════════════════════════════════════
//
// LEARNING NOTE (Day 7–8):
// ─────────────────────────
// Quantization maps a floating-point value x_float to an integer x_int using:
//
//   Symmetric (weight-friendly):
//     x_int  = clamp( round(x_float / scale), -127, 127 )
//     x_float ≈ scale * x_int
//     scale  = max(|x_float|) / 127.0f
//
//   Asymmetric (activation-friendly, e.g. after ReLU outputs are all >= 0):
//     x_int   = clamp( round(x_float / scale) + zero_point, 0, 255 )
//     x_float ≈ scale * (x_int - zero_point)
//     scale   = (x_max - x_min) / 255.0f
//     zero_pt = round(-x_min / scale)
//
// Why INT8? On Ada Lovelace (RTX 4060) Tensor Cores, INT8 GEMM throughput is
// roughly 2–4× higher than FP16 and ~8× higher than FP32.
// The tradeoff: ~0–2% accuracy loss with good calibration.
//
// ═══════════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>

// ── Utility: error-checking macro ─────────────────────────────────────────────
//
// Wrap every CUDA runtime call with CUDA_CHECK so errors surface immediately
// with the file name and line number rather than silently corrupting data.
//
// Usage:  CUDA_CHECK( cudaMalloc(&ptr, size) );
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)


// ══════════════════════════════════════════════════════════════════════════════
// Symmetric INT8 quantization
// ══════════════════════════════════════════════════════════════════════════════

/// Quantize an array of FP32 values to INT8 using symmetric per-tensor scaling.
///
/// Each thread handles one element: output[i] = clamp(round(input[i] / scale), -127, 127)
///
/// @param input     Device pointer to FP32 source array
/// @param output    Device pointer to INT8 destination array
/// @param n         Number of elements
/// @param scale     Scale factor: max(|input|) / 127.0f  (computed on CPU before launch)
__global__ void quantize_symmetric_int8(
    const float*   input,
    int8_t*        output,
    int            n,
    float          scale
);


/// Dequantize INT8 → FP32 using symmetric scale.
///
/// @param input     Device pointer to INT8 source
/// @param output    Device pointer to FP32 destination
/// @param n         Number of elements
/// @param scale     Same scale used during quantization
__global__ void dequantize_symmetric_int8(
    const int8_t*  input,
    float*         output,
    int            n,
    float          scale
);


// ══════════════════════════════════════════════════════════════════════════════
// Asymmetric UINT8 quantization
// ══════════════════════════════════════════════════════════════════════════════

/// Quantize FP32 → UINT8 using asymmetric (min/max) scaling.
/// Better for activations that are non-negative (post-ReLU, post-softmax).
///
/// @param zero_point  Integer offset: round(-x_min / scale)
__global__ void quantize_asymmetric_uint8(
    const float*   input,
    uint8_t*       output,
    int            n,
    float          scale,
    int            zero_point
);

__global__ void dequantize_asymmetric_uint8(
    const uint8_t* input,
    float*         output,
    int            n,
    float          scale,
    int            zero_point
);


// ══════════════════════════════════════════════════════════════════════════════
// Tiled version (Phase 2, Day 12–14) — uses shared memory for better throughput
// ══════════════════════════════════════════════════════════════════════════════
//
// LEARNING NOTE (Day 12):
// ──────────────────────
// The naive kernel above reads one element per thread from global memory.
// Global memory latency is ~300–600 cycles on Ada. Shared memory latency is ~5 cycles.
//
// Tiling: each thread block loads TILE_SIZE elements into shared memory together,
// then each thread reads from smem. For a pure element-wise kernel like quantize,
// the bandwidth speedup comes from coalesced loads — all threads in a warp access
// consecutive addresses, saturating the L2/DRAM bus.
//
// The real payoff of shared memory tiling shows up in reduction kernels (e.g.
// computing max(|input|) for scale calibration) where threads need to communicate.
//
// Tile size: 256 is a safe default. 512 can improve occupancy on some kernels.
// Profile with Nsight Compute → "Shared Memory" section to verify utilization.

#define TILE_SIZE 256

/// Tiled quantize: loads TILE_SIZE elements into shared memory per block,
/// then quantizes. Compare throughput vs quantize_symmetric_int8 in benchmark.cu.
__global__ void quantize_symmetric_int8_tiled(
    const float*   input,
    int8_t*        output,
    int            n,
    float          scale
);


// ══════════════════════════════════════════════════════════════════════════════
// Host-side helper: compute symmetric scale factor on CPU
// ══════════════════════════════════════════════════════════════════════════════

/// Computes scale = max(|data|) / 127.0f over a host-side float array.
/// Call this before launching the quantize kernel.
float compute_symmetric_scale(const float* host_data, int n);


// ══════════════════════════════════════════════════════════════════════════════
// INT4 packing helpers (Phase 3, Days 32–33)
// ══════════════════════════════════════════════════════════════════════════════
//
// LEARNING NOTE (Day 32):
// ──────────────────────
// INT4 stores two 4-bit values per byte. There is no native int4_t in C++.
// We pack two INT4 values into a single uint8_t:
//   packed = (lo & 0x0F) | ((hi & 0x0F) << 4)
//   lo = packed & 0x0F       (lower nibble)
//   hi = (packed >> 4) & 0x0F (upper nibble)
//
// This halves memory footprint vs INT8 at the cost of slightly more complex
// kernel logic. Ada Tensor Cores don't natively support INT4 MMA (that's H100+),
// so on the 4060 we simulate INT4 by unpacking to INT8 before the GEMM.
// The benefit is purely memory bandwidth: a 2× smaller weight tensor fits
// better in L2 cache and takes half the time to load from DRAM.

__global__ void pack_int4(
    const int8_t*  input,      // INT8 array where values are in [-7, 7]
    uint8_t*       output,     // Packed INT4: output[i] holds input[2i] and input[2i+1]
    int            n           // Number of INT8 elements (must be even)
);

__global__ void unpack_int4(
    const uint8_t* input,
    int8_t*        output,
    int            n           // Number of INT8 elements to unpack into
);
