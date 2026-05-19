// ═══════════════════════════════════════════════════════════════════════════════
// quantize.cu  —  INT8/UINT8 quantization kernel implementations
//
// BUILD:   See CMakeLists.txt — compiled as part of quant_kernels static lib.
// PROFILE: nsys profile --stats=true ./benchmark
//          ncu --set full ./benchmark
//
// READING ORDER FOR LEARNING:
//   1. quantize_symmetric_int8        ← Start here. Simplest possible kernel.
//   2. dequantize_symmetric_int8      ← Exact inverse.
//   3. quantize_asymmetric_uint8      ← Adds zero_point complexity.
//   4. quantize_symmetric_int8_tiled  ← Shared memory version. Compare in profiler.
//   5. pack_int4 / unpack_int4        ← Read after Day 32 (INT4 experiments).
// ═══════════════════════════════════════════════════════════════════════════════

#include "quantize.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 1: quantize_symmetric_int8  (naive — one element per thread)
// ══════════════════════════════════════════════════════════════════════════════
//
// THREAD → DATA MAPPING:
//   - Each thread has a unique global index: idx = blockIdx.x * blockDim.x + threadIdx.x
//   - Thread idx processes element input[idx]
//   - Grid is 1D: launch with <<< ceil(n/256), 256 >>>
//
// MEMORY ACCESS PATTERN:
//   - input[idx]: consecutive threads read consecutive addresses → coalesced read
//   - output[idx]: consecutive threads write consecutive addresses → coalesced write
//   Coalesced = all 32 threads in a warp access a 128-byte aligned contiguous block.
//   This is the single biggest factor in global memory throughput. Non-coalesced
//   access (e.g., stride-2 reads) can drop bandwidth by 8–16× on Ada.
//
// REGISTER USAGE (check with --ptxas-options=-v):
//   Expected: ~8 registers per thread. At 256 threads/block that's 2048 regs/block.
//   Ada SM has 65536 registers, so this can run ~32 blocks simultaneously per SM.
//   That's excellent occupancy. Nsight Compute → "Launch Statistics" will confirm.
__global__ void quantize_symmetric_int8(
    const float* input,
    int8_t*      output,
    int          n,
    float        scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: the last block may have threads that exceed array bounds.
    // Without this, those threads would read/write garbage memory.
    if (idx >= n) return;

    // Core quantization math:
    //   1. Divide by scale to normalize into [-127, 127]
    //   2. Round to nearest integer (rintf is hardware-accelerated on GPU)
    //   3. Clamp to [-127, 127] — we use -127 not -128 to keep the range symmetric
    //      (avoids an edge case in dequantization where -128 * scale != 127 * scale)
    float val   = input[idx] / scale;
    float rounded = rintf(val);                       // GPU intrinsic, ~1 cycle
    float clamped = fmaxf(-127.0f, fminf(127.0f, rounded));
    output[idx] = static_cast<int8_t>(clamped);
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 2: dequantize_symmetric_int8
// ══════════════════════════════════════════════════════════════════════════════
//
// Simply reverses the quantize step. Note the output is approximate:
//   dequant(quant(x)) ≈ x, not exactly x, because we lost precision.
//
// ACCURACY NOTE (relevant for Day 22 accuracy audit):
//   The quantization error for any single value is at most ±0.5 * scale.
//   For a layer with scale ≈ 0.01 (typical for weights), max error ≈ ±0.005.
//   Whether that matters depends on how many layers accumulate the error.
__global__ void dequantize_symmetric_int8(
    const int8_t* input,
    float*        output,
    int           n,
    float         scale)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = static_cast<float>(input[idx]) * scale;
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 3: quantize_asymmetric_uint8
// ══════════════════════════════════════════════════════════════════════════════
//
// WHEN TO USE ASYMMETRIC vs SYMMETRIC:
//   Symmetric: good for weights. Weight distributions are usually bell-curved
//   around zero, so [-127, 127] covers them well.
//
//   Asymmetric: good for activations. After ReLU, all values are >= 0.
//   Symmetric INT8 would waste half its range on negative numbers that never occur.
//   Asymmetric shifts the range to [0, 255] (UINT8), covering [x_min, x_max] efficiently.
//
// ZERO POINT MATH:
//   scale     = (x_max - x_min) / 255.0
//   zero_pt   = round(-x_min / scale)          range: [0, 255]
//   x_uint8   = clamp(round(x / scale) + zero_pt, 0, 255)
//   x_float   ≈ scale * (x_uint8 - zero_pt)
__global__ void quantize_asymmetric_uint8(
    const float* input,
    uint8_t*     output,
    int          n,
    float        scale,
    int          zero_point)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val     = input[idx] / scale + static_cast<float>(zero_point);
    float rounded = rintf(val);
    float clamped = fmaxf(0.0f, fminf(255.0f, rounded));
    output[idx]   = static_cast<uint8_t>(clamped);
}

__global__ void dequantize_asymmetric_uint8(
    const uint8_t* input,
    float*         output,
    int            n,
    float          scale,
    int            zero_point)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = scale * (static_cast<float>(input[idx]) - static_cast<float>(zero_point));
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 4: quantize_symmetric_int8_tiled  (shared memory version)
// ══════════════════════════════════════════════════════════════════════════════
//
// SHARED MEMORY REVIEW:
//   Shared memory (__shared__) is on-chip SRAM local to each SM.
//   - Latency: ~5 cycles (vs ~200-600 for L2/DRAM)
//   - Size: 100KB per SM on Ada (configurable between L1 cache and shared mem)
//   - Lifetime: per-block (data doesn't persist across block boundaries)
//   - Access: all threads in a block can read/write the same shared array
//
// FOR THIS KERNEL the tiling pattern is:
//   1. Each block loads TILE_SIZE elements from global → shared memory (coalesced)
//   2. __syncthreads() ensures all loads complete before any thread proceeds
//   3. Each thread reads its element from shared memory and quantizes
//   4. Each thread writes result back to global memory (coalesced)
//
// WHY IS THIS FASTER HERE? (if at all):
//   For a pure element-wise kernel with no data reuse, tiling into shared memory
//   doesn't reduce global memory traffic (each element is still read once, written once).
//   The benefit here is subtler: by loading as a block, we can guarantee alignment
//   and coalescing even if the input pointer isn't 128-byte aligned.
//
//   The REAL speedup from shared memory comes in kernels with data reuse:
//   - Reductions (max, sum): threads read the same element multiple times
//   - GEMM tiling: each element is used by an entire row/column of threads
//   - Stencil ops: neighboring threads access overlapping regions
//
//   Benchmark both kernels in benchmark.cu and measure throughput (GB/s).
//   If tiled ≈ naive here, that confirms the theory. Document this finding.
__global__ void quantize_symmetric_int8_tiled(
    const float* input,
    int8_t*      output,
    int          n,
    float        scale)
{
    // Shared memory tile — TILE_SIZE floats per block
    __shared__ float tile[TILE_SIZE];

    int tid  = threadIdx.x;
    int gid  = blockIdx.x * blockDim.x + tid;  // global element index

    // Step 1: Load from global → shared memory
    // All threads in the block participate in loading, making this a coalesced read.
    if (gid < n) {
        tile[tid] = input[gid];
    } else {
        tile[tid] = 0.0f;  // pad out-of-bounds with zero (safe; will be discarded)
    }

    // Step 2: Barrier — wait until ALL threads in this block have finished loading.
    // Without __syncthreads() here, some threads might start quantizing before
    // other threads have written their values to shared memory → data race.
    __syncthreads();

    // Step 3: Quantize from shared memory → global output
    if (gid < n) {
        float val     = tile[tid] / scale;
        float rounded = rintf(val);
        float clamped = fmaxf(-127.0f, fminf(127.0f, rounded));
        output[gid]   = static_cast<int8_t>(clamped);
    }
}


// ══════════════════════════════════════════════════════════════════════════════
// Host helper: compute_symmetric_scale
// ══════════════════════════════════════════════════════════════════════════════
//
// This runs on the CPU before launching the quantize kernel.
// In a production system (TensorRT) this would run a "calibration" pass over
// a representative dataset to find the true max activation, not just weight max.
// For weights (which are fixed after training) this simple max-abs is sufficient.
float compute_symmetric_scale(const float* host_data, int n)
{
    float max_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float abs_val = fabsf(host_data[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    // Avoid division by zero for all-zero tensors (e.g. bias in early training)
    if (max_abs == 0.0f) return 1.0f;
    return max_abs / 127.0f;
}


// ══════════════════════════════════════════════════════════════════════════════
// Kernel 5 & 6: INT4 pack / unpack  (implement Day 32)
// ══════════════════════════════════════════════════════════════════════════════
__global__ void pack_int4(
    const int8_t* input,
    uint8_t*      output,
    int           n)
{
    // Each thread packs two consecutive INT8 values into one byte.
    // n must be even; output has n/2 elements.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    // Clamp to [-7, 7] before packing (INT4 range with sign bit)
    int8_t lo = static_cast<int8_t>(fmaxf(-7.0f, fminf(7.0f, input[2 * idx])));
    int8_t hi = static_cast<int8_t>(fmaxf(-7.0f, fminf(7.0f, input[2 * idx + 1])));

    // Pack: lower nibble = lo, upper nibble = hi
    output[idx] = (static_cast<uint8_t>(lo) & 0x0F) |
                  ((static_cast<uint8_t>(hi) & 0x0F) << 4);
}

__global__ void unpack_int4(
    const uint8_t* input,
    int8_t*        output,
    int            n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n / 2) return;

    uint8_t packed = input[idx];

    // Extract lower nibble and sign-extend from 4-bit to 8-bit
    int8_t lo = static_cast<int8_t>(packed & 0x0F);
    if (lo > 7) lo -= 16;  // two's complement sign extension for 4-bit

    int8_t hi = static_cast<int8_t>((packed >> 4) & 0x0F);
    if (hi > 7) hi -= 16;

    output[2 * idx]     = lo;
    output[2 * idx + 1] = hi;
}
