#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# build_kernels.sh  —  CMake build script for CUDA kernels
# Run from project root: ./scripts/build_kernels.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"

echo ""
echo "Building CUDA kernels..."
echo "  Project root: $PROJECT_ROOT"
echo "  Build dir:    $BUILD_DIR"
echo "  Architecture: sm_89 (RTX 4060 Ada — change CMAKE_CUDA_ARCHITECTURES if different)"
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89

make -j$(nproc)

echo ""
echo "Build complete. Binaries:"
echo "  ./build/benchmark          — throughput benchmarks"
echo "  ./build/tests/test_quantize — correctness tests"
echo "  ./build/tests/test_gemm    — GEMM tests"
echo ""
