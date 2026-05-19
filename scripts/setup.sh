#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# setup.sh  —  One-command environment setup for quant_study
# Run once after cloning: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e   # exit on first error

echo ""
echo "══════════════════════════════════════════════"
echo "  quant_study environment setup"
echo "══════════════════════════════════════════════"

# ── Step 1: Verify CUDA ────────────────────────────────────────────────────────
echo ""
echo "[1/5] Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found. Install CUDA Toolkit 12.x from:"
    echo "  https://developer.nvidia.com/cuda-downloads"
    exit 1
fi
nvcc --version
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader

# ── Step 2: Python venv ────────────────────────────────────────────────────────
echo ""
echo "[2/5] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# ── Step 3: PyTorch (CUDA 12.1 build) ─────────────────────────────────────────
echo ""
echo "[3/5] Installing PyTorch with CUDA 12.1 support..."
pip install --upgrade pip --quiet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

# Verify GPU visible from PyTorch
python3 -c "
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU!'
print(f'  PyTorch {torch.__version__} — GPU: {torch.cuda.get_device_name(0)}')
print(f'  CUDA version: {torch.version.cuda}')
"

# ── Step 4: Python dependencies ───────────────────────────────────────────────
echo ""
echo "[4/5] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# ── Step 5: Build CUDA kernels ────────────────────────────────────────────────
echo ""
echo "[5/5] Building CUDA kernels..."
./scripts/build_kernels.sh

echo ""
echo "══════════════════════════════════════════════"
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Activate the venv:  source venv/bin/activate"
echo "  2. Run tests:          ./build/tests/test_quantize"
echo "  3. Run benchmark:      ./build/benchmark"
echo "  4. Start Day 1:        docs/daily_log/day_001.md"
echo "══════════════════════════════════════════════"
echo ""
