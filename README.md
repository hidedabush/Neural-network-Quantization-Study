# Neural Network Quantization: A From-Scratch Study

> A 50-day research project implementing INT8/INT4 quantization with custom CUDA kernels,
> measuring accuracy/speed tradeoffs on real models, and producing a paper-style report.
> Targeting NVIDIA's TensorRT and inference teams.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Hardware & Software Requirements](#hardware--software-requirements)
- [Repository Structure](#repository-structure)
- [Setup Guide](#setup-guide)
- [How to Reproduce Results](#how-to-reproduce-results)
- [Project Phases](#project-phases)
- [Key Concepts Glossary](#key-concepts-glossary)
- [Daily Learning Journal](#daily-learning-journal)
- [Results Summary](#results-summary)
- [References](#references)

---

## Project Overview

**Research question:** How much accuracy do we lose when quantizing a neural network from
FP32 to INT8 or INT4, and how does a custom CUDA kernel implementation compare to
NVIDIA's TensorRT in latency and throughput?

**What this project builds:**
- Custom CUDA C++ kernels for INT8 quantization, dequantization, and GEMM
- A Python framework to apply those kernels to real models (ResNet-18, DistilBERT)
- A systematic benchmark: FP32 → FP16 → INT8 → INT4 across accuracy, latency, and model size
- Per-layer sensitivity analysis (which layers tolerate quantization, which don't)
- A comparison against NVIDIA TensorRT INT8
- A paper-style technical report summarizing findings

**Why this matters for NVIDIA:**
Quantization is the core technology behind TensorRT, NVIDIA's production inference engine.
The INT8 Tensor Cores on Ada Lovelace GPUs (RTX 4060) are only useful if the software
stack can correctly quantize and route computation through them. Understanding quantization
at the kernel level — including scale factors, zero points, and accuracy tradeoffs — is
exactly the domain knowledge NVIDIA's inference team works with daily.

---

## Hardware & Software Requirements

### Hardware
| Component     | Spec                                      |
|---------------|-------------------------------------------|
| GPU           | NVIDIA RTX 4060 Laptop (Ada Lovelace)     |
| VRAM          | 8 GB GDDR6                                |
| Architecture  | Ada Lovelace (sm_89)                      |
| INT8 support  | Yes — 4th-gen Tensor Cores                |
| Driver        | >= 525.xx recommended                     |

### Software
| Tool                  | Version       | Purpose                              |
|-----------------------|---------------|--------------------------------------|
| CUDA Toolkit          | 12.x          | Kernel compilation, Nsight tools     |
| Python                | 3.10+         | Experiments, data loading            |
| PyTorch               | 2.x + cu121   | Model baseline, QAT                  |
| TensorRT              | 8.6+          | Comparison baseline                  |
| CMake                 | 3.20+         | Build system for CUDA kernels        |
| Nsight Systems        | bundled       | Timeline profiling                   |
| Nsight Compute        | bundled       | Kernel-level profiling               |
| torchvision           | latest        | Datasets (CIFAR-10, ImageNet subset) |
| matplotlib / pandas   | latest        | Result plotting and tables           |

---

## Repository Structure

```
quant_study/
│
├── README.md                   ← You are here
├── CMakeLists.txt              ← Top-level CMake build
├── requirements.txt            ← Python dependencies
│
├── cuda_kernels/               ← All CUDA C++ source
│   ├── include/
│   │   ├── quantize.cuh        ← Kernel declarations + shared utilities
│   │   └── gemm_int8.cuh       ← INT8 GEMM declarations
│   ├── src/
│   │   ├── quantize.cu         ← Quantize / dequantize kernels
│   │   ├── gemm_int8.cu        ← INT8 GEMM kernel
│   │   └── benchmark.cu        ← Standalone bandwidth + throughput benchmarks
│   └── tests/
│       ├── test_quantize.cu    ← Correctness tests vs CPU reference
│       └── test_gemm.cu        ← GEMM output validation
│
├── experiments/
│   ├── ptq/                    ← Post-training quantization scripts
│   │   ├── run_ptq.py          ← Apply INT8 PTQ to a model
│   │   └── calibrate.py        ← Scale factor calibration pass
│   ├── qat/                    ← Quantization-aware training
│   │   └── run_qat.py          ← Fine-tune with fake-quant nodes
│   ├── sensitivity/            ← Per-layer sensitivity study
│   │   └── layer_sweep.py      ← Quantize one layer at a time
│   └── mixed_precision/
│       └── assign_precision.py ← Auto-assign INT8/FP16 per layer
│
├── models/
│   └── model_loader.py         ← Load ResNet-18 / DistilBERT with consistent API
│
├── datasets/
│   └── dataset_loader.py       ← CIFAR-10 / ImageNet-100 loaders
│
├── results/
│   ├── benchmarks/             ← CSV files: accuracy, latency, model size
│   ├── figures/                ← Charts generated from benchmark CSVs
│   └── nsight_profiles/        ← .ncu-rep and .nsys-rep files (Nsight exports)
│
├── scripts/
│   ├── setup.sh                ← One-command environment setup
│   ├── build_kernels.sh        ← CMake build script
│   └── run_all_benchmarks.sh   ← Reproduce all results end-to-end
│
├── docs/
│   ├── daily_log/              ← Markdown journal: one file per day
│   │   └── day_001.md          ← Template (copy for each day)
│   └── paper/
│       └── report_draft.md     ← Growing paper draft
│
└── tools/
    └── plot_results.py         ← Generate all figures from results/benchmarks/
```

---

## Setup Guide

### Step 1 — Verify your GPU and CUDA driver

```bash
nvidia-smi
# Should show: RTX 4060, Driver >= 525, CUDA >= 12.x
nvcc --version
# Should show: release 12.x
```

### Step 2 — Clone and enter the project

```bash
git clone <your-repo-url> quant_study
cd quant_study
```

### Step 3 — Python environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Step 4 — Build the CUDA kernels

```bash
chmod +x scripts/build_kernels.sh
./scripts/build_kernels.sh

# Manual equivalent:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..
```

### Step 5 — Verify everything works

```bash
# Run the kernel correctness tests
./build/tests/test_quantize
./build/tests/test_gemm

# Run the Python baseline
python experiments/ptq/run_ptq.py --model resnet18 --dry-run
```

If the tests pass and the dry-run completes, your environment is ready.

---

## How to Reproduce Results

All results in `results/benchmarks/` can be reproduced by running:

```bash
./scripts/run_all_benchmarks.sh
```

This runs in order:
1. FP32 baseline accuracy + latency
2. INT8 PTQ with calibration
3. INT4 experiment on robust layers
4. Per-layer sensitivity sweep
5. TensorRT INT8 comparison
6. Throughput scaling curve (batch size 1 → 256)

Results are written as CSVs to `results/benchmarks/`.
Figures are regenerated by:

```bash
python tools/plot_results.py
```

---

## Project Phases

| Phase | Days  | Focus                              | Key output                        |
|-------|-------|------------------------------------|-----------------------------------|
| 1     | 1–8   | GPU + ML foundations               | Precision benchmark, env setup    |
| 2     | 9–22  | Custom CUDA quantization kernels   | INT8 kernels, Nsight profiles     |
| 3     | 23–35 | Quantization study on real model   | Benchmark table, sensitivity plot |
| 4     | 36–44 | QAT, mixed precision, sparsity     | QAT vs PTQ comparison             |
| 5     | 45–50 | Paper + portfolio polish           | PDF report, clean repo            |

See `docs/paper/report_draft.md` for the growing technical report.
See `docs/daily_log/` for day-by-day notes and observations.

---

## Key Concepts Glossary

This section grows as the project progresses. New terms get added on the day they're first encountered.

**Quantization** — the process of representing model weights and/or activations using
lower-precision integers (e.g. INT8) instead of floating-point (FP32). Reduces model
size and enables faster integer arithmetic on hardware that supports it.

**Scale factor (S)** — a floating-point multiplier that maps the integer range back to
the original floating-point range. For symmetric INT8: `x_float ≈ S × x_int8`.

**Zero point (Z)** — an integer offset used in asymmetric quantization to shift the
integer range so it covers non-symmetric distributions. `x_float ≈ S × (x_int8 - Z)`.

**PTQ (Post-Training Quantization)** — quantize a model after training is complete,
using a small calibration dataset to determine good scale factors. Fast, no retraining.

**QAT (Quantization-Aware Training)** — simulate quantization during training by inserting
"fake quantize" nodes that round weights/activations to integer precision in the forward
pass but pass gradients through in FP32. Recovers accuracy lost by PTQ.

**INT8 Tensor Cores** — dedicated hardware units on NVIDIA GPUs (Volta and later) that
perform matrix multiply-accumulate (MMA) operations in INT8 at very high throughput.
The RTX 4060 has 4th-generation Tensor Cores supporting INT8.

**GEMM** — General Matrix Multiply. The core operation in every linear layer and
convolution. Most quantization work targets GEMM since it dominates inference time.

**Calibration** — a pass over a small representative dataset used to determine the
range (min/max) of activations, from which scale factors are derived. Only needed for
activation quantization (weight ranges are fixed after training).

**Symmetric quantization** — maps the float range [-max, +max] symmetrically to
[-127, 127] (INT8). Zero point is always 0. Simpler, slightly less accurate.

**Asymmetric quantization** — maps [float_min, float_max] to [0, 255] (UINT8).
Uses a non-zero zero point. Better for activations like ReLU outputs (always positive).

**Per-tensor vs per-channel quantization** — per-tensor uses one scale for the whole
weight matrix; per-channel uses one scale per output channel. Per-channel is more accurate
but requires storing N scale factors. TensorRT uses per-channel by default.

**Sensitivity** — how much accuracy a layer loses when quantized. Measured by quantizing
one layer at a time and recording the accuracy drop. High-sensitivity layers are kept in
FP16 in mixed-precision strategies.

---

## Daily Learning Journal

Each day's notes live in `docs/daily_log/day_NNN.md`.

The template (see `docs/daily_log/day_001.md`) captures:
- What was planned vs what actually happened
- Specific things learned (with code snippets)
- Profiler observations (with numbers)
- Surprises or unexpected results
- Open questions to investigate later

This journal is the raw material for the paper's Experiments section and the primary
resource for NVIDIA interview conversations.

---

## Results Summary

> This section is updated as experiments complete. Initially empty.

| Experiment              | Model      | Accuracy (FP32) | Accuracy (INT8) | Latency speedup | Size reduction |
|-------------------------|------------|-----------------|-----------------|-----------------|----------------|
| PTQ INT8                | ResNet-18  | TBD             | TBD             | TBD             | TBD            |
| PTQ INT4 (robust layers)| ResNet-18  | TBD             | TBD             | TBD             | TBD            |
| QAT INT8                | ResNet-18  | TBD             | TBD             | TBD             | TBD            |
| TensorRT INT8           | ResNet-18  | TBD             | TBD             | TBD             | TBD            |

---

## References

Papers and resources referenced throughout the project:

1. NVIDIA. *Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation.* arXiv:2004.09602 (2020).
2. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
3. Nagel et al. *A White Paper on Neural Network Quantization.* arXiv:2106.08295 (2021).
4. NVIDIA TensorRT Developer Guide. https://docs.nvidia.com/deeplearning/tensorrt/
5. NVIDIA CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
6. NVIDIA Ampere Architecture Whitepaper. https://www.nvidia.com/content/dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V2.pdf

> Papers 1 and 3 are the most directly relevant — read them during Days 7–8.
