# Neural network quantization from scratch: an empirical study on RTX 4060

**Author:** [Your Name]
**Date:** [Start Date] – [End Date]
**Hardware:** NVIDIA RTX 4060 Laptop (Ada Lovelace, sm_89, 8GB GDDR6)
**Code:** https://github.com/[your-handle]/quant_study

---

## Abstract

> Write this last. 150–200 words summarizing: what you did, what models you used,
> what your key quantitative findings were, and one sentence on implications.

[TODO — write after Day 49]

---

## 1. Introduction

Neural network quantization reduces model weights and activations from 32-bit
floating-point (FP32) to lower-precision integers (INT8, INT4), enabling faster
inference on hardware with dedicated integer arithmetic units. NVIDIA's Ada Lovelace
architecture, including the RTX 4060 used in this study, features 4th-generation
Tensor Cores capable of INT8 general matrix multiply (GEMM) at significantly higher
throughput than FP16 or FP32 equivalents.

This project investigates quantization from first principles: rather than treating
it as a library call, we implement quantization and dequantization CUDA kernels from
scratch, apply them to a real classification model (ResNet-18 on CIFAR-10), and
systematically measure the accuracy–latency tradeoff across precision levels
(FP32, FP16, INT8, INT4).

Our contributions:
1. Custom CUDA kernels for symmetric and asymmetric INT8 quantization and a
   tiled INT8 GEMM, with detailed profiling against theoretical hardware limits.
2. A per-layer sensitivity analysis identifying which layers drive accuracy loss
   under post-training quantization (PTQ).
3. A direct comparison of our PTQ implementation against NVIDIA TensorRT INT8.
4. Initial results on quantization-aware training (QAT) as an accuracy recovery method.

---

## 2. Background

### 2.1 Floating-point vs integer representation

[Fill in Day 3–4. Cover: FP32 bit layout (1 sign, 8 exponent, 23 mantissa),
why lower precision is lossy, what FP16 and BF16 trade off, and why integer
arithmetic is faster on dedicated hardware.]

### 2.2 Quantization fundamentals

[Fill in Day 7–8. Cover: symmetric vs asymmetric, scale and zero point,
per-tensor vs per-channel, PTQ vs QAT. Reference papers 1 and 3 from README.]

### 2.3 INT8 Tensor Cores on Ada Lovelace

[Fill in Day 5–6. Cover: what a Tensor Core MMA instruction does, throughput
numbers for the RTX 4060 (from NVIDIA's architecture whitepaper), why INT8
is 2–4× faster than FP16 in practice.]

---

## 3. Method

### 3.1 Quantization kernel implementation

[Fill in Days 9–22. Describe the naive kernel, the tiled shared-memory kernel,
the INT8 GEMM, and the fused variant. Include a figure or pseudocode block.]

**Scale factor computation:**
For symmetric INT8: scale = max(|W|) / 127, where W is the weight tensor.
For activations: scale determined by calibration pass over representative data.

**Kernel launch configuration:**
Block size: 256 threads. Grid: ceil(N / 256) blocks.
For GEMM: 2D grid of (ceil(N/16), ceil(M/16)) thread blocks, each of size (16, 16).

### 3.2 Model and dataset

Model: ResNet-18 (11M parameters)
Dataset: CIFAR-10 (10 classes, 50K train / 10K test, 32×32 images)
FP32 baseline trained for: [X] epochs, achieving [Y]% top-1 accuracy.

[Fill in Day 23–25.]

### 3.3 Post-training quantization (PTQ) protocol

1. Load trained FP32 model.
2. For each linear/conv layer, compute scale factor from weight max-abs.
3. For activations, run calibration pass over 1000 training images.
4. Replace each layer's forward pass with our INT8 kernel.
5. Measure accuracy on the full 10K test set.

### 3.4 Per-layer sensitivity analysis

[Fill in Days 29–31. Describe the sweep: quantize one layer at a time,
record accuracy, restore FP32, move to next layer.]

### 3.5 QAT protocol

[Fill in Days 36–38. Describe fake-quant node insertion, number of fine-tuning
epochs, learning rate, dataset used.]

---

## 4. Experiments and Results

### 4.1 Kernel performance

| Kernel                          | N         | Latency (ms) | Throughput (GB/s) | % of peak |
|---------------------------------|-----------|--------------|-------------------|-----------|
| quantize_symmetric (naive)      | 16M elems | TBD          | TBD               | TBD       |
| quantize_symmetric (tiled smem) | 16M elems | TBD          | TBD               | TBD       |
| INT8 GEMM (ours)                | 512×512   | TBD          | TBD               | TBD       |
| cuBLAS FP32 GEMM (reference)    | 512×512   | TBD          | TBD               | TBD       |

*RTX 4060 theoretical peak DRAM bandwidth: ~272 GB/s*

[Fill in Day 12–14 for quantize kernels, Day 15–17 for GEMM.]

### 4.2 PTQ accuracy–latency tradeoff

| Precision  | Top-1 Accuracy | Accuracy drop | Latency (ms/img) | Speedup vs FP32 | Model size |
|------------|----------------|---------------|------------------|-----------------|------------|
| FP32       | TBD            | —             | TBD              | 1.0×            | ~44 MB     |
| FP16       | TBD            | TBD           | TBD              | TBD             | ~22 MB     |
| INT8 PTQ   | TBD            | TBD           | TBD              | TBD             | ~11 MB     |
| INT4 PTQ   | TBD            | TBD           | TBD              | TBD             | ~5.5 MB    |

[Fill in Days 26–33.]

### 4.3 Per-layer sensitivity

[Insert figure: bar chart of accuracy drop per layer, sorted by sensitivity.
Generated by tools/plot_results.py from results/benchmarks/sensitivity.csv]

Key finding: [write this after Day 31]

### 4.4 TensorRT comparison

| Method                 | Accuracy | Latency (ms/img) | Notes                    |
|------------------------|----------|------------------|--------------------------|
| Our INT8 kernels       | TBD      | TBD              |                          |
| TensorRT INT8          | TBD      | TBD              | With INT8 calibration    |
| TensorRT FP16          | TBD      | TBD              |                          |

[Fill in Days 34–35.]

### 4.5 QAT vs PTQ accuracy recovery

| Method    | Accuracy | Drop vs FP32 | Fine-tune epochs |
|-----------|----------|--------------|------------------|
| PTQ INT8  | TBD      | TBD          | 0                |
| QAT INT8  | TBD      | TBD          | TBD              |

[Fill in Days 36–38.]

---

## 5. Discussion

### 5.1 What worked

[Write after Day 44.]

### 5.2 Where accuracy was lost

[Key insight from sensitivity analysis — which layers matter and why.]

### 5.3 Where our kernels fall short of TensorRT

[TensorRT uses cuDNN/cuBLAS under the hood, plus graph optimization and
layer fusion that we don't implement. Document the specific gap and why it exists.]

### 5.4 Implications for deployment

[At what accuracy drop is INT8 acceptable? How does the latency speedup change
the economics of model serving?]

---

## 6. Conclusion

[Write after Day 47. 2–3 paragraphs: what you found, what surprised you,
what you would do next (QAT on larger model, INT4 native Tensor Core support
on H100, extending to LLM weight-only quantization).]

---

## Appendix A — Hardware details

- GPU: NVIDIA RTX 4060 Laptop
- Architecture: Ada Lovelace (sm_89)
- CUDA Toolkit: 12.x
- Driver: [fill in]
- PyTorch: 2.x

## Appendix B — Reproducing results

See README.md → "How to Reproduce Results". All benchmark CSVs are in `results/benchmarks/`.

---

## References

1. NVIDIA. *Integer Quantization for Deep Learning Inference.* arXiv:2004.09602, 2020.
2. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018.
3. Nagel et al. *A White Paper on Neural Network Quantization.* arXiv:2106.08295, 2021.
4. NVIDIA TensorRT Developer Guide. https://docs.nvidia.com/deeplearning/tensorrt/
5. NVIDIA CUDA C++ Programming Guide. https://docs.nvidia.com/cuda/cuda-c-programming-guide/
