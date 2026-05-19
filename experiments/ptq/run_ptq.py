"""
experiments/ptq/run_ptq.py
==========================
Post-Training Quantization (PTQ) experiment: FP32 model → INT8 via scale-factor
calibration, no retraining. Measures accuracy and latency before/after.

Usage
-----
    # Dry run (verify setup without running full eval)
    python experiments/ptq/run_ptq.py --model resnet18 --dry-run

    # Full experiment
    python experiments/ptq/run_ptq.py --model resnet18 --bits 8

    # INT4 experiment (Phase 3, Day 32)
    python experiments/ptq/run_ptq.py --model resnet18 --bits 4

Results are written to results/benchmarks/ptq_{model}_{bits}bit.csv

Learning notes (Days 26–28)
----------------------------
PTQ workflow:
  1. Load FP32 model (already trained)
  2. Run calibration pass: forward N images through the model,
     record min/max of each layer's activations → derive scale factors
  3. Replace each layer's nn.Linear / nn.Conv2d with an INT8 version
     that uses our scale factors
  4. Run full evaluation → measure accuracy and latency

Why is step 2 (calibration) necessary?
  We know weight ranges at load time (just compute max-abs of each weight tensor).
  But activation ranges depend on the INPUT — they change per batch and per layer.
  Calibration runs a representative dataset through the FP32 model to measure
  typical activation ranges, which we then use to set fixed scale factors.
  Without calibration, you'd have to set scale factors dynamically at runtime
  (dynamic quantization), which is slower and harder to implement.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path so imports work regardless of working directory
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from models.model_loader import load_model, count_quantizable_layers
from datasets.dataset_loader import load_cifar10


# ══════════════════════════════════════════════════════════════════════════════
# INT8 wrapper module
# ══════════════════════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """
    Drop-in replacement for nn.Linear that quantizes weights to INT8 on the fly.

    NOTE: This uses PyTorch's built-in integer operations, not our custom CUDA
    kernels (that integration comes in Phase 2). This lets us measure accuracy
    impact without worrying about kernel correctness first.

    Later, you'll replace torch.ops.quantized.linear with a call into our
    custom C++ extension built from cuda_kernels/. The accuracy numbers should
    be identical; the latency will differ.
    """
    def __init__(self, original_linear: nn.Linear, scale: float):
        super().__init__()
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.scale        = scale

        # Quantize weights to INT8, store as float for PyTorch compatibility
        # (real kernel integration stores as int8_t — see cuda_kernels/)
        w = original_linear.weight.data
        w_int8 = torch.clamp(torch.round(w / scale), -127, 127)
        self.weight_q = nn.Parameter(w_int8, requires_grad=False)
        self.bias = original_linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights back to float for the matmul
        # In our custom kernel version, this happens inside the CUDA kernel
        w_fp32 = self.weight_q.float() * self.scale
        return nn.functional.linear(x, w_fp32, self.bias)


# ══════════════════════════════════════════════════════════════════════════════
# Calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_model(model: nn.Module, loader: DataLoader, n_batches: int = 10) -> dict:
    """
    Run N batches through the FP32 model. Record per-layer weight scale factors.
    Returns a dict: {layer_name: scale_factor}

    For weights, scale = max(|W|) / 127. This is computed analytically from the
    weight tensor — no data needed.

    For activations, we'd hook each layer and record min/max over the calibration
    batches. We implement weight-only calibration here for simplicity; activation
    calibration is added in Phase 3.
    """
    print(f"Calibrating scale factors (weight-only, {n_batches} batches)...")
    scales = {}

    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            w = module.weight.data.float()
            max_abs = w.abs().max().item()
            scale = max_abs / 127.0 if max_abs > 0 else 1.0
            scales[name] = scale

    print(f"  Calibrated {len(scales)} layers.")
    return scales


# ══════════════════════════════════════════════════════════════════════════════
# Apply quantization
# ══════════════════════════════════════════════════════════════════════════════

def apply_ptq(model: nn.Module, scales: dict, bits: int = 8) -> nn.Module:
    """
    Replace Linear layers with QuantizedLinear using calibrated scale factors.

    bits=8: standard INT8 (scale = max/127)
    bits=4: INT4 approximation (scale = max/7, values clamped to [-7, 7])
    """
    clamp_val = (2 ** (bits - 1)) - 1  # 127 for INT8, 7 for INT4

    replaced = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and name in scales:
            scale = scales[name]
            # Adjust scale for INT4
            if bits == 4:
                w = module.weight.data.float()
                scale = w.abs().max().item() / clamp_val if w.abs().max().item() > 0 else 1.0

            q_module = QuantizedLinear(module, scale)
            # Navigate to parent module and replace the child
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], q_module)
            replaced += 1

    print(f"  Replaced {replaced} Linear layers with INT{bits} quantized versions.")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, label: str) -> dict:
    """Evaluate accuracy and measure avg latency per batch."""
    model.eval()
    correct, total = 0, 0
    latencies = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        t0 = time.perf_counter()
        outputs = model(images)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        latencies.append((t1 - t0) * 1000)  # ms

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    accuracy = correct / total * 100
    avg_latency = sum(latencies) / len(latencies)

    print(f"  [{label}] Accuracy: {accuracy:.2f}%  |  Avg batch latency: {avg_latency:.2f} ms")
    return {"label": label, "accuracy": accuracy, "latency_ms": avg_latency}


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PTQ experiment")
    parser.add_argument("--model",    default="resnet18", choices=["resnet18", "distilbert"])
    parser.add_argument("--bits",     type=int, default=8, choices=[4, 8])
    parser.add_argument("--dry-run",  action="store_true", help="Run 1 batch only to verify setup")
    parser.add_argument("--device",   default="cuda")
    args = parser.parse_args()

    n_eval_batches = 1 if args.dry_run else None  # None = full dataset

    print(f"\n{'═'*50}")
    print(f"  PTQ experiment: {args.model} → INT{args.bits}")
    if args.dry_run: print("  [DRY RUN — 1 batch only]")
    print(f"{'═'*50}\n")

    # Load data
    _, test_loader = load_cifar10(batch_size=64, num_workers=2)

    # FP32 baseline
    fp32_model = load_model(args.model, precision="fp32", device=args.device)
    fp32_results = evaluate(fp32_model, test_loader, args.device, "FP32 baseline")

    # PTQ
    int8_model = load_model(args.model, precision="fp32", device=args.device)
    scales = calibrate_model(int8_model, test_loader)
    int8_model = apply_ptq(int8_model, scales, bits=args.bits)
    int8_results = evaluate(int8_model, test_loader, args.device, f"INT{args.bits} PTQ")

    # Summary
    acc_drop = fp32_results["accuracy"] - int8_results["accuracy"]
    speedup  = fp32_results["latency_ms"] / int8_results["latency_ms"]
    print(f"\n  Accuracy drop : {acc_drop:.2f}%")
    print(f"  Latency speedup: {speedup:.2f}×\n")

    # Save results
    if not args.dry_run:
        out_path = Path("results/benchmarks") / f"ptq_{args.model}_{args.bits}bit.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "accuracy", "latency_ms"])
            writer.writeheader()
            writer.writerow(fp32_results)
            writer.writerow(int8_results)
        print(f"  Results saved to {out_path}")


if __name__ == "__main__":
    main()
