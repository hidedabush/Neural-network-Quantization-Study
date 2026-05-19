"""
experiments/sensitivity/layer_sweep.py
=======================================
Per-layer sensitivity analysis: quantize one layer at a time, measure accuracy
drop, restore FP32, move to next layer.

This produces the most important chart in the paper:
  X-axis: layer index (or layer name)
  Y-axis: accuracy drop (FP32 accuracy - quantized accuracy)
  High bar = sensitive layer (keep in FP16 in mixed-precision strategy)
  Low bar  = robust layer   (safe to quantize to INT8 or INT4)

Usage
-----
    python experiments/sensitivity/layer_sweep.py --model resnet18 --bits 8

Output: results/benchmarks/sensitivity_resnet18_int8.csv
        results/figures/sensitivity_resnet18_int8.png  (via tools/plot_results.py)

Learning notes (Days 29–31)
----------------------------
Why does sensitivity vary by layer?
  - Early conv layers learn low-level features (edges, textures). Their weight
    distributions are often wider → larger quantization error per step.
  - Later layers learn high-level semantics. They tend to be more robust.
  - The final FC layer has very large impact because it directly produces class logits.
  - Skip connections in ResNet add residuals that can amplify errors in sensitive layers.

This analysis is a real research contribution. If your sensitivity chart differs
from published results (e.g. on ImageNet), that's interesting — CIFAR-10 vs ImageNet
statistics differ and that's worth noting in your paper.
"""

import argparse
import csv
import copy
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from models.model_loader import load_model
from datasets.dataset_loader import load_cifar10
from experiments.ptq.run_ptq import QuantizedLinear, evaluate


def get_quantizable_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Return list of (name, module) for all Linear and Conv2d layers."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append((name, module))
    return layers


def quantize_single_layer(model: nn.Module, target_name: str, bits: int) -> nn.Module:
    """Quantize only the named layer; all others remain FP32."""
    clamp_val = (2 ** (bits - 1)) - 1
    for name, module in list(model.named_modules()):
        if name == target_name and isinstance(module, nn.Linear):
            w = module.weight.data.float()
            scale = w.abs().max().item() / clamp_val if w.abs().max().item() > 0 else 1.0
            q_module = QuantizedLinear(module, scale)
            # Navigate to parent and replace
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], q_module)
            break
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet18")
    parser.add_argument("--bits",  type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    print(f"\n{'═'*55}")
    print(f"  Sensitivity sweep: {args.model} INT{args.bits}")
    print(f"{'═'*55}\n")

    _, test_loader = load_cifar10(batch_size=64)

    # FP32 baseline accuracy
    fp32_model   = load_model(args.model, device=args.device)
    fp32_results = evaluate(fp32_model, test_loader, args.device, "FP32 baseline")
    fp32_acc     = fp32_results["accuracy"]

    layers = get_quantizable_layers(fp32_model)
    print(f"\nFound {len(layers)} quantizable layers. Running sweep...\n")

    results = []
    for i, (layer_name, _) in enumerate(layers):
        # Fresh FP32 model for each experiment (deepcopy is slow but correct)
        model = copy.deepcopy(fp32_model)
        model = quantize_single_layer(model, layer_name, args.bits)

        res = evaluate(model, test_loader, args.device, f"Layer {i}: {layer_name}")
        acc_drop = fp32_acc - res["accuracy"]
        results.append({
            "layer_index": i,
            "layer_name":  layer_name,
            "accuracy":    res["accuracy"],
            "acc_drop":    acc_drop,
        })
        print(f"    Accuracy drop: {acc_drop:.3f}%\n")
        del model  # free GPU memory

    # Save
    out_path = Path("results/benchmarks") / f"sensitivity_{args.model}_int{args.bits}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["layer_index", "layer_name", "accuracy", "acc_drop"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {out_path}")
    print("Run tools/plot_results.py to generate the sensitivity chart.\n")

    # Print top-5 most sensitive layers
    top5 = sorted(results, key=lambda r: r["acc_drop"], reverse=True)[:5]
    print("Top 5 most sensitive layers (quantize these last / keep in FP16):")
    for r in top5:
        print(f"  [{r['layer_index']:2d}] {r['layer_name']:<40}  drop: {r['acc_drop']:.3f}%")
    print()


if __name__ == "__main__":
    main()
