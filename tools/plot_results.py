"""
tools/plot_results.py
=====================
Generate all paper figures from results/benchmarks/ CSVs.
Run after experiments complete: python tools/plot_results.py

Figures saved to results/figures/
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

RESULTS_DIR = Path("results/benchmarks")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Use a clean style for paper figures
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "figure.dpi":       150,
})


def plot_ptq_comparison():
    """Bar chart: FP32 vs INT8 vs INT4 accuracy and latency."""
    fp = RESULTS_DIR / "ptq_resnet18_8bit.csv"
    if not fp.exists():
        print(f"  [skip] {fp} not found — run experiments/ptq/run_ptq.py first")
        return

    with open(fp) as f:
        rows = list(csv.DictReader(f))

    labels    = [r["label"] for r in rows]
    accuracy  = [float(r["accuracy"])   for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    colors = ["#378ADD", "#EF9F27", "#D85A30"][:len(rows)]
    ax1.bar(labels, accuracy, color=colors, width=0.5)
    ax1.set_ylabel("Top-1 accuracy (%)")
    ax1.set_title("Accuracy: FP32 vs quantized")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
    ax1.set_ylim(max(0, min(accuracy) - 2), 100)

    ax2.bar(labels, latencies, color=colors, width=0.5)
    ax2.set_ylabel("Avg batch latency (ms)")
    ax2.set_title("Latency: FP32 vs quantized")

    plt.tight_layout()
    out = FIGURES_DIR / "ptq_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_sensitivity():
    """Bar chart: per-layer accuracy drop."""
    fp = RESULTS_DIR / "sensitivity_resnet18_int8.csv"
    if not fp.exists():
        print(f"  [skip] {fp} not found — run experiments/sensitivity/layer_sweep.py first")
        return

    with open(fp) as f:
        rows = list(csv.DictReader(f))

    names = [r["layer_name"].split(".")[-1] + f"\n({r['layer_index']})" for r in rows]
    drops = [float(r["acc_drop"]) for r in rows]

    # Color bars: red = high sensitivity, green = low
    max_drop = max(drops) if drops else 1
    colors = [plt.cm.RdYlGn_r(d / max_drop) for d in drops]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(names, drops, color=colors, width=0.7)
    ax.set_xlabel("Layer (index)")
    ax.set_ylabel("Accuracy drop (%)")
    ax.set_title("Per-layer INT8 quantization sensitivity (ResNet-18 / CIFAR-10)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="0.5% threshold")
    ax.legend()
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    out = FIGURES_DIR / "sensitivity.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


def plot_kernel_throughput():
    """
    Placeholder: plot kernel throughput from benchmark.cu output.
    Populate results/benchmarks/kernel_throughput.csv manually from
    the benchmark binary output, then run this script.

    CSV columns: kernel_name, n_elements, latency_ms, throughput_gbs
    """
    fp = RESULTS_DIR / "kernel_throughput.csv"
    if not fp.exists():
        print(f"  [skip] {fp} not found — run ./build/benchmark and paste output into CSV")
        return

    with open(fp) as f:
        rows = list(csv.DictReader(f))

    names  = [r["kernel_name"]       for r in rows]
    gbs    = [float(r["throughput_gbs"]) for r in rows]
    peak   = 272.0  # RTX 4060 theoretical DRAM bandwidth

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, gbs, color="#534AB7", height=0.5)
    ax.axvline(peak, color="red", linestyle="--", linewidth=1, label=f"Theoretical peak ({peak} GB/s)")
    ax.set_xlabel("Throughput (GB/s)")
    ax.set_title("CUDA kernel memory throughput vs theoretical peak")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "kernel_throughput.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out}")


if __name__ == "__main__":
    print("\nGenerating figures from results/benchmarks/...\n")
    plot_ptq_comparison()
    plot_sensitivity()
    plot_kernel_throughput()
    print("\nDone. Figures in results/figures/\n")
