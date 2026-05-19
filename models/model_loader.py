"""
models/model_loader.py
======================
Consistent API for loading ResNet-18 and DistilBERT with FP32 / FP16 / INT8 modes.

Usage
-----
    from models.model_loader import load_model

    # FP32 baseline
    model = load_model("resnet18", precision="fp32", pretrained=True)

    # FP16 (move weights to half precision)
    model = load_model("resnet18", precision="fp16", pretrained=True)

Learning notes (Day 23–25)
--------------------------
When you load a model here, the weights live in CPU RAM as FP32 tensors.
Calling .cuda() copies them to GPU VRAM. Calling .half() converts FP32 → FP16.

The "INT8 mode" here is a placeholder — real INT8 inference requires replacing
the model's linear layers with custom INT8 modules. See experiments/ptq/run_ptq.py
for how that works.
"""

from typing import Literal
import torch
import torch.nn as nn
import torchvision.models as tv_models

ModelName = Literal["resnet18", "distilbert"]
Precision = Literal["fp32", "fp16"]


def load_model(
    name: ModelName,
    precision: Precision = "fp32",
    pretrained: bool = True,
    num_classes: int = 10,   # CIFAR-10
    device: str = "cuda",
) -> nn.Module:
    """
    Load a model with the given precision and move it to device.

    Parameters
    ----------
    name        : "resnet18" or "distilbert"
    precision   : "fp32" (default) or "fp16"
    pretrained  : if True, load ImageNet weights (will be fine-tuned on CIFAR-10)
    num_classes : output classes (10 for CIFAR-10)
    device      : "cuda" or "cpu"

    Returns
    -------
    nn.Module — model in eval mode on the requested device
    """

    if name == "resnet18":
        model = _load_resnet18(pretrained, num_classes)
    elif name == "distilbert":
        model = _load_distilbert(num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose 'resnet18' or 'distilbert'.")

    model = model.to(device)

    if precision == "fp16":
        # Convert all parameters and buffers to FP16.
        # NOTE: batch norm layers can be unstable in FP16; keep them in FP32.
        # This is why PyTorch's autocast uses BF16 for BN by default.
        # For this study we do a simple .half() and note any instability.
        model = model.half()

    model.eval()  # disable dropout and batch norm running stats update

    _print_model_info(model, name, precision, device)
    return model


def _load_resnet18(pretrained: bool, num_classes: int) -> nn.Module:
    """
    Load ResNet-18.

    Architecture overview (relevant for sensitivity analysis):
      Input → conv1 → bn1 → relu → maxpool
           → layer1 (2× BasicBlock)
           → layer2 (2× BasicBlock)
           → layer3 (2× BasicBlock)
           → layer4 (2× BasicBlock)
           → avgpool → fc (512 → num_classes)

    Each BasicBlock: conv3×3 → bn → relu → conv3×3 → bn → (+ skip connection)
    Total: 8 conv layers + 1 FC = 9 quantization candidates (for sensitivity sweep).

    Why ResNet-18 for quantization study:
    - Small enough (11M params, ~44MB FP32) to run experiments fast on 4060
    - Residual connections make it interesting: does the skip-connection path
      tolerate quantization differently from the main path? (Good research question.)
    - Well-studied in the quantization literature, so you can compare your results
      to published numbers (e.g. Table 1 in arXiv:2004.09602).
    """
    weights = tv_models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = tv_models.resnet18(weights=weights)

    # Replace the final FC layer for CIFAR-10 (10 classes vs ImageNet's 1000)
    model.fc = nn.Linear(512, num_classes)
    return model


def _load_distilbert(num_classes: int) -> nn.Module:
    """
    Load DistilBERT for sequence classification.

    Requires: pip install transformers
    Used in Phase 2 experiments as an NLP counterpart to ResNet-18.

    NOTE: DistilBERT's linear layers are in the attention and FFN blocks.
    For quantization, the interesting targets are:
      - q_lin, k_lin, v_lin, out_lin (attention projections)
      - lin1, lin2 (FFN layers)
    The embedding layer is typically kept in FP32 (it's a lookup table, not a GEMM).
    """
    try:
        from transformers import DistilBertForSequenceClassification
    except ImportError:
        raise ImportError(
            "DistilBERT requires the 'transformers' library.\n"
            "Install with: pip install transformers"
        )
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    return model


def _print_model_info(model: nn.Module, name: str, precision: str, device: str):
    """Print a summary useful for the daily journal."""
    total_params = sum(p.numel() for p in model.parameters())
    total_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

    print(f"\nModel loaded: {name}")
    print(f"  Precision : {precision}")
    print(f"  Device    : {device}")
    print(f"  Parameters: {total_params:,}")
    print(f"  Size      : {total_mb:.1f} MB")
    print(f"  GPU VRAM  : {torch.cuda.memory_allocated() / 1e6:.1f} MB allocated\n")


def count_quantizable_layers(model: nn.Module) -> int:
    """
    Count the number of layers that can be quantized (Conv2d and Linear).
    Used in the sensitivity sweep to know how many experiments to run.
    """
    count = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            count += 1
    return count
