"""
datasets/dataset_loader.py
==========================
Loaders for CIFAR-10 (and optionally ImageNet-100).

CIFAR-10 is the default for all experiments:
  - 50,000 training images, 10,000 test images
  - 10 classes, 32×32 RGB images
  - Downloads automatically on first run (~162 MB)
  - Fits entirely in RAM — no bottleneck during calibration
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from pathlib import Path

DATA_ROOT = Path(__file__).parents[1] / "datasets" / "data"


def load_cifar10(
    batch_size: int = 64,
    num_workers: int = 2,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """
    Returns (train_loader, test_loader) for CIFAR-10.

    Normalization uses ImageNet mean/std because we start from a pretrained
    ResNet-18 with ImageNet weights. The model was trained expecting this
    normalization. Using different stats would degrade accuracy from day 1.
    """
    # ImageNet normalization (mean and std per channel)
    # These are fixed constants, not computed from CIFAR-10.
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

    # ResNet-18 expects 224×224 inputs (designed for ImageNet)
    # CIFAR-10 images are 32×32 — we resize up.
    # NOTE: This isn't ideal for CIFAR-10 (the model will still learn)
    # but it lets us use pretrained weights without architecture changes.
    # An alternative: train from scratch with 32×32. Either approach is fine
    # for our quantization study — document your choice in the paper.
    train_transform = T.Compose([
        T.Resize(224),
        T.RandomHorizontalFlip(),      # light augmentation for fine-tuning
        T.RandomCrop(224, padding=28),
        T.ToTensor(),
        normalize,
    ])

    test_transform = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        normalize,
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=True,  transform=train_transform, download=download)
    test_dataset  = torchvision.datasets.CIFAR10(
        root=DATA_ROOT, train=False, transform=test_transform,  download=download)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    print(f"CIFAR-10 loaded:")
    print(f"  Train: {len(train_dataset):,} images")
    print(f"  Test : {len(test_dataset):,} images")
    print(f"  Batch size: {batch_size}  |  Num workers: {num_workers}")

    return train_loader, test_loader
