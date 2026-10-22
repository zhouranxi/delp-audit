"""exp6_fixed_fashion.py -- corrected 8-arm comparison on Permuted-Fashion-MNIST.

Second-benchmark replication of exp6_fixed.py. Protocol, arms, lambda grids,
val-selection rule, seeds and architecture are IDENTICAL; only the dataset
changes: Fashion-MNIST (Zalando 2017) in place of MNIST.

Implementation note: the server venv has no torchvision, so the raw IDX files
are parsed directly with numpy/gzip from ./fashion_data/ (same directory
layout as torchvision's ./data). Everything else is delegated to exp6_fixed.

Usage (matches exp6_fixed.py):
    python exp6_fixed_fashion.py --lam-rule val --task-id 0 --n-tasks 8 \
        --stage sweep --out results_fashion/sweep.json
    python exp6_fixed_fashion.py --lam-rule val --task-id 0 --n-tasks 8 \
        --stage main --lam-map '<json>' --out results_fashion/main.json
"""
import gzip
import json
import os

import numpy as np
import torch

import exp6_fixed

DATA_DIR = os.environ.get("FASHION_DATA", "./fashion_data")

_FILES = {
    "train_images": "fashion-train-images-idx3-ubyte.gz",
    "train_labels": "fashion-train-labels-idx1-ubyte.gz",
    "test_images": "fashion-t10k-images-idx3-ubyte.gz",
    "test_labels": "fashion-t10k-labels-idx1-ubyte.gz",
}


def _read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[0:4], "big")
    n = int.from_bytes(raw[4:8], "big")
    if magic == 2051:                       # images: rows 784, uint8 -> /255
        assert len(raw) == 16 + n * 784, f"truncated {path}"
        x = torch.from_numpy(
            np.frombuffer(raw, np.uint8, offset=16, count=n * 784).copy()
        ).float().view(n, 784) / 255.0
        return x
    assert magic == 2049, f"bad magic {magic} in {path}"
    return torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8, count=n).copy())


def load_fashion():
    """Same return signature as exp6_baselines.load_mnist()."""
    xtr = _read_idx(os.path.join(DATA_DIR, _FILES["train_images"]))
    ytr = _read_idx(os.path.join(DATA_DIR, _FILES["train_labels"]))
    xte = _read_idx(os.path.join(DATA_DIR, _FILES["test_images"]))
    yte = _read_idx(os.path.join(DATA_DIR, _FILES["test_labels"]))
    return xtr, ytr, xte, yte


def load_val_fashion():
    """Held-out split train[NTRAIN:NTRAIN+2000], mirroring exp6_fixed.load_val."""
    ntrain = exp6_fixed.NTRAIN
    xtr, ytr, _, _ = load_fashion()
    return xtr[ntrain:ntrain + 2000], ytr[ntrain:ntrain + 2000]


# Patch both entry points, then delegate to the original driver.
exp6_fixed.load_mnist = load_fashion
exp6_fixed.load_val = load_val_fashion

if __name__ == "__main__":
    exp6_fixed.main()
