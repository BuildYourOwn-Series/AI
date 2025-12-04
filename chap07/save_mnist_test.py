#!/usr/bin/env python3
"""
save_mnist_test.py

Use the Chapter 05 load_mnist() function to extract the
test split and save it as mnist_test.npz. This file is
used by the quantized inference script in Chapter 07.
"""

import numpy as np
import gzip
import os
from pathlib import Path

def load_mnist(path: str):
    def read_images(fname: str) -> np.ndarray:
        full = os.path.join(path, fname)
        with gzip.open(full, "rb") as f:
            data = f.read()
        header = np.frombuffer(data[:16], dtype=">i4")
        magic, num, rows, cols = header
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in {fname}, expected 2051.")
        images = np.frombuffer(data[16:], dtype=np.uint8)
        images = images.reshape(num, rows * cols).astype(np.float32) / 255.0
        return images

    def read_labels(fname: str) -> np.ndarray:
        full = os.path.join(path, fname)
        with gzip.open(full, "rb") as f:
            data = f.read()
        header = np.frombuffer(data[:8], dtype=">i4")
        magic, num = header
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {fname}, expected 2049.")
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        if labels.shape[0] != num:
            raise ValueError(f"Label count mismatch in {fname}: {labels.shape[0]} vs {num}.")
        return labels

    Xtr = read_images("train-images-idx3-ubyte.gz")
    Ytr_raw = read_labels("train-labels-idx1-ubyte.gz")

    Xte = read_images("t10k-images-idx3-ubyte.gz")
    Yte_raw = read_labels("t10k-labels-idx1-ubyte.gz")

    def one_hot(y: np.ndarray, C: int = 10) -> np.ndarray:
        out = np.zeros((y.size, C), np.float32)
        out[np.arange(y.size), y] = 1.0
        return out

    Ytr = one_hot(Ytr_raw)
    Yte = one_hot(Yte_raw)

    return Xtr, Ytr, Xte, Yte

OUT = Path("mnist_test.npz")

def main():
    Xtr, Ytr, Xte, Yte = load_mnist("./mnist")

    # If Yte is one-hot, compress it now:
    if Yte.ndim == 2 and Yte.shape[1] == 10:
        y_idx = np.argmax(Yte, axis=1).astype(np.int64)
    else:
        y_idx = Yte.astype(np.int64)

    # Store the raw test images as uint8; normalization happens at inference.
    np.savez_compressed(
        OUT,
        x_test=Xte.astype(np.float32),
        y_test=y_idx,
    )

    print(f"Saved {OUT} with {Xte.shape[0]} test images.")

if __name__ == "__main__":
    main()
