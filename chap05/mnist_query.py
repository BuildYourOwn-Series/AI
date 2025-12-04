#!/usr/bin/env python3
# mnist_query.py
# Load a trained MNIST model (saved by mnist_train.py) and query it interactively.

from mynn import Tensor, nn, no_grad
import numpy as np
import gzip
import os
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Same MNIST loader as in mnist_train.py
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Same MLP architecture as in mnist_train.py
# ----------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))

# ----------------------------------------------------------------------
# Load, evaluate, and interactively query
# ----------------------------------------------------------------------

if __name__ == "__main__":
    Xtr, Ytr, Xte, Yte = load_mnist("./mnist")

    model = MLP()
    model.load_npz("mnist_mlp.npz")
    print("Loaded model parameters from mnist_mlp.npz")

    # Evaluate
    with no_grad():
        probs = model(Tensor(Xte)).softmax(axis=1).data
        pred  = probs.argmax(axis=1)
        acc   = (pred == Yte.argmax(axis=1)).mean()

    print("Test accuracy (after reload):", acc)

    # Interactive visualization
    while True:
        idx = np.random.randint(0, Xte.shape[0])
        x_np = Xte[idx]                   # (784,)
        y_true = int(Yte[idx].argmax())

        with no_grad():
            logits = model(Tensor(x_np[None, :]))      # (1, 10)
            probs  = logits.softmax(axis=1).data[0]    # (10,)
            y_pred = int(probs.argmax())
            confidence = float(probs[y_pred])

        print(f"Index {idx} — predicted {y_pred} (p={confidence:.3f}), true {y_true}")

        plt.figure()
        plt.imshow(x_np.reshape(28, 28), cmap="gray")
        plt.title(f"Pred: {y_pred} (p={confidence:.2f}) — True: {y_true}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

        ans = input("[n]ext / [s]top? ").strip().lower()
        if ans.startswith("s"):
            break
