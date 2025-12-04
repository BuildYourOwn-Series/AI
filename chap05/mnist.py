#!/usr/bin/env python3
# mnist.py
# A classification of "MNIST" using our "mynn" library
# Python >=3.9 recommended.

from mynn import Tensor, nn, optim, no_grad
import numpy as np
import os, gzip
import matplotlib.pyplot as plt

def load_mnist(path: str):
    """
    Load MNIST from the original IDX .gz files located in `path`:

        train-images-idx3-ubyte.gz
        train-labels-idx1-ubyte.gz
        t10k-images-idx3-ubyte.gz
        t10k-labels-idx1-ubyte.gz

    Returns:
        Xtr, Ytr, Xte, Yte
        where X* are float32 arrays of shape (N, 784) in [0,1],
        and Y* are one-hot float32 arrays of shape (N, 10).
    """

    def read_images(fname: str) -> np.ndarray:
        full = os.path.join(path, fname)
        with gzip.open(full, "rb") as f:
            data = f.read()
        # First 16 bytes: magic (2051), num, rows, cols as big-endian int32
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
        # First 8 bytes: magic (2049), num as big-endian int32
        header = np.frombuffer(data[:8], dtype=">i4")
        magic, num = header
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in {fname}, expected 2049.")
        labels = np.frombuffer(data[8:], dtype=np.uint8)
        if labels.shape[0] != num:
            raise ValueError(f"Label count mismatch in {fname}: {labels.shape[0]} vs {num}.")
        return labels

    # Read the four standard MNIST files
    Xtr = read_images("train-images-idx3-ubyte.gz")
    Ytr_raw = read_labels("train-labels-idx1-ubyte.gz")

    Xte = read_images("t10k-images-idx3-ubyte.gz")
    Yte_raw = read_labels("t10k-labels-idx1-ubyte.gz")

    # One-hot encode labels
    def one_hot(y: np.ndarray, C: int = 10) -> np.ndarray:
        out = np.zeros((y.size, C), np.float32)
        out[np.arange(y.size), y] = 1.0
        return out

    Ytr = one_hot(Ytr_raw)
    Yte = one_hot(Yte_raw)

    return Xtr, Ytr, Xte, Yte

def batches(X, Y, batch_size=128, shuffle=True):
    """
    Simple minibatch generator. Keeps the framework clean by leaving
    dataset logic outside `mynn`.
    """
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)

    for start in range(0, N, batch_size):
        end = start + batch_size
        b = idx[start:end]
        yield X[b], Y[b]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 128)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(128, 10)

    def forward(self, x):
        return self.l2(self.a1(self.l1(x)))

Xtr, Ytr, Xte, Yte = load_mnist("./mnist")

model = MLP()
opt   = optim.Adam(model.parameters(), lr=1e-3)
lossf = nn.CrossEntropyLoss()

for epoch in range(5):
    for xb, yb in batches(Xtr, Ytr, batch_size=128):
        logits = model(Tensor(xb))
        loss   = lossf(logits, Tensor(yb))

        opt.zero_grad()
        loss.backward()
        opt.step()

# Evaluate
with no_grad():
    pred = model(Tensor(Xte)).softmax(axis=1).data.argmax(axis=1)
    acc  = (pred == Yte.argmax(axis=1)).mean()

print("Test accuracy:", acc)

# ----------------------------------------------------------------------
# Interactive visualization loop: show random test digits and predictions
# ----------------------------------------------------------------------
while True:
    idx = np.random.randint(0, Xte.shape[0])
    x_np = Xte[idx]                   # shape: (784,)
    y_true = int(Yte[idx].argmax())

    with no_grad():
        logits = model(Tensor(x_np[None, :]))         # (1, 10)
        probs = logits.softmax(axis=1).data[0]        # (10,)
        y_pred = int(probs.argmax())
        confidence = float(probs[y_pred])

    print(f"Index {idx} — predicted {y_pred} (p={confidence:.3f}), true {y_true}")

    # Show the image
    plt.figure()
    plt.imshow(x_np.reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {y_pred} (p={confidence:.2f}) — True: {y_true}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Prompt the user
    ans = input("[n]ext / [s]top? ").strip().lower()
    if ans.startswith("s"):
        break
