#!/usr/bin/env python3
# mnist_train.py
# Train an MNIST classifier using our "mynn" library and save its weights.

from mynn import Tensor, nn, optim, seed, no_grad
import numpy as np
import gzip
import os

# ----------------------------------------------------------------------
# MNIST LOADING (IDX .gz files, lives outside the framework)
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# Model definition (must match the one we will use for querying)
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
# Training
# ----------------------------------------------------------------------

if __name__ == "__main__":
    seed(0)

    Xtr, Ytr, Xte, Yte = load_mnist("./mnist")

    model = MLP()
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    for epoch in range(5):
        epoch_loss = 0.0
        for xb, yb in batches(Xtr, Ytr, batch_size=128):
            logits = model(Tensor(xb))
            loss   = lossf(logits, Tensor(yb))

            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += float(loss.data)

        print(f"Epoch {epoch+1}: loss = {epoch_loss:.3f}")

    # Evaluate once before saving
    with no_grad():
        probs = model(Tensor(Xte)).softmax(axis=1).data
        pred  = probs.argmax(axis=1)
        acc   = (pred == Yte.argmax(axis=1)).mean()

    print("Test accuracy:", acc)

    # Save trained weights
    model.save_npz("mnist_mlp.npz")
    print("Saved model parameters to mnist_mlp.npz")
