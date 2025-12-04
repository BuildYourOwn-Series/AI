#!/usr/bin/env python3
# xor.py
# A multilayer perceptron for binary classification of "xor"
# Python >=3.9 recommended.

import numpy as np
from dataclasses import dataclass
from typing import List
import gzip, struct
import matplotlib.pyplot as plt

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)      # numerical stability
    e = np.exp(z, dtype=np.float32)
    return e / e.sum(axis=1, keepdims=True)

@dataclass
class Layer:
    W: np.ndarray  # (in_dim, out_dim)
    b: np.ndarray  # (out_dim,)

class MLP:
    def __init__(self, layers: List[Layer], alpha: float = 1.0):
        self.layers = layers
        self.alpha  = float(alpha)

    @classmethod
    def from_shapes(cls, shapes: List[int], seed: int = 0, alpha: float = 1.0, bias_break: bool = False):
        rng = np.random.default_rng(seed)
        layers: List[Layer] = []
        for din, dout in zip(shapes[:-1], shapes[1:]):
            a = np.sqrt(6.0/(din+dout)) * 0.5     # gentle Glorot
            W = rng.uniform(-a, a, size=(din, dout)).astype(np.float32)
            b = np.zeros((dout,), np.float32)
            layers.append(Layer(W,b))
        return cls(layers, alpha=alpha)

    # ------- Forward: hidden = tanh, output = linear logits -------
    def forward(self, X: np.ndarray):
        a = self.alpha
        A = [X.astype(np.float32)]
        Zs = []
        L  = len(self.layers)
        for li, layer in enumerate(self.layers):
            Z = A[-1] @ layer.W + layer.b
            Zs.append(Z)
            if li == L-1:
                A.append(Z.astype(np.float32))     # last layer: linear logits
            else:
                A.append(np.tanh(a*Z, dtype=np.float32))
        return Zs, A  # logits are Zs[-1], last activation is A[-1] (same here)

    # ------- One CE mini-batch step (vectorized) -------
    def train_step_ce(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.05):
        Zs, A = self.forward(X)
        logits = Zs[-1]                 # (B,10)
        P = softmax(logits)             # (B,10)
        B = X.shape[0]
        dZ = (P - Y) / B                # dL/dZ_last for softmax+CE

        a = self.alpha
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            A_prev = A[li]
            gW = A_prev.T @ dZ
            gb = dZ.sum(axis=0)
            layer.W -= lr * gW
            layer.b -= lr * gb
            if li > 0:
                dA_prev = dZ @ layer.W.T
                A_li    = A[li]                     # tanh activation
                dZ = dA_prev * (a*(1.0 - A_li*A_li))

        # negative log-likelihood for logging
        eps = 1e-12
        nll = -np.log(P[np.arange(B), Y.argmax(axis=1)] + eps).mean()
        return nll

    # ------- Inference helpers -------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward(X)[0][-1]
        return softmax(logits)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    # ------- Query (single image, top-k) -------
    def query(self, x, topk: int = 3, normalize: bool = True):
        """
        Query the trained classifier with one image.
        x: shape (28,28) or (784,), dtype uint8/float32.
        Returns (pred_class, pred_prob, topk_indices, topk_probs).
        """
        x = np.asarray(x)
        if x.ndim == 2:                 # 28x28 -> 784
            x = x.reshape(1, -1)
        elif x.ndim == 1:               # 784 ->
            x = x.reshape(1, -1)
        else:
            raise ValueError(f"bad shape: {x.shape}")

        x = x.astype(np.float32)
        if normalize and x.max() > 1.0: # allow raw 0..255
            x = x / 255.0

        P = self.predict_proba(x)[0]    # (10,)
        topk = int(np.clip(topk, 1, P.size))
        idx = np.argsort(-P)[:topk]
        probs = P[idx]
        pred = int(idx[0]); p = float(probs[0])

        # pretty print
        tops = ", ".join(f"{i}:{probs[j]:.3f}" for j,i in enumerate(idx))
        print(f"pred={pred}  p={p:.3f}  |  top{topk}: {tops}")
        return pred, p, idx, probs

def _read_idx_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, "bad magic number for images"
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(n, rows*cols).astype(np.float32) / 255.0

def _read_idx_labels(path):
    with gzip.open(path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, "bad magic number for labels"
        return np.frombuffer(f.read(), dtype=np.uint8)

def one_hot(y, num_classes=10):
    Y = np.zeros((y.size, num_classes), np.float32)
    Y[np.arange(y.size), y] = 1.0
    return Y

def load_mnist(root="mnist"):
    Xtr = _read_idx_images("mnist/train-images-idx3-ubyte.gz")
    ytr = _read_idx_labels("mnist/train-labels-idx1-ubyte.gz")
    Xte = _read_idx_images("mnist/t10k-images-idx3-ubyte.gz")
    yte = _read_idx_labels("mnist/t10k-labels-idx1-ubyte.gz")
    return (Xtr, one_hot(ytr), ytr), (Xte, one_hot(yte), yte)

(Xtr, Ytr, ytr), (Xte, Yte, yte) = load_mnist("mnist")

net = MLP.from_shapes([784, 64, 10], seed=42)

def fit_ce(net, X, Y, epochs=5, lr=0.05, batch_size=128, seed=123):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    for ep in range(epochs):
        idx = rng.permutation(N)
        Xs, Ys = X[idx], Y[idx]
        for s in range(0, N, batch_size):
            e = min(s+batch_size, N)
            loss = net.train_step_ce(Xs[s:e], Ys[s:e], lr=lr)
        print(f"epoch {ep+1:2d}/{epochs}: loss={loss:.4f}")

def show_digit(net, x, topk=3, normalize=True):
    """
    Display a single MNIST image and annotate prediction.
    x: shape (784,) or (28,28); dtype uint8/float32.
    """
    x = np.asarray(x)
    img = x.reshape(28,28) if x.size == 784 else x
    assert img.shape == (28,28), f"bad shape: {img.shape}"

    # Query the network (reuses the same normalization as training)
    pred, p, idx, probs = net.query(x, topk=topk, normalize=normalize)

    # Plot
    plt.figure(figsize=(2.8,2.8))
    plt.imshow(img, cmap="gray_r", interpolation="nearest")
    plt.axis("off")
    title = f"pred = {pred} (p={p:.3f})   top{topk}: " + \
            ", ".join(f"{i}:{probs[j]:.3f}" for j,i in enumerate(idx))
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.show()

def interactive_test(net, Xte, yte, topk=3):
    """
    Interactive MNIST viewer: displays random test samples and predictions.
    Type 'n' or Enter for next, 's' to stop.
    """
    rng = np.random.default_rng()
    while True:
        i = rng.integers(0, Xte.shape[0])
        true_label = int(yte[i])
        print(f"\n=== Sample {i} (true label: {true_label}) ===")
        show_digit(net, Xte[i], topk=topk)
        cmd = input("predict [n]ext or [s]top? ").strip().lower()
        if cmd.startswith("s"):
            print("Stopped.")
            break

fit_ce(net, Xtr, Ytr, epochs=5, lr=0.05, batch_size=128)

# Evaluate accuracy
pred_tr = net.predict_classes(Xtr)
pred_te = net.predict_classes(Xte)
acc_tr = (pred_tr == ytr).mean()
acc_te = (pred_te == yte).mean()
print(f"train acc={acc_tr:.3f}, test acc={acc_te:.3f}")

# Query a sample
interactive_test(net, Xte, yte, topk=5)
