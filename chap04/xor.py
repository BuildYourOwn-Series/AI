#!/usr/bin/env python3
# xor.py
# A multilayer perceptron for binary classification of "xor"
# Python >=3.9 recommended.

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List

def glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator, scale=1.0):
    a = scale * np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out)).astype(np.float32)

@dataclass
class Layer:
    W: np.ndarray  # (in_dim, out_dim)
    b: np.ndarray  # (out_dim,)

class MLP:
    def __init__(self, layers: List[Layer], alpha: float = 1.0):
        self.layers = layers
        self.alpha = float(alpha)  # scale inside tanh to avoid saturation if needed

    @staticmethod
    def tanh(x: np.ndarray, a: float) -> np.ndarray:
        return np.tanh(a * x, dtype=np.float32)

    @classmethod
    def from_shapes(cls, shapes: List[int], seed: int = 0, alpha: float = 1.0, bias_break: bool = True):
        rng = np.random.default_rng(seed)
        layers: List[Layer] = []
        for din, dout in zip(shapes[:-1], shapes[1:]):
            W = glorot_uniform(din, dout, rng, scale=0.5)  # slightly gentler init
            b = np.zeros((dout,), dtype=np.float32)
            layers.append(Layer(W=W, b=b))
        # tiny symmetry break for XOR: opposite biases on first hidden if present
        if bias_break and len(layers) >= 2 and layers[0].b.size >= 2:
            layers[0].b[0] = +0.5
            layers[0].b[1] = -0.5
        return cls(layers, alpha=alpha)

    # -------- Forward (batch) --------
    def forward(self, X: np.ndarray):
        """Return (Zs, As) where As[0]=X and As[-1]=Yhat."""
        a = self.alpha
        A = [X.astype(np.float32)]   # A[0] = input
        Zs = []
        for layer in self.layers:
            Z = A[-1] @ layer.W + layer.b        # (B, out_dim)
            A_next = np.tanh(a * Z, dtype=np.float32)
            Zs.append(Z)
            A.append(A_next)
        return Zs, A  # A[-1] = Yhat

    def train_step_batch(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.1):
        a = self.alpha
        Zs, A = self.forward(X)
        Yh = A[-1]
        B = X.shape[0]

        # dL/dZ_L  (MSE with tanh output)
        dY  = (Yh - Y) / B                          # (B, o)
        dZ  = dY * (a * (1.0 - Yh * Yh))            # (B, o)

        # Backprop layers L..1
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            A_prev = A[li]                          # activation feeding this layer
            # grads for this layer
            gW = A_prev.T @ dZ                      # (in_dim, out_dim)
            gb = dZ.sum(axis=0)                     # (out_dim,)
            # SGD update
            layer.W -= lr * gW
            layer.b -= lr * gb
            # propagate to previous layer (if any)
            if li > 0:
                dA_prev = dZ @ layer.W.T            # (B, in_dim)
                A_li = A[li]                        # activation of previous (post-nonlinearity)
                dZ = dA_prev * (a * (1.0 - A_li * A_li))
        loss = 0.5 * float(np.mean((Yh - Y)**2))
        return loss

    def train_step_sgd(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        i = int(rng.integers(0, X.shape[0]))
        xi = X[i:i+1, :]     # (1,d)
        yi = Y[i:i+1, :]     # (1,o)
        return self.train_step_batch(xi, yi, lr)

    def train_step_batch(self, X: np.ndarray, Y: np.ndarray, lr: float = 0.1):
        a = self.alpha
        Zs, A = self.forward(X)          # A[0]=X, A[-1]=Yhat
        Yh = A[-1]; B = X.shape[0]
        dZ = ((Yh - Y) / B) * (a * (1.0 - Yh*Yh))
        for li in reversed(range(len(self.layers))):
            layer = self.layers[li]
            A_prev = A[li]
            gW = A_prev.T @ dZ            # (in_dim, out_dim)
            gb = dZ.sum(axis=0)           # (out_dim,)
            layer.W -= lr * gW
            layer.b -= lr * gb
            if li > 0:
                dA_prev = dZ @ layer.W.T
                A_li = A[li]
                dZ = dA_prev * (a * (1.0 - A_li*A_li))
        return 0.5 * float(np.mean((Yh - Y)**2))

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int = 4000, lr: float = 0.1,
            mode: str = "sgd", batch_size: int = 32, seed: int = 0, verbose_every: int = 300):
        rng = np.random.default_rng(seed)
        losses = []
        N = X.shape[0]
        for ep in range(epochs):
            if mode == "sgd":
                i = int(rng.integers(0, N))
                loss = self.train_step_batch(X[i:i+1], Y[i:i+1], lr)
            else:
                idx = rng.permutation(N)
                Xs, Ys = X[idx], Y[idx]
                loss = 0.0
                for s in range(0, N, batch_size):
                    e = min(s + batch_size, N)
                    loss = self.train_step_batch(Xs[s:e], Ys[s:e], lr)
            losses.append(loss)
            if verbose_every and ep % verbose_every == 0:
                print(f"epoch {ep:4d}: loss={loss:.6f}")
        return losses

    def predict(self, X: np.ndarray):
        Zs, A = self.forward(X)      # A is [A0, A1, ..., AL]
        Yh = A[-1]                   # final activation (B, o)
        return np.sign(Yh), Yh

    def query(self, x1: float, x2: float):
        """
        Query the trained network with a single 2-D input (e.g., XOR pair).
        Returns (label, raw_value).
        """
        x = np.array([[x1, x2]], np.float32)
        label, raw = self.predict(x)
        label_i = int(label.item())
        raw_f = float(raw.item())
        print(f"input=({x1:+.1f}, {x2:+.1f}) -> raw={raw_f:+.3f}, label={label_i:+d}")
        return label_i, raw_f

# XOR dataset in {-1,+1}
X01 = np.array([[0,0],[0,1],[1,0],[1,1]], np.float32)
X = 2*X01 - 1
Y = np.array([[-1],[+1],[+1],[-1]], np.float32)

# Create the network
net = MLP.from_shapes([2,2,1], seed=7, alpha=0.5, bias_break=True)

# Train and record loss
losses = []
epochs = 4000
for ep in range(epochs):
    loss = net.train_step_sgd(X, Y, lr=0.1)
    losses.append(loss)
    if ep % 300 == 0:
        print(f"epoch {ep:4d}: loss={loss:.6f}")

# Evaluate
pred, raw = net.predict(X)
print("pred:", pred.ravel(), "raw:", np.round(raw.ravel(), 3))
print("tgt :", Y.ravel())

# Plot loss curve
plt.figure(figsize=(5,3))
plt.plot(losses, lw=1.5)
plt.xlabel("Training step")
plt.ylabel("Mean squared error / 2")
plt.title("XOR training loss")
plt.grid(True, ls=":")
plt.tight_layout()
#plt.show()

net.query(-1, -1)
net.query(-1, +1)
net.query(+1, -1)
net.query(+1, +1)
