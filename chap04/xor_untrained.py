#!/usr/bin/env python3
# xor.py
# A multilayer perceptron for binary classification of "xor"
# Python >=3.9 recommended.

import numpy as np
from dataclasses import dataclass
from typing import List

def glorot_uniform(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    a = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-a, a, size=(fan_in, fan_out)).astype(np.float32)

@dataclass
class Layer:
    W: np.ndarray  # (in_dim, out_dim)
    b: np.ndarray  # (out_dim,)

@dataclass
class MLP:
    layers: List[Layer]

    @staticmethod
    def tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(x, dtype=np.float32)

    @classmethod
    def from_shapes(cls, shapes: List[int], seed: int = 0) -> "MLP":
        rng = np.random.default_rng(seed)
        layers: List[Layer] = []
        for din, dout in zip(shapes[:-1], shapes[1:]):
            W = glorot_uniform(din, dout, rng)
            b = np.zeros((dout,), dtype=np.float32)
            assert W.shape == (din, dout) and b.shape == (dout,)
            layers.append(Layer(W=W, b=b))
        return cls(layers)

    def forward(self, X: np.ndarray) -> np.ndarray:
        Z = X.astype(np.float32)
        for i, layer in enumerate(self.layers):
            Z = Z @ layer.W + layer.b  # affine
            # keep tanh on all layers including output for XOR in {-1,+1}
            Z = self.tanh(Z)
        return Z

if __name__ == "__main__":
    # XOR dataset mapped to {-1,+1}
    X01 = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    X   = 2*X01 - 1
    Y   = np.array([[-1],[+1],[+1],[-1]], dtype=np.float32)

    net = MLP.from_shapes([2,2,1], seed=42)
    Yhat = net.forward(X)
    print("Yhat (untrained):\n", Yhat)
