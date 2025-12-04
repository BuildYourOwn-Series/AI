# mynn/nn/activations.py
from __future__ import annotations
import numpy as np
from .module import Module
from ..tensor import Tensor

class ReLU(Module):
    """Rectified Linear Unit activation."""
    def forward(self, x: Tensor) -> Tensor:
        mask = (x.data > 0).astype(np.float32)

        out = Tensor(x.data * mask, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = "relu"

        if not x.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if x.grad is None:
                x.grad = np.zeros_like(x.data, dtype=np.float32)
            x.grad += out.grad * mask

        out._backward = _backward
        return out


class Tanh(Module):
    """Hyperbolic tangent activation."""
    def forward(self, x: Tensor) -> Tensor:
        y = np.tanh(x.data)

        out = Tensor(y, requires_grad=x.requires_grad)
        out._prev = {x}
        out._op = "tanh"

        if not x.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if x.grad is None:
                x.grad = np.zeros_like(x.data, dtype=np.float32)
            # d/dx tanh(x) = 1 - tanh^2(x) = 1 - y^2
            x.grad += out.grad * (1.0 - y * y)

        out._backward = _backward
        return out
