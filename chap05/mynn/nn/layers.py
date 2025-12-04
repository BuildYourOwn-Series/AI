# mynn/nn/layers.py
from __future__ import annotations
import numpy as np
from .module import Module, Parameter
from ..tensor import Tensor

class Linear(Module):
    """
    Affine layer: y = x W + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # very simple initialization for now
        self.W = Parameter(
            np.random.randn(in_features, out_features).astype(np.float32) * 0.1
        )
        self.b = Parameter(
            np.zeros((1, out_features), dtype=np.float32)
        ) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        y = x @ self.W
        if self.b is not None:
            y = y + self.b
        return y
