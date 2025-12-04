# mynn/optim/sgd.py
from __future__ import annotations
from typing import Iterable
from ..nn.module import Parameter

class SGD:
    """
    Stochastic Gradient Descent optimizer.

    This is a minimal implementation: it assumes that each Parameter
    has a .data array and an optional .grad array of the same shape.
    """
    def __init__(self, params: Iterable[Parameter], lr: float = 1e-2):
        self.params = list(params)
        self.lr = lr

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

