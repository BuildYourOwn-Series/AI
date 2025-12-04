# mynn/autograd.py
from __future__ import annotations
from typing import List, Set
import numpy as np
from .tensor import Tensor

def _topological_sort(t: Tensor) -> List[Tensor]:
    """Return tensors in reverse topological order ending at t."""
    visited: Set[Tensor] = set()
    order: List[Tensor] = []

    def build(x: Tensor) -> None:
        if x not in visited:
            visited.add(x)
            for p in x._prev:
                build(p)
            order.append(x)

    build(t)
    return order

def backward(t: Tensor) -> None:
    """
    Entry point for backpropagation: given a scalar tensor t,
    compute gradients for all tensors that contributed to it.
    For now this only sets up the traversal; operations will
    plug their own _backward hooks into the graph.
    """
    if t.grad is None:
        t.grad = np.ones_like(t.data, dtype=np.float32)
    for node in reversed(_topological_sort(t)):
        node._backward()

