# mynn/tensor.py
from __future__ import annotations
from typing import Any, Optional, Set, Callable
import numpy as np


def _ensure_tensor(x: Any) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


def _unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Sum grad over broadcasted dimensions so it matches 'shape'.
    This lets us handle cases like adding a bias of shape (1, D) to (N, D).
    """
    g = grad
    # remove leading dims
    while len(g.shape) > len(shape):
        g = g.sum(axis=0)
    # collapse axes where original shape was 1
    for i, (gs, s) in enumerate(zip(g.shape, shape)):
        if s == 1 and gs != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Tensor:
    """
    Thin wrapper around a NumPy array that carries gradient information
    and links into the autograd graph.
    """
    def __init__(self, data: Any, requires_grad: bool = False):
        self.data = np.asarray(data, dtype=np.float32)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad

        # autograd bookkeeping
        self._backward: Callable[[], None] = lambda: None
        self._prev: Set["Tensor"] = set()
        self._op: str = ""

    def __repr__(self) -> str:
        return f"Tensor(shape={self.data.shape}, requires_grad={self.requires_grad})"

    # ------------------------------------------------------------------ core ops

    def __add__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data + other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = "add"

        if not out.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data, dtype=np.float32)
                grad_self = out.grad
                if grad_self.shape != self.data.shape:
                    grad_self = _unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data, dtype=np.float32)
                grad_other = out.grad
                if grad_other.shape != other.data.shape:
                    grad_other = _unbroadcast(grad_other, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __radd__(self, other: Any) -> "Tensor":
        return self.__add__(other)

    def __neg__(self) -> "Tensor":
        out = Tensor(-self.data, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "neg"

        if not self.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=np.float32)
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: Any) -> "Tensor":
        return self.__add__(-_ensure_tensor(other))

    def __rsub__(self, other: Any) -> "Tensor":
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other: Any) -> "Tensor":
        other = _ensure_tensor(other)
        out = Tensor(self.data * other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = "mul"

        if not out.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data, dtype=np.float32)
                grad_self = out.grad * other.data
                if grad_self.shape != self.data.shape:
                    grad_self = _unbroadcast(grad_self, self.data.shape)
                self.grad += grad_self
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data, dtype=np.float32)
                grad_other = out.grad * self.data
                if grad_other.shape != other.data.shape:
                    grad_other = _unbroadcast(grad_other, other.data.shape)
                other.grad += grad_other

        out._backward = _backward
        return out

    def __rmul__(self, other: Any) -> "Tensor":
        return self.__mul__(other)

    def __matmul__(self, other: Any) -> "Tensor":
        """
        Matrix multiplication: self @ other
        """
        other = _ensure_tensor(other)
        out = Tensor(self.data @ other.data,
                     requires_grad=self.requires_grad or other.requires_grad)
        out._prev = {self, other}
        out._op = "matmul"

        if not out.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data, dtype=np.float32)
                self.grad += g @ other.data.T
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data, dtype=np.float32)
                other.grad += self.data.T @ g

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- reductions

    def mean(self) -> "Tensor":
        """
        Scalar mean of all elements.
        """
        out = Tensor(self.data.mean(), requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "mean"

        if not self.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=np.float32)
            # out.grad is scalar; distribute it evenly
            self.grad += out.grad / self.data.size

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- softmax (as you already had)

    def softmax(self, axis: int = -1) -> "Tensor":
        """
        Numerically stable softmax along the given axis.
        Useful for turning logits into probabilities at inference time.
        """
        x = self.data
        shifted = x - x.max(axis=axis, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / exps.sum(axis=axis, keepdims=True)

        out = Tensor(probs, requires_grad=self.requires_grad)
        out._prev = {self}
        out._op = "softmax"

        if not self.requires_grad:
            return out

        def _backward():
            if out.grad is None:
                return
            g = out.grad
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=np.float32)
            s = (g * out.data).sum(axis=axis, keepdims=True)
            self.grad += out.data * (g - s)

        out._backward = _backward
        return out

    # ---------------------------------------------------------------- entrypoint

    def backward(self) -> None:
        """
        Convenience wrapper: tensor.backward() instead of autograd.backward(tensor).
        """
        from . import autograd  # local import to avoid circular dependency
        autograd.backward(self)
