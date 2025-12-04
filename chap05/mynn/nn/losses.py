# mynn/nn/losses.py
from __future__ import annotations
import numpy as np
from .module import Module
from ..tensor import Tensor

class MSELoss(Module):
    """Mean-square error loss."""
    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        # L = mean( (prediction - target)^2 )
        diff = prediction - target
        return (diff * diff).mean()


class CrossEntropyLoss(Module):
    """
    Cross-entropy loss for classification.

    Expects:
      - logits: Tensor of shape (N, C)
      - target: Tensor of shape (N, C) with one-hot rows

    We operate directly on logits (no softmax layer), using a
    numerically stable log-softmax under the hood.
    """
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits: (N, C), target: (N, C) one-hot
        x = logits.data  # NumPy array
        y = target.data

        # log-softmax(x) in a stable way
        # shifted: x - max(x)
        shifted = x - x.max(axis=1, keepdims=True)
        logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
        log_probs = shifted - logsumexp  # (N, C)

        # per-example loss: -sum_c y_{nc} * log_probs_{nc}
        per_example = -(y * log_probs).sum(axis=1)  # (N,)
        loss_value = per_example.mean()             # scalar

        out = Tensor(loss_value, requires_grad=logits.requires_grad)
        out._prev = {logits, target}
        out._op = "cross_entropy"

        if not logits.requires_grad:
            return out

        # for backprop: dL/dlogits = (softmax(logits) - y) / N
        N = x.shape[0]
        # reuse log_probs to get softmax probabilities
        probs = np.exp(log_probs)  # (N, C)

        def _backward():
            if out.grad is None:
                return
            g = out.grad  # scalar gradient dL/d(out); usually 1
            if logits.grad is None:
                logits.grad = np.zeros_like(x, dtype=np.float32)
            # broadcast scalar g over the batch
            logits.grad += g * (probs - y) / N
            # we do not propagate gradients into target (labels)

        out._backward = _backward
        return out
