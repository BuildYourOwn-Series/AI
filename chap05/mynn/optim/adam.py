# mynn/optim/adam.py
from __future__ import annotations
from typing import Iterable, Dict
import numpy as np
from ..nn.module import Parameter

class Adam:
    """
    Adam optimizer.

    Keeps per-parameter first and second moment estimates and applies
    bias-corrected updates. This implementation is intentionally minimal
    but mathematically faithful to the original algorithm.
    """

    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps

        # moment buffers
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}
        self.t: int = 0

    def step(self) -> None:
        self.t += 1
        b1, b2, lr, eps = self.beta1, self.beta2, self.lr, self.eps
        t = self.t

        for p in self.params:
            if p.grad is None:
                continue

            pid = id(p)
            g = p.grad

            m = self.m.get(pid)
            v = self.v.get(pid)
            if m is None:
                m = self.m[pid] = np.zeros_like(g)
                v = self.v[pid] = np.zeros_like(g)

            # update biased first and second moments
            m[:] = b1 * m + (1.0 - b1) * g
            v[:] = b2 * v + (1.0 - b2) * (g * g)

            # bias correction
            m_hat = m / (1.0 - b1**t)
            v_hat = v / (1.0 - b2**t)

            # parameter update
            p.data -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

