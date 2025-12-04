# mynn/utils.py
from __future__ import annotations
from contextlib import contextmanager
import numpy as np

# Global flag to control gradient recording (will be consulted by Tensor ops)
_grad_enabled: bool = True

def seed(value: int) -> None:
    """Set NumPy's RNG seed for reproducible experiments."""
    np.random.seed(value)

@contextmanager
def no_grad():
    """
    Context manager that temporarily disables gradient tracking.
    The Tensor operations will respect this flag once implemented.
    """
    global _grad_enabled
    old = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = old

