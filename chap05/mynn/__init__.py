# mynn/__init__.py
"""
Public entry point for the mini neural-network library.

Typical usage:

    from mynn import Tensor, nn, optim, seed, no_grad
"""

from .tensor import Tensor
from . import nn, optim
from .utils import seed, no_grad

__all__ = ["Tensor", "nn", "optim", "seed", "no_grad"]

