# mynn/optim/__init__.py
"""
Optimizers: update rules applied to parameters produced by Modules.
"""

from .sgd import SGD
from .adam import Adam

__all__ = ["SGD", "Adam"]

