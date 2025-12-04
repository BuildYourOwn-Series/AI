# mynn/nn/__init__.py
"""
Neural-network building blocks: Module, Parameter, layers, activations, losses.
"""

from .module import Module, Parameter
from .layers import Linear
from .activations import ReLU, Tanh
from .losses import MSELoss, CrossEntropyLoss

__all__ = [
    "Module", "Parameter",
    "Linear",
    "ReLU", "Tanh",
    "MSELoss", "CrossEntropyLoss",
]
