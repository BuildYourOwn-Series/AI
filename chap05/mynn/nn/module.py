# mynn/nn/module.py
from __future__ import annotations
from collections import OrderedDict
from typing import Iterator, Dict, Any, Callable
import numpy as np

from ..tensor import Tensor

class Parameter(Tensor):
    """A Tensor that is meant to be learned by optimization."""
    def __init__(self, data: Any):
        super().__init__(data, requires_grad=True)


class Module:
    """
    Base class for all trainable components.

    Responsibilities:
      - register child modules (for composition),
      - expose an iterator over parameters(),
      - support named_parameters() and simple querying,
      - provide state_dict() / load_state_dict() and NPZ save/load helpers.
    """
    def __init__(self) -> None:
        # use object.__setattr__ to avoid going through our __setattr__
        object.__setattr__(self, "_children", OrderedDict())

    # ------------------------------------------------------------------ wiring / registration

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Automatically register Modules assigned as attributes as children.
        Parameters do not need explicit registration here; they live inside
        child modules and are discovered via parameters().
        """
        if isinstance(value, Module) and name != "_children":
            # register as child module
            self._children[name] = value
        object.__setattr__(self, name, value)

    def add_module(self, name: str, module: Module) -> Module:
        """
        Explicit registration helper (rarely needed now that __setattr__
        auto-registers child modules).
        """
        self._children[name] = module
        object.__setattr__(self, name, module)
        return module

    def children(self) -> Iterator[Module]:
        return iter(self._children.values())

    def modules(self) -> Iterator[Module]:
        """Iterate over this module and all descendants."""
        yield self
        for m in self.children():
            yield from m.modules()

    # -------------------------------------------------------- parameter access

    def _named_members(self):
        """Yield (name, value) for direct attributes that are Parameters."""
        for name, val in self.__dict__.items():
            if isinstance(val, Parameter):
                yield name, val

    def parameters(self) -> Iterator[Parameter]:
        """Iterate over all Parameters owned by this module and its children."""
        for _, p in self._named_members():
            yield p
        for m in self.children():
            yield from m.parameters()

    def named_parameters(self, prefix: str = ""):
        """Yield (qualified_name, Parameter) pairs."""
        for name, p in self._named_members():
            yield prefix + name, p
        for cname, child in self._children.items():
            child_prefix = f"{prefix}{cname}."
            for name, p in child.named_parameters(child_prefix):
                yield name, p

    # --------------------------------------------------------------- querying

    def find(self, pred: Callable[[Module], bool]):
        return [m for m in self.modules() if pred(m)]

    # ------------------------------------------------------------ state I/O

    def state_dict(self, prefix: str = "") -> "OrderedDict[str, np.ndarray]":
        sd: "OrderedDict[str, np.ndarray]" = OrderedDict()
        for name, p in self.named_parameters(prefix):
            sd[name] = p.data.copy()
        return sd

    def load_state_dict(self, sd: Dict[str, np.ndarray]) -> None:
        for name, p in self.named_parameters():
            if name not in sd:
                raise KeyError(f"missing parameter in state_dict: {name}")
            arr = sd[name]
            if p.data.shape != arr.shape:
                raise ValueError(
                    f"shape mismatch for {name}: expected {p.data.shape}, got {arr.shape}"
                )
            p.data[...] = arr

    def save_npz(self, path: str) -> None:
        np.savez(path, **self.state_dict())

    def load_npz(self, path: str) -> None:
        with np.load(path, allow_pickle=False) as z:
            sd = {k: z[k] for k in z.files}
        self.load_state_dict(sd)

    # --------------------------------------------------------------- forward

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward()")
