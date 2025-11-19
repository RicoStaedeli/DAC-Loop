# ports.py
from typing import Protocol, Iterable, Tuple

import torch
import torch.nn as nn


class PruningStrategy(Protocol):
    """
    Protocol for pruning strategies.
    """
    def prepare(self, model: nn.Module) -> None: ...

    def apply(self, model: nn.Module) -> None: ...

    def finalize(self, model: nn.Module) -> None: ...