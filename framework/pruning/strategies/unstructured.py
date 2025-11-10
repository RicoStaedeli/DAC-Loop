# strategies/unstructured.py
import torch.nn.utils.prune as prune
import torch.nn as nn
from typing import Literal

from framework.pruning.services.pruning_utils import prunable_layers

class UnstructuredPruning:
    def __init__(self, amount: float, method: Literal["l1", "random"]="l1"):
        self.amount = amount
        self.method = method

    def prepare(self, model: nn.Module) -> None:
        pass

    def apply(self, model: nn.Module) -> None:
        params = [(m, "weight") for _, m in prunable_layers(model)]
        if self.method == "l1":
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=self.amount)
        elif self.method == "random":
            prune.global_unstructured(params, pruning_method=prune.RandomUnstructured, amount=self.amount)
        else:
            raise ValueError("Unknown method")

    def finalize(self, model: nn.Module) -> None:
        # does not remove reparam; caller decides when to make pruning permanent
        pass