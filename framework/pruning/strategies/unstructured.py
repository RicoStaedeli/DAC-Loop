# strategies/unstructured.py
import torch.nn as nn
import torch.nn.utils.prune as prune

from framework.pruning.services.pruning_utils import prunable_layers


class UnstructuredPruning:
    def __init__(self, amount: float, method: str = "l1"):
        """
        Unstructured global pruning strategy.
        Args:
            amount (float): The fraction of connections to prune globally across all prunable layers.
            method (str): The pruning method to use. Either "l1" for L1 unstructured pruning or "random" for random unstructured pruning.
        """
        self.amount = amount
        self.method = method

    def prepare(self, model: nn.Module) -> None:
        pass

    def apply(self, model: nn.Module) -> None:
        """
        Apply unstructured global pruning to all prunable layers in the model. Either L1 or random pruning can be used.
        Args:
            model (nn.Module): The model to be pruned.
        """
        params = [(m, "weight") for _, m in prunable_layers(model)]
        if self.method == "l1":
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=self.amount)
        elif self.method == "random":
            prune.global_unstructured(params, pruning_method=prune.RandomUnstructured, amount=self.amount)
        else:
            raise ValueError("Unknown method")

    def finalize(self, model: nn.Module) -> None:
        """
        Make unstructured pruning permanent by removing the pruning reparam.
        This copies (weight_orig * weight_mask) into a plain `weight` parameter
        and deletes the buffers.
        """
        for _, m in prunable_layers(model):
            # Only remove if the module was pruned (has buffers like weight_mask)
            try:
                if prune.is_pruned(m):
                    prune.remove(m, "weight")
            except Exception:
                # robust to layers that weren't pruned / don't have 'weight'
                pass
