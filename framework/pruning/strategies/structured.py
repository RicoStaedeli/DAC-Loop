# strategies/structured.py
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch_pruning as tp

from framework.pruning.services.pruning_utils import prunable_layers


class StructuredPruning:
    """
    Structured pruning strategy that prunes entire channels from convolutional layers
    based on L2 norm. It uses a dependency graph to ensure that pruning is consistent
    across the model.
    """

    def __init__(self, amount: float, dim: int, layer_prune_ratio: float, min_channels: int):
        """
        Args:
            amount (float): The fraction of weights to prune in each convolutional layer.
            dim (int): The dimension along which to prune (0 for output channels, 1 for input channels).
            layer_prune_ratio (float): The fraction of channels to prune from each layer.
            min_channels (int): The minimum number of channels to retain in each layer.
        """
        self.amount = amount
        self.dim = dim
        self.layer_prune_ratio = layer_prune_ratio
        self.min_channels = min_channels
        self._DG = None

    def prepare(self, model: nn.Module) -> None:
        # apply masks first (ln_structured)
        for _, m in prunable_layers(model):
            if isinstance(m, nn.Conv2d):
                prune.ln_structured(m, name="weight", amount=self.amount, n=2, dim=self.dim)

    def build_dependency(self, model: nn.Module, example_input: torch.Tensor):
        self._DG = tp.DependencyGraph().build_dependency(model, example_inputs=example_input)

    def apply(self, model: nn.Module) -> None:
        if self._DG is None:
            raise RuntimeError("Dependency graph not built. Call build_dependency(...) first.")
        # remove masks before surgery to operate on dense weights
        for _, m in prunable_layers(model):
            try:
                prune.remove(m, "weight")
            except Exception:
                pass

        for m in list(model.modules()):
            if isinstance(m, nn.Conv2d):
                out_c = m.out_channels
                if out_c <= self.min_channels:
                    continue
                n_prune = max(0, int(out_c * self.layer_prune_ratio))
                if out_c - n_prune < self.min_channels:
                    n_prune = max(0, out_c - self.min_channels)
                if n_prune == 0:
                    continue
                # simplest heuristic: prune smallest-L2 output channels
                W = m.weight.detach()
                l2 = W.pow(2).sum(dim=(1, 2, 3))
                idx = torch.argsort(l2)[:n_prune].tolist()
                group = self._DG.get_pruning_group(m, tp.prune_conv_out_channels, idx)
                group.prune()

    def finalize(self, model: nn.Module) -> None:
        # clean any lingering buffers
        for _, m in prunable_layers(model):
            if hasattr(m, "weight_mask"):
                try:
                    prune.remove(m, "weight")
                except Exception:
                    pass
