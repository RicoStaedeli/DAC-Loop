import torch
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import logging

log = logging.getLogger(__name__)


def _strip_module_prefix(sd: dict) -> dict:
    if sd and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd

class PruningManager:
    def __init__(self, model,  ckpt_path: str | None = None, strict: bool = True):
        """
        Args:
            model (nn.Module): The model to prune.
        """
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.model = model
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            # accept various formats
            if isinstance(ckpt, dict) and "model_state" in ckpt:
                sd = ckpt["model_state"]
            elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                sd = ckpt["state_dict"]
            elif isinstance(ckpt, dict) and all(hasattr(v, "shape") for v in ckpt.values()):
                sd = ckpt  # already a state_dict
            else:
                raise RuntimeError(
                    "Checkpoint does not contain a model state_dict. "
                    "Expected keys like 'model_state' or 'state_dict'."
                )
            sd = _strip_module_prefix(sd)
            self.model.load_state_dict(sd, strict=strict)

    def get_model(self):
        """Return the pruned model."""
        return self.model

    def _get_prunable_layers(self):
        """Return all convolutional and linear layers that can be pruned."""
        prunable_layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                prunable_layers.append((name, module))
        return prunable_layers

    def apply_global_pruning(self, amount=0.3, method="l1_unstructured"):
        """
        Apply global unstructured pruning across all conv and linear layers.

        Args:
            amount (float): Fraction of connections to prune globally.
            method (str): "l1_unstructured" or "random_unstructured".
        """
        parameters_to_prune = []
        for _, module in self._get_prunable_layers():
            parameters_to_prune.append((module, "weight"))

        if method == "l1_unstructured":
            prune.global_unstructured(
                parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount
            )
        elif method == "random_unstructured":
            prune.global_unstructured(
                parameters_to_prune, pruning_method=prune.RandomUnstructured, amount=amount
            )
        else:
            raise ValueError("Unsupported pruning method.")

    def apply_layerwise_pruning(self, amount=0.3, method="l1_unstructured"):
        """
        Apply unstructured pruning layer by layer.

        Args:
            amount (float): Fraction of weights to prune in each layer.
            method (str): "l1_unstructured" or "random_unstructured".
        """
        for name, module in self._get_prunable_layers():
            if method == "l1_unstructured":
                prune.l1_unstructured(module, name="weight", amount=amount)
            elif method == "random_unstructured":
                prune.random_unstructured(module, name="weight", amount=amount)
            else:
                raise ValueError("Unsupported pruning method.")
            print(f"Pruned {name}")

    def apply_structured_pruning(self, amount=0.3, dim=0):
        """
        Apply structured pruning (entire channels/filters).

        Args:
            amount (float): Fraction of channels/filters to prune.
            dim (int): Dimension to prune along (0 for output channels, 1 for input channels).
        """
        for name, module in self._get_prunable_layers():
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=amount, n=2, dim=dim)
                print(f"Structured prune: {name}")

    def remove_pruning_reparam(self):
        """Remove pruning reparametrizations to make pruning permanent."""
        for name, module in self._get_prunable_layers():
            try:
                prune.remove(module, "weight")
            except ValueError:
                # Layer might not be pruned
                pass

    def report_sparsity(self):
        """Print and return overall and per-layer sparsity."""
        total_zeros = 0
        total_params = 0
        log.info("\nLayer-wise sparsity report:")
        for name, module in self._get_prunable_layers():
            weight = module.weight.detach()
            zeros = torch.sum(weight == 0).item()
            params = weight.numel()
            sparsity = 100.0 * zeros / params
            total_zeros += zeros
            total_params += params
            log.info(f"{name}: {sparsity:.2f}% sparse")

        overall_sparsity = 100.0 * total_zeros / total_params
        log.info(f"\nOverall sparsity: {overall_sparsity:.2f}%")
        return overall_sparsity