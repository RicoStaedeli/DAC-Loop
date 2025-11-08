import torch
import torch.nn.utils.prune as prune
import torch_pruning as tp

import logging

log = logging.getLogger(__name__)


def _strip_module_prefix(sd: dict) -> dict:
    if sd and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


class PruningManager:
    def __init__(self, model, prefix_path, ckpt_path: str | None = None, strict: bool = True):
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
            ckpt = torch.load(f"{prefix_path}{ckpt_path}", map_location=device)
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

    def save_pruned_model(self, path: str):
        """Save the pruned model to the specified path."""
        torch.save(self.model.state_dict(), path)
        log.info(f"Pruned model saved to {path}")

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

    def build_dependency(self, example_input: torch.Tensor):
        """
        Build a dependency graph for safe channel deletion.
        Call this after loading weights and before surgery.
        """
        self.model.eval()
        self.example_input = example_input
        self.DG = tp.DependencyGraph().build_dependency(self.model, example_inputs=example_input)

    def _filters_kept_from_mask(self, conv: torch.nn.Conv2d):
        """
        If you previously ran ln_structured(..., dim=0), use its mask to decide which
        output channels (filters) survive. Fall back to L2 if no mask exists.
        """
        W = conv.weight.detach()
        if hasattr(conv, "weight_mask"):
            mask = conv.weight_mask.detach()
            # a filter survives if any weight in that filter is non-zero by mask
            keep = (mask.view(mask.size(0), -1).any(dim=1)).nonzero(as_tuple=False).view(-1)
            if keep.numel() > 0:
                return keep

        # Fallback: keep by largest-L2 filters
        scores = W.pow(2).sum(dim=(1, 2, 3))
        # By default: keep all (no-op); caller can slice this further if needed
        keep = torch.argsort(scores, descending=True)
        return keep

    def _pick_prune_indices(self, conv: torch.nn.Conv2d, prune_ratio: float):
        """
        Decide which output channels to DELETE in this conv, using mask if present,
        else magnitude. Returns a Python list of indices to prune.
        """
        out_c = conv.out_channels
        n_prune = int(out_c * prune_ratio)
        if n_prune <= 0:
            return []

        W = conv.weight.detach()
        if hasattr(conv, "weight_mask"):
            mask = conv.weight_mask.detach()
            # channels with all-zero mask are good prune candidates
            dead = (~mask.view(out_c, -1).any(dim=1)).nonzero(as_tuple=False).view(-1)
            # if we have enough "dead" channels, prune those; else fill by L2
            if dead.numel() >= n_prune:
                return dead[:n_prune].tolist()
            # fill with smallest-L2 among the rest
            alive = torch.tensor([i for i in range(out_c) if i not in set(dead.tolist())], device=W.device)
            l2 = W[alive].pow(2).sum(dim=(1, 2, 3))
            fill = alive[torch.argsort(l2)[: (n_prune - dead.numel())]]
            return torch.cat([dead, fill]).tolist()
        else:
            # pure magnitude-based
            l2 = W.pow(2).sum(dim=(1, 2, 3))
            idxs = torch.argsort(l2)[:n_prune]
            return idxs.tolist()

    def structured_surgery(self, layer_prune_ratio: float = 0.3, min_channels: int = 8):
        """
        Physically delete channels (filters) in Conv2d layers across the network.
        - layer_prune_ratio: fraction of OUT channels to delete per Conv2d.
        - min_channels: never prune a layer below this width (safety).
        Requires: self.DG built via build_dependency(example_input)
        """
        assert hasattr(self, "DG"), "Call build_dependency(example_input) before structured_surgery()."

        self.model.eval()
        # Iterate over modules; create and run pruning plans
        for m in list(self.model.modules()):
            if isinstance(m, torch.nn.Conv2d):
                out_c = m.out_channels
                if out_c <= min_channels:
                    continue  # keep very narrow layers intact

                idxs_to_prune = self._pick_prune_indices(m, prune_ratio=layer_prune_ratio)
                # Ensure we don’t go under min_channels
                if out_c - len(idxs_to_prune) < min_channels:
                    keep = out_c - min_channels
                    idxs_to_prune = idxs_to_prune[: max(0, keep)]

                if len(idxs_to_prune) == 0:
                    continue

                # Build and execute a dependency-aware plan to prune OUT channels of this conv
                group = self.DG.get_pruning_group(m, tp.prune_conv_out_channels, idxs_to_prune)
                group.prune()

        # After surgery, all shapes are changed; remove any lingering pruning reparam
        self._clean_after_surgery()

    def _clean_after_surgery(self):
        """
        Remove any pruning buffers/reparams and sanity check.
        """
        for _, module in self._get_prunable_layers():
            if hasattr(module, "weight_mask"):
                try:
                    prune.remove(module, "weight")
                except Exception:
                    pass

    # Optional: utility to report "true" parameter count after surgery
    def count_params(self, trainable_only: bool = False):
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
