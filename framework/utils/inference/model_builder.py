from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import logging

import torch
import torch.nn as nn
from torchinfo import summary

from framework.models import MODEL_REGISTRY

log = logging.getLogger(__name__)


class ModelBuilder:
    """Responsible for building models and loading checkpoints."""

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def _get_parent_module(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
        """Given dotted module name like 'layer1.0.conv1', return (parent_module, 'conv1')."""
        parts = name.split(".")
        parent: nn.Module = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    def build_from_registry(self, name: Optional[str], **kwargs) -> Optional[nn.Module]:
        """Build model by name from MODEL_REGISTRY with given kwargs."""
        if not name:
            return None
        try:
            model_cls = MODEL_REGISTRY[name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
            ) from exc
        return model_cls(**kwargs)

    def load_checkpoint(
        self,
        model: nn.Module,
        ckpt_path: Path,
        state_key: str = "model_state",
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load checkpoint from disk into a model and return the checkpoint dict."""
        ckpt = torch.load(str(ckpt_path), map_location=self.device)
        if state_key not in ckpt:
            raise KeyError(f"Checkpoint at {ckpt_path} does not contain key '{state_key}'.")
        model.load_state_dict(ckpt[state_key], strict=strict)
        return ckpt

    def build_compressed_resnet18(
        self,
        pruning_cfg: Dict[str, Dict[str, Any]],
        input_channels: int,
        num_classes: int,
        feature_dim: int = 512,
    ) -> nn.Module:
        """Build a pruned ResNet18 by rewriting layer modules described in pruning_cfg."""
        # Start from a vanilla architecture and then replace layers.
        model = self.build_from_registry(
            "ResNet18",
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        if model is None:
            raise RuntimeError("Failed to build base ResNet18 from registry.")

        # 1) Replace Conv / Linear layers based on pruning_cfg entries
        for name, layer_cfg in pruning_cfg.items():
            layer_type = layer_cfg.get("type")
            if layer_type not in {"conv2d", "linear"}:
                continue

            parent, attr_name = self._get_parent_module(model, name)

            if layer_type == "conv2d":
                new = nn.Conv2d(
                    in_channels=layer_cfg["in_channels"],
                    out_channels=layer_cfg["out_channels"],
                    kernel_size=tuple(layer_cfg["kernel_size"]),
                    stride=tuple(layer_cfg["stride"]),
                    padding=tuple(layer_cfg["padding"]),
                    dilation=tuple(layer_cfg["dilation"]),
                    groups=layer_cfg["groups"],
                    bias=layer_cfg["bias"],
                )
                parent.__setattr__(attr_name, new)

            elif layer_type == "linear":
                new = nn.Linear(
                    in_features=layer_cfg["in_features"],
                    out_features=layer_cfg["out_features"],
                    bias=layer_cfg["bias"],
                )
                parent.__setattr__(attr_name, new)

        # 2) Rebuild BatchNorm layers to follow Conv out_channels (BN config not stored in pruning_cfg)
        # Naming rules for your ResNet18:
        #   - top conv:    conv1 -> bn1
        #   - block convs: layerX.Y.conv1 -> layerX.Y.bn1
        #                 layerX.Y.conv2 -> layerX.Y.bn2
        #   - shortcuts:   layerX.Y.shortcut.0 -> layerX.Y.shortcut.1
        bn_targets: Dict[str, int] = {}
        for name, layer_cfg in pruning_cfg.items():
            if layer_cfg.get("type") != "conv2d":
                continue
            out_c = layer_cfg["out_channels"]

            if name == "conv1":
                bn_targets["bn1"] = out_c
            else:
                parts = name.split(".")
                # expected formats:
                # layer{n}.{b}.conv{m}  OR  layer{n}.{b}.shortcut.0
                if len(parts) >= 4 and parts[0].startswith("layer"):
                    layer_prefix = ".".join(parts[:-1])  # everything before 'conv1'/'conv2'/'0'
                    last = parts[-1]
                    # conv1/conv2
                    if last.startswith("conv"):
                        bn_name = layer_prefix + ".bn" + last.replace("conv", "")
                        bn_targets[bn_name] = out_c
                    # shortcut.0 -> shortcut.1
                    elif last == "0" and parts[-2] == "shortcut":
                        bn_name = ".".join(parts[:-1]) + ".1"
                        bn_targets[bn_name] = out_c

        for bn_name, num_features in bn_targets.items():
            try:
                parent, attr = self._get_parent_module(model, bn_name)
                old_bn = getattr(parent, attr)
                # Preserve eps/momentum/affine/track_running_stats where possible
                eps = getattr(old_bn, "eps", 1e-5)
                momentum = getattr(old_bn, "momentum", 0.1)
                affine = getattr(old_bn, "affine", True)
                track = getattr(old_bn, "track_running_stats", True)
                new_bn = nn.BatchNorm2d(
                    num_features=num_features,
                    eps=eps,
                    momentum=momentum,
                    affine=affine,
                    track_running_stats=track,
                )
                parent.__setattr__(attr, new_bn)
            except Exception as e:
                log.warning("Failed to rebuild BN layer '%s': %s", bn_name, e)

        return model

    @staticmethod
    def print_model_summary(
        model: Optional[nn.Module],
        input_channels: int,
        bs: int,
        img_size: int,
        description: str,
    ) -> None:
        """Print and persist a torchinfo summary for a model."""
        if model is None:
            return

        model_info = summary(
            model,
            input_size=(bs, input_channels, img_size, img_size),
            col_names=("input_size", "output_size", "num_params", "trainable"),
            depth=7,
            verbose=0,
        )
        header = f"{description}:\n" + "-" * 27 + "\n"
        log.info("\n%s%s", header, model_info)

    @staticmethod
    def file_size_mb(path: Path) -> float:
        return path.stat().st_size / (1024 * 1024)
