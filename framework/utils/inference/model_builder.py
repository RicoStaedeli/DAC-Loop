from typing import Optional
import logging
import torch
from typing import Dict, Any, Tuple
import torch.nn as nn

from framework.models.ResNet18 import ResNet18
from framework.models import MODEL_REGISTRY

log = logging.getLogger(__name__)

class ModelBuilder:
    """
    This class creates a usable ResNet Model from a given checkpoint.
    The checkpoint needs to contain the model layer information.
    """

    def _get_parent_module(self, root: nn.Module, name: str) -> Tuple[nn.Module, str]:
        """
        Given a dotted module name like 'layer1.0.conv1', return (parent_module, 'conv1').
        """
        parts = name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    # -------------------- Models -------------------- #
    def build_model(self, name: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
        """Build model by name from MODEL_REGISTRY with given kwargs."""
        if not name:
            return None
        try:
            model_cls = MODEL_REGISTRY[name]
        except KeyError as exc:
            raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}") from exc
        return model_cls(**kwargs)

    def build_compressed_resnet18(
            self,
            pruning_cfg: Dict[str, Dict[str, Any]],
            input_channels: int,
            num_classes: int,
            feature_dim: int = 512,
    ) -> ResNet18:
        """
        Rebuild a structurally-pruned ResNet18 using a pruning_cfg describing each Conv2d/Linear.

        Args:
            pruning_cfg: dict mapping module name -> config
            input_channels: input channels for the first conv (typically 3)
            num_classes: number of output classes
            feature_dim: KD feature dim; mostly irrelevant because kd_proj is overridden if present in cfg

        Returns:
            ResNet18 instance whose module shapes match the pruned checkpoint.
        """
        # 1) Start from vanilla architecture
        model = ResNet18(
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )

        # 2) Resize Conv2d and Linear modules according to pruning_cfg
        for name, layer_cfg in pruning_cfg.items():
            layer_type = layer_cfg.get("type")
            parent, attr_name = self._get_parent_module(model, name)

            if layer_type == "conv2d":
                # Existing module (we only use it to check type / reuse some hyperparams if needed)
                old: nn.Conv2d = getattr(parent, attr_name)
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

            else:
                # Ignore anything else just in case
                continue

        # 3) Rebuild BatchNorm layers to follow Conv out_channels
        #
        # We infer BN shapes purely from conv out_channels because BN is not stored in pruning_cfg.
        # Naming rules for your ResNet18:
        #   - top conv:    conv1 -> bn1
        #   - block convs: layerX.Y.conv1 -> layerX.Y.bn1
        #                  layerX.Y.conv2 -> layerX.Y.bn2
        #   - shortcuts:   layerX.Y.shortcut.0 -> layerX.Y.shortcut.1
        bn_targets: Dict[str, int] = {}

        for name, layer_cfg in pruning_cfg.items():
            if layer_cfg.get("type") != "conv2d":
                continue
            out_c = layer_cfg["out_channels"]

            # top-level stem
            if name == "conv1":
                bn_targets["bn1"] = out_c

            # block conv1/conv2
            if name.endswith("conv1"):
                bn_targets[name.replace("conv1", "bn1")] = out_c
            if name.endswith("conv2"):
                bn_targets[name.replace("conv2", "bn2")] = out_c

            # shortcut conv -> shortcut bn
            if name.endswith("shortcut.0"):
                bn_targets[name.replace("shortcut.0", "shortcut.1")] = out_c

        for bn_name, num_features in bn_targets.items():
            try:
                parent, attr_name = self._get_parent_module(model, bn_name)
            except AttributeError:
                # if the name doesn't exist in this model, just skip
                continue

            old_module = getattr(parent, attr_name, None)
            if not isinstance(old_module, nn.BatchNorm2d):
                continue

            new_bn = nn.BatchNorm2d(
                num_features=num_features,
                eps=old_module.eps,
                momentum=old_module.momentum,
                affine=old_module.affine,
                track_running_stats=old_module.track_running_stats,
            )
            parent.__setattr__(attr_name, new_bn)

        return model