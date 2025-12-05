from pathlib import Path
from typing import Optional
import logging
import torch
from torchinfo import summary
import time
from typing import Dict, Any, Tuple
import torch.nn as nn
import json

from framework.models.ResNet18 import ResNet18
from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
from framework.models import MODEL_REGISTRY
from framework.utils.device_resolver import DeviceResolver
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf

log = logging.getLogger(__name__)

LOGFILE_PATH = Path("experiments/results/inference_times_resisc45.log")

def save_results(path: Path, results: Dict[str, Any]) -> None:
    """
    Save results dictionary to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)


def _get_parent_module(root: nn.Module, name: str) -> Tuple[nn.Module, str]:
    """
    Given a dotted module name like 'layer1.0.conv1', return (parent_module, 'conv1').
    """
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
# -------------------- Models -------------------- #
def _build_model(name: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    """Build model by name from MODEL_REGISTRY with given kwargs."""
    if not name:
        return None
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}") from exc
    return model_cls(**kwargs)

def _build_compressed_resnet18(
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
        parent, attr_name = _get_parent_module(model, name)

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
            old: nn.Linear = getattr(parent, attr_name)
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
            parent, attr_name = _get_parent_module(model, bn_name)
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


def _measure_inference_time(model: torch.nn.Module,
                            loader,
                            device: torch.device,
                            num_warmup_batches: int = 5) -> tuple[float, float]:
    """
    Measure inference time on `loader`.

    Returns:
        avg_batch_time (float): seconds per batch
        avg_sample_time (float): seconds per sample
    """
    model.to(device)
    model.eval()

    total_time = 0.0
    total_samples = 0

    # --- Warm-up (not timed) ---
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_warmup_batches:
                break
            x = x.to(device, non_blocking=True)
            _ = model(x)

    # --- Timed pass ---
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.time()

            _ = model(x)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.time()

            batch_time = end - start
            total_time += batch_time
            total_samples += x.size(0)

    num_batches = len(loader)
    avg_batch_time = total_time / max(num_batches, 1)
    avg_sample_time = total_time / max(total_samples, 1)

    return avg_batch_time, avg_sample_time


def _inference_time_comparison_resisc(device: torch.device) -> None:
    # Data
    train_ds, val_ds, test_ds = load_hf_splits("timm/resisc45")
    train_loader, val_loader = build_loaders_from_hf(
        train_ds,
        val_ds,
        batch_size=256,
        image_size=256,
        indices=None,
        augment=False,
        normalize=True,
        stats_samples=0,
        selected_bands=["B01", "B02", "B03"],
        num_workers=2,
        pin_memory=True,
    )

    # Models
    model_full = _build_model("ResNet18", input_channels=3, num_classes=45)
    model_compressed = _build_model("ResNet18", input_channels=3, num_classes=45)

    print_model_summary(model_full, input_channels=3, bs=256, img_size=256, description="Baseline ResNet18")
    print_model_summary(model_compressed, input_channels=3, bs=256, img_size=256, description="Compressed ResNet18")

    # Baseline ResNet18
    path_full = Path("experiments/results/Experiment_5/Run_1/best.pt")

    # Compressed ResNet18
    path_compressed = Path("experiments/results/Experiment_9/Run_9/compressed_model_iteration_9.pt")

    # -------------------------
    # Full model timing
    # -------------------------
    ckpt = torch.load(str(path_full), map_location=device)
    model_full.load_state_dict(ckpt["model_state"])

    full_batch_time, full_sample_time = _measure_inference_time(
        model_full, val_loader, device
    )

    # Free memory
    del model_full
    del ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Compressed model timing
    # -------------------------
    ckpt = torch.load(path_compressed, map_location=device)
    pruning_cfg = ckpt["pruning_cfg"]

    model_compressed = _build_compressed_resnet18(
        pruning_cfg=pruning_cfg,
        input_channels=3,
        num_classes=45,
        feature_dim=512,
    )

    model_compressed.load_state_dict(ckpt["model_state"], strict=True)

    comp_batch_time, comp_sample_time = _measure_inference_time(
        model_compressed, val_loader, device
    )

    del model_compressed
    del ckpt
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Comparison & logging
    # -------------------------
    speedup_batch = full_batch_time / comp_batch_time if comp_batch_time > 0 else float("inf")
    speedup_sample = full_sample_time / comp_sample_time if full_sample_time > 0 else float("inf")

    log.info("=== Inference Time Comparison on RESISC45 (val split) ===")
    log.info(
        "Full model:      %.3f ms / batch | %.3f ms / sample",
        full_batch_time * 1e3,
        full_sample_time * 1e3,
    )
    log.info(
        "Compressed model: %.3f ms / batch | %.3f ms / sample",
        comp_batch_time * 1e3,
        comp_sample_time * 1e3,
    )
    log.info("Speedup (batch):  x%.3f", speedup_batch)
    log.info("Speedup (sample): x%.3f", speedup_sample)

    # -------------------------
    # Final summary block in logfile (not JSON)
    # -------------------------
    # Device info for the summary
    device_name = "N/A"
    device_cap = "N/A"
    if device.type == "cuda":
        try:
            device_name = torch.cuda.get_device_name(device)
            device_cap = torch.cuda.get_device_capability(device)
        except Exception as e:
            log.warning("Could not query CUDA device properties for summary: %s", e)

    log.info("=== FINAL INFERENCE SUMMARY ===")
    log.info("Model full checkpoint:       %s", path_full)
    log.info("Model compressed checkpoint: %s", path_compressed)
    log.info("Device:                      %s", device)
    log.info("Device name:                 %s", device_name)
    log.info("Device capability:           %s", device_cap)
    log.info("Full batch time [s]:         %.6f", full_batch_time)
    log.info("Full sample time [s]:        %.6f", full_sample_time)
    log.info("Compressed batch time [s]:   %.6f", comp_batch_time)
    log.info("Compressed sample time [s]:  %.6f", comp_sample_time)
    log.info("Speedup batch:               x%.6f", speedup_batch)
    log.info("Speedup sample:              x%.6f", speedup_sample)
    log.info("Logfile written to:          %s", LOGFILE_PATH)


def print_model_summary(model: Optional[torch.nn.Module], input_channels: int, bs, img_size,
                        description: str) -> None:
    """Print and persist a torchinfo summary for a model."""
    if model is None:
        return

    img = img_size
    model_info = summary(
        model,
        input_size=(bs, input_channels, img, img),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=7,
        verbose=0,
    )

    header = f"{description}:\n" + "-" * 27 + "\n"
    text = f"{header}{model_info}\n\n"
    log.info(text)

def main():
    # Setup logging (console etc. via your existing config)
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)

    # Add file handler so ALL logs are written to LOGFILE_PATH
    LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGFILE_PATH, mode="a")  # append
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logging.getLogger().addHandler(file_handler)

    log.info("===================================================")
    log.info("Starting inference time comparison script")
    log.info("Config loaded from %s", logger_cfg_path)

    device = DeviceResolver.get()
    log.info("Using device: %s", device)

    if device.type == "cuda":
        try:
            log.info("CUDA device index: %s", device.index)
            log.info("CUDA device name: %s", torch.cuda.get_device_name(device))
            log.info(
                "CUDA capability: %s",
                torch.cuda.get_device_capability(device),
            )
        except Exception as e:
            log.warning("Could not query CUDA device properties: %s", e)


    log.info("Using device: %s", device)

    _inference_time_comparison_resisc(device)

    log.info("Finished inference time comparison script")



if __name__ == "__main__":
    main()
