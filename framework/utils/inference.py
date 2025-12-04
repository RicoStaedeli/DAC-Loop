from pathlib import Path
from typing import Optional
import logging
import torch
from torchinfo import summary
import time

from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
from framework.models import MODEL_REGISTRY
from framework.utils.device_resolver import DeviceResolver
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf

log = logging.getLogger(__name__)


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

    # If you prefer, you can build a dedicated test_loader here instead of using val_loader.

    # Models
    model_full = _build_model("ResNet18", input_channels=3, num_classes=45)
    model_compressed = _build_model("ResNet18", input_channels=3, num_classes=45)

    print_model_summary(model_full, input_channels=3, bs=256, img_size=256, description="Baseline ResNet18")
    print_model_summary(model_compressed, input_channels=3, bs=256, img_size=256, description="Compressed ResNet18")

    # Baseline ResNet18
    path_full = Path("experiments/results/Experiment_5/Run_1/best.pt")

    # Compressed ResNet18
    path_compressed = Path("experiments/results/Experiment_13/Run_1/best.pt")

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
    ckpt = torch.load(str(path_compressed), map_location=device)
    model_compressed.load_state_dict(ckpt["model_state"])

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
    speedup_sample = full_sample_time / comp_sample_time if comp_sample_time > 0 else float("inf")

    log.info("=== Inference Time Comparison on RESISC45 (val split) ===")
    log.info(f"Full model:      {full_batch_time*1e3:.3f} ms / batch | {full_sample_time*1e3:.3f} ms / sample")
    log.info(f"Compressed model:{comp_batch_time*1e3:.3f} ms / batch | {comp_sample_time*1e3:.3f} ms / sample")
    log.info(f"Speedup (batch):  x{speedup_batch:.3f}")
    log.info(f"Speedup (sample): x{speedup_sample:.3f}")

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
    # Setup logging
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info(f"Config loaded from {logger_cfg_path}")


    device = DeviceResolver.get()
    log.info("Using device: %s", device)

    _inference_time_comparison_resisc(device)





if __name__ == "__main__":
    main()
