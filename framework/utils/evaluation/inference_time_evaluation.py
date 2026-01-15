from pathlib import Path
from typing import Optional
import logging
import torch
from torchinfo import summary
import time
from typing import Dict, Any
import json
from thop import profile

from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
from framework.utils.device_resolver import DeviceResolver
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf
from framework.utils.inference.model_builder import ModelBuilder

log = logging.getLogger(__name__)

LOGFILE_PATH = Path("experiments/results/inference_times_resisc45.log")

def save_results(path: Path, results: Dict[str, Any]) -> None:
    """
    Save results dictionary to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)


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

def _compute_macs_flops_params(
    model: torch.nn.Module,
    device: torch.device,
    input_channels: int,
    img_size: int,
    batch_size: int = 1,
) -> Dict[str, float]:
    """
    Compute MACs, FLOPs (2*MACs), and parameter count using thop.

    Notes:
      - thop counts compute based on tensor shapes (dense ops).
      - Unstructured sparsity does NOT reduce MACs/FLOPs here.
      - Structured pruning DOES reduce MACs/FLOPs because shapes change.
    """
    model = model.to(device)
    model.eval()

    x = torch.randn(batch_size, input_channels, img_size, img_size, device=device)

    with torch.no_grad():
        macs, params = profile(model, inputs=(x,), verbose=False)

    flops = 2.0 * macs  # common convention: FLOPs = 2 * MACs (mul + add)
    return {
        "macs": float(macs),
        "flops": float(flops),
        "params": float(params),
        "img_size": float(img_size),
        "batch_size": float(batch_size),
    }

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

    modelBuilder = ModelBuilder()

    # Models
    model_full = modelBuilder.build_model("ResNet18", input_channels=3, num_classes=45)


    # Baseline ResNet18
    path_full = Path("experiments/results/Experiment_1a/Run_1/best.pt")

    # Compressed ResNet18
    path_compressed = Path("experiments/results/Experiment_9a/Run_1/compressed_model_iteration_9.pt")

    # -------------------------
    # Full model timing
    # -------------------------
    ckpt = torch.load(str(path_full), map_location=device)
    model_full.load_state_dict(ckpt["model_state"])
    print_model_summary(model_full, input_channels=3, bs=256, img_size=256, description="Baseline ResNet18")

    # FLOPs / MACs / Params (thop) - use batch_size=1 for per-image reporting
    full_cost = _compute_macs_flops_params(
        model_full,
        device=device,
        input_channels=3,
        img_size=256,
        batch_size=1,
    )
    log.info(
        "Full model compute @%dx%d: MACs=%.3f G | FLOPs=%.3f G | Params=%.3f M",
        int(full_cost["img_size"]), int(full_cost["img_size"]),
        full_cost["macs"] / 1e9,
        full_cost["flops"] / 1e9,
        full_cost["params"] / 1e6,
    )

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
    print_model_summary(model_compressed, input_channels=3, bs=256, img_size=256, description="Compressed ResNet18")

    comp_cost = _compute_macs_flops_params(
        model_compressed,
        device=device,
        input_channels=3,
        img_size=256,
        batch_size=1,
    )
    log.info(
        "Compressed model compute @%dx%d: MACs=%.3f G | FLOPs=%.3f G | Params=%.3f M",
        int(comp_cost["img_size"]), int(comp_cost["img_size"]),
        comp_cost["macs"] / 1e9,
        comp_cost["flops"] / 1e9,
        comp_cost["params"] / 1e6,
    )

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

    full_size_bytes = path_full.stat().st_size
    comp_size_bytes = path_compressed.stat().st_size

    full_size_mb = full_size_bytes / (1024 * 1024)
    comp_size_mb = comp_size_bytes / (1024 * 1024)
    size_reduction = full_size_mb / comp_size_mb if comp_size_mb > 0 else float("inf")

    # Compute reduction (MACs/FLOPs)
    macs_reduction = (full_cost["macs"] / comp_cost["macs"]) if comp_cost["macs"] > 0 else float("inf")
    flops_reduction = (full_cost["flops"] / comp_cost["flops"]) if comp_cost["flops"] > 0 else float("inf")



    log.info("=== FINAL INFERENCE SUMMARY ===")
    log.info("Model full checkpoint:       %s", path_full)
    log.info("Model compressed checkpoint: %s", path_compressed)
    log.info("Device:                      %s", device)
    log.info("Device name:                 %s", device_name)
    log.info("Device capability:           %s", device_cap)
    log.info("Checkpoint (full) size:       %.2f MB", full_size_mb)
    log.info("Checkpoint (compressed) size: %.2f MB", comp_size_mb)
    log.info("Checkpoint size reduction:    x%.3f", size_reduction)
    log.info("Full batch time [s]:         %.6f", full_batch_time)
    log.info("Full sample time [s]:        %.6f", full_sample_time)
    log.info("Compressed batch time [s]:   %.6f", comp_batch_time)
    log.info("Compressed sample time [s]:  %.6f", comp_sample_time)
    log.info("Speedup batch:               x%.6f", speedup_batch)
    log.info("Speedup sample:              x%.6f", speedup_sample)
    log.info("Full MACs [G]:               %.6f", full_cost["macs"] / 1e9)
    log.info("Compressed MACs [G]:         %.6f", comp_cost["macs"] / 1e9)
    log.info("MACs reduction:              x%.6f", macs_reduction)
    log.info("Full FLOPs [G]:              %.6f", full_cost["flops"] / 1e9)
    log.info("Compressed FLOPs [G]:        %.6f", comp_cost["flops"] / 1e9)
    log.info("FLOPs reduction:             x%.6f", flops_reduction)
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
