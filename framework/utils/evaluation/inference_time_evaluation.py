from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import json
import logging
import time

import torch
import torch.nn as nn
from thop import profile

from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf
from framework.utils.device_resolver import DeviceResolver
from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml

from framework.utils.inference.model_builder import ModelBuilder

log = logging.getLogger(__name__)

LOGFILE_PATH = Path("experiments/results/inference_times.log")


def save_results(path: Path, results: Dict[str, Any]) -> None:
    """Save a results dictionary to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)


class InferenceTester:
    """Responsible for building data loaders and running inference benchmarks."""

    def __init__(
        self,
        device: torch.device,
        model_builder: ModelBuilder,
        batch_size: int = 256,
        image_size: int = 256,
        num_workers: int = 2,
    ):
        self.device = device
        self.model_builder = model_builder
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    @staticmethod
    def _measure_inference_time(
        model: nn.Module,
        loader,
        device: torch.device,
        warmup_batches: int = 5,
    ) -> Tuple[float, float]:
        """Return (avg_batch_time_seconds, avg_sample_time_seconds)."""
        model.eval()
        model.to(device)

        # Warm-up
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= warmup_batches:
                    break
                x = x.to(device, non_blocking=True)
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

        # Timed loop
        total_time = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device, non_blocking=True)

                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                total_time += (t1 - t0)
                total_samples += x.size(0)

        num_batches = len(loader)
        avg_batch_time = total_time / max(num_batches, 1)
        avg_sample_time = total_time / max(total_samples, 1)
        return avg_batch_time, avg_sample_time

    @staticmethod
    def _compute_macs_flops_params(
        model: nn.Module,
        device: torch.device,
        input_channels: int,
        img_size: int,
        batch_size: int = 1,
    ) -> Dict[str, float]:
        """Compute MACs, FLOPs (2*MACs), and parameter count using thop."""
        model.eval().to(device)
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

    def _build_resisc45_val_loader(self):
        train_ds, val_ds, _ = load_hf_splits("timm/resisc45")
        _, val_loader = build_loaders_from_hf(
            train_ds,
            val_ds,
            batch_size=self.batch_size,
            image_size=self.image_size,
            indices=None,
            augment=False,
            normalize=True,
            stats_samples=0,
            selected_bands=["B01", "B02", "B03"],
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def compare_resisc45(
        self,
        full_ckpt_path: Path,
        compressed_ckpt_path: Path,
        input_channels: int = 3,
        num_classes: int = 45,
        feature_dim: int = 512,
        results_json_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Benchmark baseline vs compressed model on RESISC45 val split."""
        val_loader = self._build_resisc45_val_loader()

        # -------------------------
        # Full model
        # -------------------------
        model_full = self.model_builder.build_from_registry(
            "ResNet18", input_channels=input_channels, num_classes=num_classes
        )
        if model_full is None:
            raise RuntimeError("Failed to build baseline ResNet18 from registry.")

        ckpt_full = self.model_builder.load_checkpoint(model_full, full_ckpt_path)
        self.model_builder.print_model_summary(
            model_full,
            input_channels=input_channels,
            bs=self.batch_size,
            img_size=self.image_size,
            description="Baseline ResNet18",
        )

        full_cost = self._compute_macs_flops_params(
            model_full,
            device=self.device,
            input_channels=input_channels,
            img_size=self.image_size,
            batch_size=1,
        )
        log.info(
            "Full model compute @%dx%d: MACs=%.3f G | FLOPs=%.3f G | Params=%.3f M",
            int(full_cost["img_size"]), int(full_cost["img_size"]),
            full_cost["macs"] / 1e9,
            full_cost["flops"] / 1e9,
            full_cost["params"] / 1e6,
        )

        full_batch_time, full_sample_time = self._measure_inference_time(
            model_full, val_loader, self.device
        )

        # Free memory
        del model_full
        del ckpt_full
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # -------------------------
        # Compressed model
        # -------------------------
        ckpt_comp = torch.load(str(compressed_ckpt_path), map_location=self.device)
        pruning_cfg = ckpt_comp.get("pruning_cfg")
        if pruning_cfg is None:
            raise KeyError(
                f"Compressed checkpoint at {compressed_ckpt_path} is missing 'pruning_cfg'."
            )

        model_compressed = self.model_builder.build_compressed_resnet18(
            pruning_cfg=pruning_cfg,
            input_channels=input_channels,
            num_classes=num_classes,
            feature_dim=feature_dim,
        )
        if "model_state" not in ckpt_comp:
            raise KeyError(
                f"Compressed checkpoint at {compressed_ckpt_path} is missing 'model_state'."
            )
        model_compressed.load_state_dict(ckpt_comp["model_state"], strict=True)

        self.model_builder.print_model_summary(
            model_compressed,
            input_channels=input_channels,
            bs=self.batch_size,
            img_size=self.image_size,
            description="Compressed ResNet18",
        )

        comp_cost = self._compute_macs_flops_params(
            model_compressed,
            device=self.device,
            input_channels=input_channels,
            img_size=self.image_size,
            batch_size=1,
        )
        log.info(
            "Compressed model compute @%dx%d: MACs=%.3f G | FLOPs=%.3f G | Params=%.3f M",
            int(comp_cost["img_size"]), int(comp_cost["img_size"]),
            comp_cost["macs"] / 1e9,
            comp_cost["flops"] / 1e9,
            comp_cost["params"] / 1e6,
        )

        comp_batch_time, comp_sample_time = self._measure_inference_time(
            model_compressed, val_loader, self.device
        )

        # Free memory
        del model_compressed
        del ckpt_comp
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        # -------------------------
        # Comparison & logging
        # -------------------------
        speedup_batch = full_batch_time / comp_batch_time if comp_batch_time > 0 else float("inf")
        speedup_sample = full_sample_time / comp_sample_time if full_sample_time > 0 else float("inf")

        log.info("=== Inference Time Comparison on RESISC45 (val split) ===")
        log.info(
            "Full model:       %.3f ms / batch | %.3f ms / sample",
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
        # Summary stats
        # -------------------------
        device_name = "N/A"
        device_cap = "N/A"
        if self.device.type == "cuda":
            try:
                device_name = torch.cuda.get_device_name(self.device)
                device_cap = torch.cuda.get_device_capability(self.device)
            except Exception as e:
                log.warning("Could not query CUDA device properties for summary: %s", e)

        full_size_mb = self.model_builder.file_size_mb(full_ckpt_path)
        comp_size_mb = self.model_builder.file_size_mb(compressed_ckpt_path)
        size_reduction = (full_size_mb / comp_size_mb) if comp_size_mb > 0 else float("inf")

        macs_reduction = (full_cost["macs"] / comp_cost["macs"]) if comp_cost["macs"] > 0 else float("inf")
        flops_reduction = (full_cost["flops"] / comp_cost["flops"]) if comp_cost["flops"] > 0 else float("inf")
        params_reduction = (full_cost["params"] / comp_cost["params"]) if comp_cost["params"] > 0 else float("inf")

        results = {
            "device": {
                "type": self.device.type,
                "name": device_name,
                "capability": device_cap,
            },
            "paths": {
                "full": str(full_ckpt_path),
                "compressed": str(compressed_ckpt_path),
            },
            "timing": {
                "full": {"batch_s": full_batch_time, "sample_s": full_sample_time},
                "compressed": {"batch_s": comp_batch_time, "sample_s": comp_sample_time},
                "speedup": {"batch": speedup_batch, "sample": speedup_sample},
            },
            "compute": {
                "full": full_cost,
                "compressed": comp_cost,
                "reduction": {
                    "macs": macs_reduction,
                    "flops": flops_reduction,
                    "params": params_reduction,
                },
            },
            "checkpoint_size_mb": {
                "full": full_size_mb,
                "compressed": comp_size_mb,
                "reduction": size_reduction,
            },
            "settings": {
                "batch_size": self.batch_size,
                "image_size": self.image_size,
                "num_workers": self.num_workers,
            },
        }

        if results_json_path is not None:
            save_results(results_json_path, results)

        return results


def main() -> None:
    # Logger config
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)

    # Also log to a file
    LOGFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOGFILE_PATH)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
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
            log.info("CUDA capability: %s", torch.cuda.get_device_capability(device))
        except Exception as e:
            log.warning("Could not query CUDA device properties: %s", e)

    builder = ModelBuilder(device=device)
    tester = InferenceTester(
        device=device,
        model_builder=builder,
        batch_size=256,
        image_size=256,
        num_workers=2,
    )

    # Default paths (same as original script)
    full_path = Path("experiments/results/Experiment_1a/Run_1/best.pt")
    compressed_path = Path("experiments/results/Experiment_9a/Run_1/compressed_model_iteration_9.pt")

    tester.compare_resisc45(
        full_ckpt_path=full_path,
        compressed_ckpt_path=compressed_path,
        results_json_path=Path("experiments/results/inference_times_resisc45.json"),
    )

    log.info("Finished inference time comparison script")


if __name__ == "__main__":
    main()
