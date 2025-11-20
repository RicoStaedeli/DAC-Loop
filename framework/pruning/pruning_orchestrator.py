# orchestrator.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import json

import torch
import torch.nn as nn

from framework.pruning.ports import PruningStrategy
from framework.pruning.services.core import maybe_load, save_state_dict
from framework.pruning.services.metrics import evaluate_and_log, summarize, log_sparsity
from framework.training.training_orchestrator import Trainer
from framework.utils.device_resolver import DeviceResolver
from framework.pruning.strategies.structured import StructuredPruning
from framework.pruning.strategies.unstructured import UnstructuredPruning

log = logging.getLogger(__name__)


class PruningOrchestrator:
    def __init__(self, model: nn.Module, ckpt_path: Optional[str] = None, strict: bool = True):
        """
        Initializes the PruningOrchestrator with the model and optional checkpoint loading.
        Args:
            model (nn.Module): The neural network model to be pruned.
            ckpt_path (Optional[str]): Path to the checkpoint file to load model weights from.
            strict (bool): Whether to strictly enforce that the keys in state_dict match the model's keys.
        """
        self.device = DeviceResolver.get()
        self.model = model.to(self.device)
        maybe_load(self.model, ckpt_path, self.device, strict)

    def _get_iteration_cfg(self, cfg, iteration: int):
        """
        Return the effective pruning config for a given iteration:
        schedule entry if present, otherwise the global pruning settings.
        """
        pruning_cfg = cfg.training.pruning

        # build a dict for quick lookup
        schedule_map = {item.iteration: item for item in pruning_cfg.schedule}

        item = schedule_map[iteration]
        parameters = {
            "amount": item.pruning_amount,
            "method": item.method,
            "structured": item.structured,
            "dim": item.dim,
            "layer_prune_ratio": item.layer_prune_ratio,
            "min_channels": item.min_channels,
        }
        log.info(f"Pruning iteration {iteration} config: {parameters}")
        return parameters


    def _build_strategy(self, cfg, iteration: int, input_channels: int) -> PruningStrategy:
        """
        Build the appropriate pruning strategy (structured/unstructured)
        based on the iteration-specific config.
        """
        it_cfg = self._get_iteration_cfg(cfg, iteration)

        if not it_cfg["structured"]:
            return UnstructuredPruning(
                amount=it_cfg["amount"],
                method=it_cfg["method"],
            )

        strategy = StructuredPruning(
            amount=it_cfg["amount"],
            dim=it_cfg["dim"],
            layer_prune_ratio=it_cfg["layer_prune_ratio"],
            min_channels=it_cfg["min_channels"],
        )

        example = torch.randn(
            1,
            input_channels,
            cfg.data.image_size,
            cfg.data.image_size,
            device=self.device,
        )
        strategy.build_dependency(self.model, example)
        return strategy


    def run(
            self,
            cfg,
            train_loader,
            val_loader,
            input_channels: int,
            experiment_dir: Path,
            cfg_path,
            model_teacher: Optional[nn.Module] = None,
    ) -> nn.Module:
        """
        Orchestrates (possibly iterative) pruning: prepares the model, applies pruning,
        fine-tunes the pruned model, and evaluates performance before and after pruning.

        The number of pruning+fine-tuning cycles is controlled by:
            cfg.training.pruning.repeat   (defaults to 1 if missing)

        Args:
            cfg: Configuration object containing training and pruning settings.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            input_channels (int): Number of input channels for the model.
            experiment_dir (Path): Directory to save experiment results.
            cfg_path: Path to the configuration file.
            model_teacher (Optional[nn.Module]): Teacher model for knowledge distillation, if any.
        Returns:
            nn.Module: The pruned and fine-tuned model.
        """
        # Initial evaluation before any pruning
        summarize(cfg, self.model, input_channels, "Student - Before Pruning", experiment_dir)
        evaluate_and_log(self.model, val_loader, self.device, "BEFORE pruning.")
        log_sparsity(self.model)

        # Path for JSONL metrics
        metrics_path = experiment_dir / "metrics_pruning.jsonl"
        # overwrite on each run (use "a" if you prefer appending)
        metrics_file = metrics_path.open("w")


        # How many pruning iterations to run
        iterations = getattr(getattr(cfg.training, "pruning", cfg.training), "iterations", 0)
        if iterations < 1:
            log.warning(
                "cfg.training.pruning.iterations is < 1. "
                "Falling back to a single pruning iteration."
            )
            iterations = 1

        trainer = Trainer(cfg, experiment_dir, cfg_path)

        for iteration in range(1, iterations + 1):
            log.info(f"== PRUNING ITERATION {iteration}/{iterations} ==")

            # Build strategy from schedule / defaults
            strategy = self._build_strategy(cfg, iteration, input_channels)

            # Prepare and apply pruning masks / surgery
            strategy.prepare(self.model)
            strategy.apply(self.model)

            log.info("== AFTER pruning (with masks or surgery) ==")
            log_sparsity(self.model)
            before_metrics = evaluate_and_log(
                self.model,
                val_loader,
                self.device,
                f"PRUNED model before fine-tuning (iteration {iteration}/{iterations})."
            )

            # Fine-tune the current pruned model
            self.model = trainer.train(
                teacher=model_teacher,
                student=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                adapt_kd_proj_flag=getattr(cfg.training.pruning, "structured", False),
                iteration=iteration,
            )

            # Cleanup / finalize pruning for this iteration (e.g., remove masks)
            strategy.finalize(self.model)
            log.info("== After removing masks / cleanup ==")
            log_sparsity(self.model)
            after_metrics = evaluate_and_log(
                self.model,
                val_loader,
                self.device,
                f"Pruned + fine-tuned (iteration {iteration}/{iterations})."
            )

            json_line = {
                "iteration": iteration,
                "before_pruning": {
                    "val_acc": before_metrics["acc"],
                    "val_mse": before_metrics["mse"],
                    "val_loss": before_metrics["loss"],
                },
                "after_pruning_finetuning": {
                    "val_acc": after_metrics["acc"],
                    "val_mse": after_metrics["mse"],
                    "val_loss": after_metrics["loss"],
                },
            }
            metrics_file.write(json.dumps(json_line) + "\n")
            metrics_file.flush()

            # Final summary after the last pruning + fine-tuning iteration
            summarize(cfg, self.model, input_channels, f"Student - Pruned and Fine-tuned iteration {iteration}", experiment_dir)

        pruned_path = Path(cfg.training.base_path_ckpts) / experiment_dir / "compressed_model.pt"
        save_state_dict(self.model, pruned_path)

        return self.model