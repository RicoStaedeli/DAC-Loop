# orchestrator.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch.nn as nn

from framework.pruning.ports import PruningStrategy
from framework.pruning.services.core import maybe_load, save_state_dict
from framework.pruning.services.metrics import evaluate_and_log, summarize, log_sparsity
from framework.training.training_orchestrator import Trainer
from framework.utils.device_resolver import DeviceResolver

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

    def run(
            self,
            cfg,
            strategy: PruningStrategy,
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
            strategy (PruningStrategy): The pruning strategy to be applied.
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

        # How many pruning iterations to run
        repeat = getattr(getattr(cfg.training, "pruning", cfg.training), "repeat", 0)
        if repeat < 1:
            log.warning(
                "cfg.training.pruning.repeat is < 1. "
                "Falling back to a single pruning iteration."
            )
            repeat = 1

        trainer = Trainer(cfg, experiment_dir, cfg_path)

        for iteration in range(1, repeat + 1):
            log.info(f"== PRUNING ITERATION {iteration}/{repeat} ==")

            # Prepare and apply pruning masks / surgery
            strategy.prepare(self.model)
            strategy.apply(self.model)

            log.info("== AFTER pruning (with masks or surgery) ==")
            log_sparsity(self.model)
            evaluate_and_log(
                self.model,
                val_loader,
                self.device,
                f"PRUNED model before fine-tuning (iteration {iteration}/{repeat})."
            )

            # Fine-tune the current pruned model
            self.model = trainer.train(
                teacher=model_teacher,
                student=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                adapt_kd_proj_flag=getattr(cfg.training.pruning, "structured", False),
            )

            # Cleanup / finalize pruning for this iteration (e.g., remove masks)
            strategy.finalize(self.model)
            log.info("== After removing masks / cleanup ==")
            log_sparsity(self.model)
            evaluate_and_log(
                self.model,
                val_loader,
                self.device,
                f"Pruned + fine-tuned (iteration {iteration}/{repeat})."
            )

        # Final summary after the last pruning + fine-tuning iteration
        summarize(cfg, self.model, input_channels, "Student - Pruned and Fine-tuned", experiment_dir)

        pruned_path = Path(cfg.training.base_path_ckpts) / experiment_dir / "compressed_model.pt"
        save_state_dict(self.model, pruned_path)

        return self.model