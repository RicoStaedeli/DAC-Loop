# orchestrator.py
from __future__ import annotations
from typing import Optional
from pathlib import Path
import logging
import torch.nn as nn

from framework.pruning.ports import PruningStrategy
from framework.pruning.services.core import select_device, maybe_load, save_state_dict
from framework.pruning.services.metrics import evaluate_and_log, summarize, log_sparsity
from framework.training.trainer import train

log = logging.getLogger(__name__)

class PruningOrchestrator:
    def __init__(self, model: nn.Module, ckpt_path: Optional[str] = None, strict: bool = True):
        self.device = select_device()
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
        ce_loss = nn.CrossEntropyLoss()

        summarize(cfg, self.model, input_channels, "Student - Before Pruning", experiment_dir)
        evaluate_and_log(self.model, val_loader, self.device, ce_loss, "BEFORE pruning.")
        log_sparsity(self.model)

        strategy.prepare(self.model)
        strategy.apply(self.model)

        log.info("== AFTER pruning (with masks or surgery) ==")
        log_sparsity(self.model)
        evaluate_and_log(self.model, val_loader, self.device, ce_loss, "PRUNED model before fine-tuning.")

        self.model = train(
            cfg, teacher=model_teacher, student=self.model,
            train_loader=train_loader, val_loader=val_loader,
            experiment_dir=experiment_dir, config_path=cfg_path,
            adapt_kd_proj=getattr(cfg.training.pruning, "structured", False)
        )

        strategy.finalize(self.model)
        log.info("== After removing masks / cleanup ==")
        log_sparsity(self.model)
        evaluate_and_log(self.model, val_loader, self.device, ce_loss, "Pruned + fine-tuned.")

        summarize(cfg, self.model, input_channels, "Student - Pruned and Fine-tuned", experiment_dir)

        pruned_path = Path(cfg.training.base_path_ckpts) / experiment_dir / "compressed_model.pt"
        save_state_dict(self.model, pruned_path)
        return self.model