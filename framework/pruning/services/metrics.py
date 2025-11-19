# services/metrics.py
from __future__ import annotations

import logging
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from framework.utils.evaluator import Evaluator
from framework.utils.logger import print_model_summary

log = logging.getLogger(__name__)


def evaluate_and_log(model: nn.Module, loader, device: torch.device, msg: str) -> dict:
    evaluator = Evaluator(nn.CrossEntropyLoss())
    metrics = evaluator(model, loader, device)
    log.info("%s", msg)
    log.info("Validation Loss=%.4f | Acc=%.2f%% | MSE=%.4f",
             metrics.get("loss", float("nan")),
             metrics.get("acc", float("nan")),
             metrics.get("mse", float("nan")))
    return metrics


def summarize(cfg, model: nn.Module, input_channels: int, title: str, experiment_dir, log_obj=log) -> None:
    print_model_summary(cfg, model, input_channels, title, experiment_dir, log_obj)


def prunable_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            yield name, m


def log_sparsity(model: nn.Module) -> float:
    total_zeros = 0
    total_params = 0
    log.info("== Layer-wise sparsity ==")
    for name, m in prunable_layers(model):
        w = m.weight.detach()
        zeros = torch.sum(w == 0).item()
        sparsity = 100.0 * zeros / w.numel()
        log.info("%s: %.2f%% sparse", name, sparsity)
        total_zeros += zeros
        total_params += w.numel()
    overall = 100.0 * total_zeros / max(total_params, 1)
    log.info("Overall sparsity: %.2f%%", overall)
    return overall
