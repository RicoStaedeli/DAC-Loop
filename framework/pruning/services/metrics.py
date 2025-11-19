# services/metrics.py
from __future__ import annotations

import logging
import torch
import torch.nn as nn

from framework.utils.evaluator import Evaluator
from framework.utils.logger import print_model_summary
from framework.pruning.services.pruning_utils import prunable_layers

log = logging.getLogger(__name__)


def evaluate_and_log(model: nn.Module, loader, device: torch.device, msg: str) -> dict:
    """
    Evaluate the model on the given data loader and log the results.
    Args:
        model (nn.Module): The model to evaluate.
        loader: DataLoader providing the evaluation data.
        device (torch.device): Device to perform evaluation on.
        msg (str): Message to log before the results.
    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    evaluator = Evaluator(nn.CrossEntropyLoss())
    metrics = evaluator(model, loader, device)
    log.info("%s", msg)
    log.info("Validation Loss=%.4f | Acc=%.2f%% | MSE=%.4f",
             metrics.get("loss", float("nan")),
             metrics.get("acc", float("nan")),
             metrics.get("mse", float("nan")))
    return metrics


def summarize(cfg, model: nn.Module, input_channels: int, title: str, experiment_dir, log_obj=log) -> None:
    """
    Print a summary of the model architecture.
    Args:
        cfg: Configuration object.
        model (nn.Module): The model to summarize.
        input_channels (int): Number of input channels for the model.
        title (str): Title for the summary.
        experiment_dir: Directory to save the summary.
        log_obj: Logger object to use for logging.
    """
    print_model_summary(cfg, model, input_channels, title, experiment_dir, log_obj)


def log_sparsity(model: nn.Module) -> float:
    """
    Log the sparsity of each prunable layer in the model and the overall sparsity.
    Args:
        model (nn.Module): The model to analyze.
    Returns:
        float: Overall sparsity percentage of the model.
    """
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
