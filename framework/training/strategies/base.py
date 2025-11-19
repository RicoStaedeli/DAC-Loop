from dataclasses import dataclass
from typing import Protocol, Iterable

import torch


@dataclass
class EpochResult:
    loss: float
    acc: float


class TrainStrategy(Protocol):
    """
    Protocol for training strategies.
    """
    def train_one_epoch(self, *, teacher, student, loader: Iterable, device: torch.device,
                        optimizer, epoch: int, total_epochs: int) -> EpochResult: ...


def top1_accuracy(logits, y):
    """
    Computes the top-1 accuracy.
    Args:
        logits: Model output logits.
        y: True labels.
    Returns:
        Top-1 accuracy as a float tensor.
    """
    return (logits.argmax(dim=1) == y).float().mean()
