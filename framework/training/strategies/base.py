from dataclasses import dataclass
from typing import Protocol, Iterable

import torch


@dataclass
class EpochResult:
    loss: float
    acc: float


class TrainStrategy(Protocol):
    def train_one_epoch(self, *, teacher, student, loader: Iterable, device: torch.device,
                        optimizer, epoch: int, total_epochs: int) -> EpochResult: ...


def top1_accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean()
