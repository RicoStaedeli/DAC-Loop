from typing import Protocol, Iterable, Optional, Tuple
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class EpochResult:
    loss: float
    acc: float

class TrainStrategy(Protocol):
    def train_one_epoch(self, *, teacher, student, loader: Iterable, device: torch.device,
                        optimizer, epoch: int, total_epochs: int) -> EpochResult: ...

def kd_kl(student_logits, teacher_logits, T: float):
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean",
    ) * (T * T)

def top1_accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean()