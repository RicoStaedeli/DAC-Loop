import torch
from tqdm import tqdm

from .base import EpochResult, top1_accuracy


class PlainSupervisedStrategy:
    def __init__(self, ce_loss):
        self.ce = ce_loss

    def train_one_epoch(self, *, teacher, student, loader, device, optimizer, epoch, total_epochs) -> EpochResult:
        model = student
        model.train()
        total_loss = total_acc = total_n = 0.0

        for x, y in tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = self.ce(logits, y)
            loss.backward();
            optimizer.step()

            with torch.no_grad():
                b = x.size(0)
                total_n += b
                total_loss += loss.item() * b
                total_acc += top1_accuracy(logits, y).item() * b

        return EpochResult(total_loss / total_n, 100.0 * total_acc / total_n)
