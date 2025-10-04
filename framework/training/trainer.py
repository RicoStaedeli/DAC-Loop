from pathlib import Path
from typing import Dict, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging


log = logging.getLogger(__name__)

def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def _make_scheduler(optimizer, scheme: str, epochs: int):
    if scheme == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return None

def _train_one_epoch(model, loader, device, criterion, optimizer) -> Dict[str, float]:
    model.train()
    total_loss, total, top1_sum = 0.0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            top1, = _accuracy_topk(logits, y, topk=(1,))
        total_loss += loss.item() * x.size(0)
        total += x.size(0)
        top1_sum += top1.item() * x.size(0) / 100.0
    return {"loss": total_loss / total, "acc": 100.0 * top1_sum / total}

def _evaluate(model, loader, device, criterion) -> Dict[str, float]:
    model.eval()
    total_loss, total, top1_sum = 0.0, 0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            top1, = _accuracy_topk(logits, y, topk=(1,))
            total_loss += loss.item() * x.size(0)
            total += x.size(0)
            top1_sum += top1.item() * x.size(0) / 100.0
    return {"loss": total_loss / total, "acc": 100.0 * top1_sum / total}

def train(cfg, model, device, train_loader, val_loader,experiment_dir: Path) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.training.scheduler, cfg.training.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, cfg.training.epochs + 1):
        train_metrics = _train_one_epoch(model, train_loader, device, criterion, optimizer)
        val_metrics = _evaluate(model, val_loader, device, criterion)
        if scheduler: scheduler.step()

        log.info(f"Epoch {epoch:03d} | train loss {train_metrics['loss']:.4f} acc {train_metrics['acc']:.2f}% | val loss {val_metrics['loss']:.4f} acc {val_metrics['acc']:.2f}%")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            epochs_no_improve = 0
            _save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
            }, f"{experiment_dir}/best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stopping.patience:
                log.info("Early stopping")
                break

def _save_checkpoint(state, filename):
    torch.save(state, filename)

def _accuracy_topk(output: Tensor, target: Tensor, topk=(1,)) -> Tuple[Tensor, ...]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return tuple(res)