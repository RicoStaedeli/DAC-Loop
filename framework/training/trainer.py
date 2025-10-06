from pathlib import Path
from typing import Dict, Tuple
import torch
from torch import Tensor
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm
import time
import json, shutil

# Framework Imports
from framework.utils.logger import log_section


log = logging.getLogger(__name__)


def train(cfg, model, train_loader, val_loader, experiment_dir: Path, config_path) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = _make_optimizer(model, cfg.training.learning_rate, cfg.training.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg.training.scheduler, cfg.training.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics_history = []
    metrics_path = Path(f"{experiment_dir}/metrics.json")

    # load from checkpoint
    load_from_checkpoint = cfg.training.load_from_checkpoint
    previous_checkpoint_path = cfg.training.previous_checkpoint_path

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info(f"Using device: {device} --> move model to device: {device}")
    model.to(device)

    if load_from_checkpoint:
        log.info("Loading checkpoint from %s", previous_checkpoint_path)
        model, optimizer = _load_checkpoint(model, optimizer, previous_checkpoint_path,
                                                                    device=device)

    # Log Training start for overview in logfile
    log_section("TRAINING START")

    for epoch in range(1, cfg.training.epochs + 1):
        time_t0 = time.time()
        train_metrics = _train_one_epoch(model, train_loader, device, criterion, optimizer, epoch, cfg.training.epochs)
        val_metrics = _evaluate(model, val_loader, device, criterion)
        if scheduler: scheduler.step()

        log.info(
            f"Epoch {epoch:03d} | train_loss {train_metrics['loss']:.4f} train_accuracy {train_metrics['acc']:.2f}% | val_loss {val_metrics['loss']:.4f} val_accuracy {val_metrics['acc']:.2f}%")

        time_t1 = time.time() - time_t0
        log.info(f"Time taken for epoch {epoch} = {time_t1:.2f} sec")

        metrics = {
            "train_loss": float(train_metrics['loss']),
            "train_accuracy": float(train_metrics['acc']),
            "val_loss": float(val_metrics['loss']),
            "val_accuracy": float(val_metrics['acc']),
            "epoch": epoch,
            "lr": float(next(iter(optimizer.param_groups))["lr"]),
            "time_per_epoch": time_t1,
        }

        # Save metrics after every epoch to metrics.json
        metrics_history = [m for m in metrics_history if m.get("epoch") != metrics["epoch"]]
        metrics_history.append(metrics)
        if metrics_path:
            metrics_path.write_text(json.dumps(metrics_history, indent=2))
            log.debug("Wrote metrics.json with %d entries", len(metrics_history))

        # Save every epoch
        _save_checkpoint({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, f"{experiment_dir}/epoch_{epoch}.pt")

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

    # After training finished add config file to HF upload path
    if config_path.exists():
        shutil.copy2(config_path, experiment_dir / "config.toml")




def _train_one_epoch(model, loader, device, criterion, optimizer, current_epoch, epochs) -> Dict[str, float]:
    model.train()
    total_loss, total, top1_sum = 0.0, 0, 0.0
    num_steps = len(loader)
    log.info(f"Epoch {current_epoch}/{epochs} — {num_steps} steps total")

    for images, labels in tqdm(loader, desc=f"Epoch {current_epoch}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            top1, = _accuracy_topk(logits, labels, topk=(1,))

        # Each loss.item() returned by criterion(logits, y) is already averaged over the batch (because nn.CrossEntropyLoss by default uses reduction='mean').
        total_loss += loss.item() * images.size(0)

        # If your batch has shape [B, C, H, W] (batch size, channels, height, width), then x.size(0) is B, the number of samples in this batch.
        total += images.size(0)
        top1_sum += top1.item() * images.size(0) / 100.0
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


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def _make_scheduler(optimizer, scheme: str, epochs: int):
    if scheme == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return None

def _load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    print(f"Loaded checkpoint")
    return model, optimizer

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