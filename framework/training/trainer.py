import json
import logging
import shutil
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Framework Imports
from framework.utils.logger import log_section

log = logging.getLogger(__name__)


def train(cfg, teacher, student, train_loader, val_loader, experiment_dir: Path, config_path) -> None:
    ce_loss = nn.CrossEntropyLoss()
    cosine_loss = nn.CosineEmbeddingLoss()
    optimizer_teacher = _make_optimizer(teacher, cfg.training.learning_rate, cfg.training.weight_decay)
    optimizer_student = None

    if student is not None:
        optimizer_student = _make_optimizer(student, cfg.training.learning_rate, cfg.training.weight_decay)
        scheduler = _make_scheduler(optimizer_student, cfg.training.scheduler, cfg.training.epochs)
    else:
        scheduler = _make_scheduler(optimizer_teacher, cfg.training.scheduler, cfg.training.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0
    metrics_path = Path(f"{experiment_dir}/metrics.jsonl")

    ###################################################################################################################
    ################################## Prepare Models                ##################################################
    ###################################################################################################################
    # load from checkpoint
    load_from_checkpoint = cfg.training.load_from_checkpoint
    previous_checkpoint_path = cfg.training.previous_checkpoint_path
    teacher_model_path = cfg.training.teacher_student.teacher_model_path

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    log.info(f"Using device: {device} --> move model to device: {device}")

    if student is not None:
        student.to(device)
        teacher.to(device)
        log.info("Loading teacher from checkpoint %s", teacher_model_path)
        teacher, optimizer = _load_checkpoint(teacher, optimizer_teacher, f"{cfg.training.base_path_ckpts}{teacher_model_path}",
                                              device=device)
    else:
        teacher.to(device)

    if load_from_checkpoint:
        log.info("Loading checkpoint from %s", previous_checkpoint_path)
        teacher, optimizer = _load_checkpoint(teacher, optimizer_teacher, previous_checkpoint_path,
                                              device=device)

    ###################################################################################################################
    ################################## Start Training                ##################################################
    ###################################################################################################################
    log_section("TRAINING START")

    for epoch in range(1, cfg.training.epochs + 1):
        time_t0 = time.time()

        if student is not None and optimizer_student is not None:
            if cfg.training.kd_type == "CrossEntropyLoss":
                train_metrics = _train_one_epoch_with_crossEntropyLoss(teacher=teacher, student=student, loader=train_loader,
                                                                       device=device, ce_loss=ce_loss, optimizer=optimizer_student,
                                                                       current_epoch=epoch, epochs=cfg.training.epochs,
                                                                       T=cfg.training.T_param, soft_target_loss_weight=cfg.training.soft_target_loss_weight, ce_loss_weight=cfg.training.ce_loss_weight)

            if cfg.training.kd_type == "CosineEmbeddingLoss":
                train_metrics = _train_one_epoch_with_cosineEmbeddingLoss(teacher=teacher, student=student,
                                                                          loader=train_loader, device=device,
                                                                          ce_loss=ce_loss, cosine_loss=cosine_loss,
                                                                          optimizer=optimizer_student,
                                                                          current_epoch=epoch,
                                                                          epochs=cfg.training.epochs,
                                                                          hidden_rep_loss_weight=cfg.training.hidden_rep_loss_weight,
                                                                          ce_loss_weight=cfg.training.ce_loss_weight)
            if cfg.training.kd_type == "KLDivLoss":
                train_metrics = train_one_epoch_kldiv(teacher=teacher, student=student, loader=train_loader, device=device,
                                                      optimizer=optimizer_student,
                                                      current_epoch=epoch,
                                                      epochs=cfg.training.epochs,
                                                      temperature=4,
                                                      alpha_kd=0.7,
                                                      beta_feat=0.25,
                                                      ce_loss_weight=0.3,
                                                      ce_criterion=ce_loss,
                                                      feat_projector=None,
                                                      use_mse_feat=True)

            val_metrics = _evaluate(student, val_loader, device, ce_loss)

        else:
            train_metrics = _train_one_epoch(teacher, train_loader, device, ce_loss, optimizer_teacher, epoch,
                                             cfg.training.epochs)

            val_metrics = _evaluate(teacher, val_loader, device, ce_loss)
        if scheduler: scheduler.step()

        log.info(
            f"Epoch {epoch:03d} | train_loss {train_metrics['loss']:.4f} train_accuracy {train_metrics['acc']:.2f}% | val_loss {val_metrics['loss']:.4f} val_accuracy {val_metrics['acc']:.2f} val_MSE {val_metrics['mse']:.2f}%")

        time_t1 = time.time() - time_t0
        log.info(f"Time taken for epoch {epoch} = {time_t1:.2f} sec")

        metrics = {
            "train_loss": float(train_metrics['loss']),
            "train_accuracy": float(train_metrics['acc']),
            "val_loss": float(val_metrics['loss']),
            "val_accuracy": float(val_metrics['acc']),
            "val_MSE": float(val_metrics['mse']),
            "epoch": epoch,
            "time_per_epoch": time_t1,
        }

        # Save metrics after every epoch to metrics.json
        if metrics_path:
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics) + "\n")

        save_dir = Path(f"{cfg.training.base_path_ckpts}{experiment_dir}")

        # Create the directory if it does not exist
        save_dir.mkdir(parents=True, exist_ok=True)
        # Save every epoch
        _save_checkpoint({
            "epoch": epoch,
            "model_state": student.state_dict() if student is not None else teacher.state_dict(),
            "optimizer_state": optimizer_student.state_dict() if student is not None else optimizer_teacher.state_dict(),
        }, f"{save_dir}/epoch_{epoch}.pt")

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            epochs_no_improve = 0
            _save_checkpoint({
                "epoch": epoch,
                "model_state": student.state_dict() if student is not None else teacher.state_dict(),
                "optimizer_state": optimizer_student.state_dict() if student is not None else optimizer_teacher.state_dict(),
                "best_val_acc": best_val_acc,
            }, f"{save_dir}/best.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.training.early_stopping.patience:
                log.info("Early stopping")
                break

    # After training finished add config file to HF upload path
    if config_path.exists():
        shutil.copy2(config_path, experiment_dir / "config.toml")



class FeatureProjector(nn.Module):
    """Optional: aligns student hidden to teacher hidden size."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
    def forward(self, x):
        return self.proj(x)

def kd_loss(student_logits, teacher_logits, T: float):
    # KLDiv expects log-prob from student and prob from teacher
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction="batchmean"
    ) * (T * T)

@torch.no_grad()
def top1_accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean()

def train_one_epoch_kldiv(
    *,
    teacher,
    student,
    loader,
    device,
    optimizer,
    current_epoch,
    epochs,
    temperature: float = 4.0,
    alpha_kd: float = 0.7,          # weight for KD on logits
    beta_feat: float = 0.25,        # weight for feature loss, set 0.0 to disable
    ce_loss_weight: float = 0.3,    # weight for CE on hard labels, usually 1 - alpha_kd (adjust if using beta_feat)
    ce_criterion = nn.CrossEntropyLoss(),
    feat_projector: nn.Module = None,   # optional FeatureProjector
    use_mse_feat: bool = True           # True -> MSE on normalized features, False -> 1 - cosine sim
):
    teacher.eval()
    student.train()

    total_loss = 0.0
    total_acc = 0.0
    total = 0

    for images, labels in tqdm(loader, desc=f"Epoch {current_epoch}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            t_logits, t_feat = teacher(images)

        s_logits, s_feat = student(images)

        # logits KD
        loss_kd = kd_loss(s_logits, t_logits, temperature)

        # hard CE
        loss_ce = ce_criterion(s_logits, labels)

        # optional feature loss with normalization and projection
        loss_feat = torch.tensor(0.0, device=device)
        if beta_feat > 0.0:
            if feat_projector is not None:
                s_feat = feat_projector(s_feat)

            s_n = F.normalize(s_feat, dim=1)
            t_n = F.normalize(t_feat, dim=1)

            if use_mse_feat:
                loss_feat = F.mse_loss(s_n, t_n)
            else:
                # 1 - mean cosine similarity
                loss_feat = 1.0 - F.cosine_similarity(s_n, t_n, dim=1).mean()

        # combine losses
        loss = alpha_kd * loss_kd + ce_loss_weight * loss_ce + beta_feat * loss_feat
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            bsz = images.size(0)
            total += bsz
            total_loss += loss.item() * bsz
            total_acc += top1_accuracy(s_logits, labels).item() * bsz

    return {
        "loss": total_loss / total,
        "acc": 100.0 * (total_acc / total)
    }

# Tutorial here: https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
def _train_one_epoch_with_crossEntropyLoss(*, teacher, student, loader, device, ce_loss, optimizer, current_epoch, epochs, T=2,
                                           soft_target_loss_weight=0.25, ce_loss_weight=0.75) -> Dict[str, float]:
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    total_loss, total, top1_sum = 0.0, 0, 0.0
    num_steps = len(loader)
    log.info(f"Epoch {current_epoch}/{epochs} — {num_steps} steps total")

    for images, labels in tqdm(loader, desc=f"Epoch {current_epoch}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            teacher_logits, _ = teacher(images)

        # Forward pass with the student model
        student_logits, _ = student(images)

        # Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (T ** 2)

        # Calculate the true label loss
        label_loss = ce_loss(student_logits, labels)

        # Weighted sum of the two losses
        loss = soft_target_loss_weight * soft_targets_loss + ce_loss_weight * label_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            top1, = _accuracy_topk(student_logits, labels, topk=(1,))

        # Each loss.item() returned by criterion(logits, y) is already averaged over the batch (because nn.CrossEntropyLoss by default uses reduction='mean').
        total_loss += loss.item() * images.size(0)

        # If your batch has shape [B, C, H, W] (batch size, channels, height, width), then x.size(0) is B, the number of samples in this batch.
        total += images.size(0)
        top1_sum += top1.item() * images.size(0) / 100.0
    return {"loss": total_loss / total, "acc": 100.0 * top1_sum / total}


# Tutorial here: https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
def _train_one_epoch_with_cosineEmbeddingLoss(*, teacher, student, loader, device, ce_loss, cosine_loss, optimizer,
                                              current_epoch, epochs, hidden_rep_loss_weight=0.25,
                                              ce_loss_weight=0.75) -> Dict[str, float]:
    teacher.eval()  # Teacher set to evaluation mode
    student.train()  # Student to train mode

    total_loss, total, top1_sum = 0.0, 0, 0.0
    num_steps = len(loader)
    log.info(f"Epoch {current_epoch}/{epochs} — {num_steps} steps total")

    for images, labels in tqdm(loader, desc=f"Epoch {current_epoch}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # Forward pass with the teacher model - do not save gradients here as we do not change the teacher's weights
        with torch.no_grad():
            _, teacher_hidden_representation = teacher(images)

        # Forward pass with the student model
        student_logits, student_hidden_representation = student(images)

        # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
        hidden_rep_loss = cosine_loss(student_hidden_representation, teacher_hidden_representation,
                                      target=torch.ones(images.size(0)).to(device))

        # Calculate the true label loss
        label_loss = ce_loss(student_logits, labels)

        # Weighted sum of the two losses
        loss = hidden_rep_loss_weight * hidden_rep_loss + ce_loss_weight * label_loss

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            top1, = _accuracy_topk(student_logits, labels, topk=(1,))

        # Each loss.item() returned by criterion(logits, y) is already averaged over the batch (because nn.CrossEntropyLoss by default uses reduction='mean').
        total_loss += loss.item() * images.size(0)

        # If your batch has shape [B, C, H, W] (batch size, channels, height, width), then x.size(0) is B, the number of samples in this batch.
        total += images.size(0)
        top1_sum += top1.item() * images.size(0) / 100.0
    return {"loss": total_loss / total, "acc": 100.0 * top1_sum / total}


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


def _evaluate(model, loader, device, ce_loss) -> Dict[str, float]:
    model.eval()
    total_loss, total_mse, total, top1_sum, correct = 0.0, 0.0, 0, 0.0, 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits, _ = model(inputs)
            loss = ce_loss(logits, labels)
            total_loss += loss.item() * inputs.size(0)

            probs = F.softmax(logits, dim=1)
            y_onehot = F.one_hot(labels, num_classes=probs.size(1)).float()
            mse = F.mse_loss(probs, y_onehot, reduction='mean')
            total_mse += mse.item() * inputs.size(0)

            _, predicted = torch.max(logits.data, 1)

            correct += (predicted == labels).sum().item()

            total += inputs.size(0)


    return {
        "loss": total_loss / total,
        "mse": total_mse / total,
        "acc": 100 * correct / total
    }


def _make_optimizer(model: nn.Module, lr: float, weight_decay: float):
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def _make_scheduler(optimizer, scheme: str, epochs: int):
    if scheme == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    return None


def _load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

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
