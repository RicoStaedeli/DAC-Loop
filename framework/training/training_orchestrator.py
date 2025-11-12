import logging
import shutil
import time
from pathlib import Path

import torch.nn as nn
import torch

from framework.training.factories import OptimizerFactory, SchedulerFactory, build_strategy
from framework.training.io_ckpt import CheckpointIO
from framework.training.metrics_writer import JsonlMetricsLogger
from framework.training.strategies.plain import PlainSupervisedStrategy
from framework.utils.device_resolver import DeviceResolver
from framework.utils.evaluator import Evaluator

log = logging.getLogger(__name__)


def adapt_student_kd_proj(student: nn.Module, teacher: nn.Module, device: torch.device) -> None:
    s_dim = student.fc.in_features
    t_dim = teacher.kd_proj.out_features if hasattr(teacher, "kd_proj") and isinstance(teacher.kd_proj,
                                                                                       nn.Linear) else teacher.fc.in_features
    student.kd_proj = nn.Linear(s_dim, t_dim, bias=False).to(device)
    nn.init.kaiming_normal_(student.kd_proj.weight, nonlinearity="linear")


class Trainer:
    def __init__(self, cfg, experiment_dir: Path, config_path: Path):
        self.cfg = cfg
        self.experiment_dir = Path(experiment_dir)
        self.config_path = Path(config_path)
        self.ckpt_dir = Path(f"{cfg.training.base_path_ckpts}{self.experiment_dir}")
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = JsonlMetricsLogger(self.experiment_dir / "metrics.jsonl")
        self.evaluator = Evaluator(nn.CrossEntropyLoss())

    def _maybe_load_teacher(self, teacher, opt_teacher, device):
        p = Path(f"{self.cfg.training.base_path_ckpts}{self.cfg.training.teacher_student.teacher_model_path}")
        if p.exists():
            log.info("Loading teacher from %s", p)
            CheckpointIO.load(teacher, opt_teacher, p, device)

    def _maybe_resume(self, model, optimizer, device):
        if self.cfg.training.load_from_checkpoint:
            prev = Path(self.cfg.training.previous_checkpoint_path)
            log.info("Resuming from %s", prev)
            CheckpointIO.load(model, optimizer, prev, device)

    def _save_epoch(self, model, optimizer, epoch):
        if not self.cfg.training.save_only_best:
            CheckpointIO.save(
                {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict()},
                self.ckpt_dir / f"epoch_{epoch}.pt")

    def _save_best(self, model, optimizer, epoch, best_val_acc):
        CheckpointIO.save({"epoch": epoch, "model_state": model.state_dict(),
                           "optimizer_state": optimizer.state_dict(), "best_val_acc": best_val_acc},
                          self.ckpt_dir / "best.pt")

    def _copy_config(self):
        if self.config_path.exists():
            shutil.copy2(self.config_path, self.experiment_dir / "config.toml")
        else:
            log.warning("Config file %s not found; skipping copy", self.config_path)

    def train(self, teacher, student, train_loader, val_loader, adapt_kd_proj_flag: bool = False):
        best_model = None
        device = DeviceResolver.get()
        log.info("Using device: %s", device)

        teacher.to(device)
        model_to_train = student if student is not None else teacher
        model_to_train.to(device)

        opt_teacher = OptimizerFactory.make(teacher, self.cfg)
        opt_student = OptimizerFactory.make(student, self.cfg) if student is not None else None
        active_opt = opt_student or opt_teacher
        scheduler = SchedulerFactory.make(active_opt, self.cfg)

        if student is not None:
            self._maybe_load_teacher(teacher, opt_teacher, device)
        self._maybe_resume(model_to_train, active_opt, device)

        if adapt_kd_proj_flag and student is not None:
            adapt_student_kd_proj(student, teacher, device)

        ce_loss = nn.CrossEntropyLoss()
        strategy = build_strategy(self.cfg, ce_loss)

        best_val_acc = 0.0
        patience_ctr = 0

        for epoch in range(1, self.cfg.training.epochs + 1):
            t0 = time.time()
            if student is None:
                # teacher-only, reuse plain strategy
                train_res = PlainSupervisedStrategy(ce_loss).train_one_epoch(
                    teacher=None, student=teacher, loader=train_loader, device=device, optimizer=opt_teacher,
                    epoch=epoch, total_epochs=self.cfg.training.epochs)
                val_res = self.evaluator(teacher, val_loader, device)
                trained_model, active_opt = teacher, opt_teacher
            else:
                train_res = strategy.train_one_epoch(
                    teacher=teacher, student=student, loader=train_loader, device=device, optimizer=opt_student,
                    epoch=epoch, total_epochs=self.cfg.training.epochs)
                val_res = self.evaluator(student, val_loader, device)
                trained_model, active_opt = student, opt_student

            if scheduler: scheduler.step()

            log.info(f"Epoch {epoch:03d} | train_loss {train_res.loss:.4f} train_acc {train_res.acc:.2f}% | "
                     f"val_loss {val_res['loss']:.4f} val_acc {val_res['acc']:.2f}% val_MSE {val_res['mse']:.4f}")
            elapsed = time.time() - t0

            self.metrics_logger.write({
                "train_loss": float(train_res.loss), "train_accuracy": float(train_res.acc),
                "val_loss": float(val_res["loss"]), "val_accuracy": float(val_res["acc"]),
                "val_MSE": float(val_res["mse"]), "epoch": epoch, "time_per_epoch": elapsed
            })

            self._save_epoch(trained_model, active_opt, epoch)

            if val_res["acc"] > best_val_acc:
                best_val_acc = val_res["acc"]
                patience_ctr = 0
                self._save_best(trained_model, active_opt, epoch, best_val_acc)
                best_model = trained_model
            else:
                patience_ctr += 1
                if self.cfg.training.early_stopping.enabled and patience_ctr >= self.cfg.training.early_stopping.patience:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        self._copy_config()
        return best_model if student is not None else teacher
