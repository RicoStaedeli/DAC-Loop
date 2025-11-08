import argparse
import logging
import os
import shutil
import time
import tomllib
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torchinfo import summary

# Framework Imports
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf
from framework.models import MODEL_REGISTRY
from framework.training.trainer import train, _evaluate
from framework.utils.logger import setup_logging, log_meta_info
from framework.utils.plot_repository import plot_teacher_student_metrics
from framework.utils.push_to_remote_repository import git_commit_and_push
from framework.utils.utils import set_seed, load_config_toml
from framework.compression.pruning_manager import PruningManager


log = logging.getLogger(__name__)
load_dotenv()


# -------------------- CLI & Config -------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment with specified GPU device.")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device IDs (e.g. '0', '1', or '0,1').")
    parser.add_argument("--exp", type=str, default="XY", help="Experiment Number")
    return parser.parse_args()


def _load_manifest(p: Path) -> list[Path]:
    """Load experiment manifest (.toml) and return a list of config paths."""
    with p.open("rb") as f:
        cfg = tomllib.load(f)
    return [Path(x) for x in cfg["experiments"]["paths"]]


# -------------------- Filesystem -------------------- #
def _create_log_directory(cfg) -> Path:
    experiment_log_dir = Path("experiments") / cfg.experiment.name / f"Run_{cfg.experiment.run}"
    exists = experiment_log_dir.exists()

    if exists and not cfg.saving.overwrite_run:
        raise ValueError(
            f"Directory '{experiment_log_dir}' already exists. "
            "Update the run ID or set saving.overwrite_run=true in the config."
        )

    if exists and cfg.saving.overwrite_run:
        log.warning(
            "Directory '%s' exists but overwrite_run=true. Removing and recreating...", experiment_log_dir
        )
        shutil.rmtree(experiment_log_dir)

    if not exists:
        log.info("Creating directory '%s'", experiment_log_dir)

    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    return experiment_log_dir


# -------------------- Logging helpers -------------------- #
def _print_experiment_overview(cfg, input_channels: int, extra_channels: int) -> None:
    log.info("Experiment name: %s", cfg.experiment.name)
    log.info("Run ID: %s", cfg.experiment.run)
    log.info("=" * 80)
    log.info("Input channels: %d", input_channels)
    log.info("Extra channels: %d", extra_channels)
    log.info("Selected Bands: %s", cfg.data.indices.selected_bands)
    log.info("Calculated Indices: %s", cfg.data.indices.calculated_indices)
    log.info("Number of Classes: %d", cfg.model.args.classes)
    log.info("Image Size: %dx%d", cfg.data.image_size, cfg.data.image_size)
    log.info("=" * 80)
    log.info("Number of Workers: %d", cfg.data.num_workers)
    log.info("Pin Memory: %s", cfg.data.pin_memory)
    log.info("Batch size: %d", cfg.training.batch_size)
    log.info("=" * 80)
    log.info("Dataset Name: %s", cfg.data.dataset_hf_repo_id)
    log.info("Data Augmentation: %s", cfg.data.augment)
    log.info("Data Normalization: %s", cfg.data.normalize)
    log.info("=" * 80)

    if cfg.training.teacher_student.train_student:
        ts = cfg.training.teacher_student
        log.info("Train with teacher-student knowledge distillation.")
        log.info("Teacher: %s | Student: %s | KD: %s", ts.teacher_model_name, cfg.model.name, ts.kd_type)
        log.info("HiddenRep=%.4f | CE=%.4f | SoftTarget=%.4f | T=%.4f",
                 ts.hidden_rep_loss_weight, ts.ce_loss_weight, ts.soft_target_loss_weight, ts.T_param)
    else:
        log.info("No teacher knowledge distillation. Training teacher itself (%s).", cfg.model.name)
    log.info("=" * 80)


def _print_model_summary(cfg, model: Optional[torch.nn.Module], input_channels: int,
                         description: str, experiment_dir: Path) -> None:
    """Print and persist a torchinfo summary for a model."""
    if model is None:
        return

    bs = cfg.training.batch_size
    img = cfg.data.image_size
    model_info = summary(
        model,
        input_size=(bs, input_channels, img, img),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=7,
        verbose=0,
    )

    header = f"{description}:\n" + "-" * 27 + "\n"
    text = f"{header}{model_info}\n\n"
    log.info(text)

    (experiment_dir / f"model_summary_{description}.log").write_text(text, encoding="utf-8")


# -------------------- Models -------------------- #
def _build_model(name: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    if not name:
        return None
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}") from exc
    return model_cls(**kwargs)


def _load_models(cfg, input_channels: int) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    """Create teacher (always) and student (optional) models based on the config."""
    log.info("Building models...")
    ts_cfg = cfg.training.teacher_student
    teacher_name = ts_cfg.teacher_model_name if ts_cfg.train_student else cfg.model.name
    # student if training student OR pruning enabled (pruning path expects student)
    student_name = cfg.model.name if (ts_cfg.train_student or cfg.training.pruning.prune) else None

    common = {"input_channels": input_channels, "num_classes": cfg.model.args.classes}
    teacher = _build_model(teacher_name, **common)
    student = _build_model(student_name, **common)
    return teacher, student


# -------------------- Training / Pruning flow -------------------- #
def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _evaluate_and_log(model: torch.nn.Module, loader, device: torch.device,
                      loss_fn: nn.Module, msg: str) -> None:
    metrics = _evaluate(model, loader, device, loss_fn)
    log.info("%s", msg)
    log.info("Validation Loss=%.4f | Acc=%.2f%% | MSE=%.4f",
             metrics["loss"], metrics["acc"], metrics["mse"])


def _prune_and_finetune(cfg, model_student: torch.nn.Module, model_teacher: Optional[torch.nn.Module],
                        train_loader, val_loader, input_channels: int, experiment_dir: Path) -> None:
    device = _select_device()
    ce_loss = nn.CrossEntropyLoss()
    model_student.to(device)

    pruner = PruningManager(
        model=model_student,
        prefix_path=cfg.training.base_path_ckpts,
        ckpt_path=cfg.training.pruning.path_checkpoint_to_prune,
    )

    _print_model_summary(cfg, model_student, input_channels, "Student - Before Pruning", experiment_dir)
    _evaluate_and_log(model_student, val_loader, device, ce_loss, "Evaluation BEFORE pruning completed.")
    log.info("== Sparsity BEFORE pruning ==")
    pruner.report_sparsity()

    if not cfg.training.pruning.structured:
        log.info("Pruning (unstructured, L1) amount=%.3f", cfg.training.pruning.pruning_amount)
        pruner.apply_global_pruning(amount=cfg.training.pruning.pruning_amount, method="l1_unstructured")
    else:
        log.info("Pruning (structured) amount=%.3f dim=%s",
                 cfg.training.pruning.pruning_amount, cfg.training.pruning.dim)
        pruner.apply_structured_pruning(amount=cfg.training.pruning.pruning_amount,
                                        dim=cfg.training.pruning.dim)
        example = torch.randn(1, input_channels, cfg.data.image_size, cfg.data.image_size, device=device)
        pruner.build_dependency(example)

        # 4) SURGERY: physically delete channels
        #    - If you already masked with amount=X, choose a modest ratio here (e.g., 0.0–0.2),
        #      OR skip 2) and control everything here with layer_prune_ratio alone.
        pruner.structured_surgery(layer_prune_ratio=0.15, min_channels=8)

        # 5) Verify the model truly shrank
        log.info(f"Params (trainable): {pruner.count_params(trainable_only=True):,}")
        log.info(f"Params (total):     {pruner.count_params():,}")
        _print_model_summary(cfg, pruner.get_model(), input_channels, "Student - After Surgery", experiment_dir)

    log.info("== Sparsity AFTER pruning (with masks) ==")
    pruner.report_sparsity()
    _evaluate_and_log(model_student, val_loader, device, ce_loss,
                      "Evaluation of PRUNED model before fine-tuning completed.")

    # Fine-tune the pruned model
    log.info("Starting fine-tuning of the pruned model...")
    train(cfg, teacher=model_teacher, student=pruner.get_model(),
          train_loader=train_loader, val_loader=val_loader,
          experiment_dir=experiment_dir, config_path=experiment_dir / "config.toml")

    # Remove masks and evaluate
    pruner.remove_pruning_reparam()
    log.info("== After removing masks (permanent zeros) ==")
    pruner.report_sparsity()
    _evaluate_and_log(model_student, val_loader, device, ce_loss,
                      "Evaluation of PRUNED + FINETUNED model completed.")
    _print_model_summary(cfg, model_student, input_channels, "Student - Pruned and Fine-tuned", experiment_dir)

    # Save the pruned model
    pruned_path = Path(cfg.training.base_path_ckpts) / experiment_dir / "pruned_finetuned_model.pth"
    pruner.save_pruned_model(str(pruned_path))


# -------------------- Main -------------------- #
def main(gpu_id: str, experiment_number: str) -> None:
    # Make CUDA selection visible to child processes (if any)
    # Note: multi-GPU strings like "0,1" are respected by CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # If experiment_number is used to pick orchestrator, wire it here.
    orchestrator_path = Path("configs") / "Experiment_7" / "Experiment_Orchestrator_Config.toml"

    for cfg_path in _load_manifest(orchestrator_path):
        t0 = time.perf_counter()

        try:
            # Validate .toml and freeze seed
            cfg = load_config_toml(cfg_path)
            seed = cfg.training.seed
            if seed is not None:
                set_seed(seed)

            # Prepare experiment directory & logging
            experiment_dir = _create_log_directory(cfg)
            experiment_name = cfg.experiment.name
            setup_logging(cfg.model_dump(), experiment_dir)
            log_meta_info(cfg.model_dump(), experiment_dir, seed)

            log.info("Loading experiment config: %s", cfg_path)

            # Channels and overview
            extra_channels = len(cfg.data.indices.calculated_indices or [])
            input_channels = len(cfg.data.indices.selected_bands or []) + extra_channels
            _print_experiment_overview(cfg, input_channels, extra_channels)

            # Models
            model_teacher, model_student = _load_models(cfg, input_channels)

            if cfg.training.teacher_student.train_student:
                _print_model_summary(cfg, model_student, input_channels, "Student", experiment_dir)
            _print_model_summary(cfg, model_teacher, input_channels, "Teacher", experiment_dir)

            # Data
            train_ds, val_ds, test_ds = load_hf_splits(cfg.data.dataset_hf_repo_id)
            train_loader, val_loader = build_loaders_from_hf(train_ds, val_ds,
                                                             cfg.training.batch_size,
                                                             cfg.data.image_size,
                                                             cfg.data.indices.calculated_indices,
                                                             cfg.data.augment,
                                                             normalize=cfg.data.normalize,
                                                             stats_samples=0,
                                                             selected_bands=cfg.data.indices.selected_bands,
                                                             num_workers=cfg.data.num_workers,
                                                             pin_memory=cfg.data.pin_memory)

            # Train (with or without pruning)
            if not cfg.training.pruning.prune:
                train(cfg, teacher=model_teacher, student=model_student,
                      train_loader=train_loader, val_loader=val_loader,
                      experiment_dir=experiment_dir, config_path=cfg_path)

                plot_teacher_student_metrics(
                    cfg_path=str(experiment_dir / "config.toml"),
                    project_root=".",
                    export_png=True, export_pdf=False, png_dpi=600, use_latex=False,
                )
            else:
                if model_student is None:
                    raise RuntimeError("Pruning requested but student model is None.")
                _prune_and_finetune(
                    cfg, model_student, model_teacher,
                    train_loader, val_loader,
                    input_channels, experiment_dir,
                )

            # Wrap-up
            dur = time.perf_counter() - t0
            log.info("Experiment %s --> end training", experiment_name)
            log.info("\n" + "=" * 88)
            log.info("RUN END | Experiment=%s | Run ID=%s | Duration=%.1fs",
                     experiment_name, cfg.experiment.run, dur)
            log.info("=" * 88)

            git_commit_and_push(f"Experiment={cfg.experiment.name} Run={cfg.experiment.run}")

        except Exception as e:
            log.error("Error during Experiment --> %s", e, exc_info=True)


if __name__ == "__main__":
    args = _parse_args()
    main(args.gpu, args.exp)
