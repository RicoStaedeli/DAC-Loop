import argparse
import logging
import os
import shutil
import time
import tomllib
from pathlib import Path
from typing import Optional, Tuple

import torch
from dotenv import load_dotenv

# Framework Imports
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf
from framework.models import MODEL_REGISTRY
from framework.pruning.pruning_orchestrator import PruningOrchestrator
from framework.pruning.strategies.structured import StructuredPruning
from framework.pruning.strategies.unstructured import UnstructuredPruning
from framework.training.training_orchestrator import Trainer
from framework.utils.logger import setup_logging, log_meta_info, print_model_summary, print_experiment_overview
from framework.utils.plots.plot_repository import plot_teacher_student_metrics
from framework.utils.push_to_remote_repository import git_commit_and_push
from framework.utils.utils import set_seed, load_config_toml

log = logging.getLogger(__name__)
load_dotenv()


# -------------------- CLI & Config -------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment with specified GPU device.")
    parser.add_argument("--gpu", type=str, default="0",
                        help="GPU device IDs (e.g. '0', '1', or '0,1').")
    parser.add_argument("--exp", type=str, required=True, help="Name of the experiment directory eg. 'Experiment_XY'")
    return parser.parse_args()


def _load_manifest(p: Path) -> list[Path]:
    """Load experiment orchestrator (.toml) and return a list of run paths."""
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


# -------------------- Models -------------------- #
def _build_model(name: Optional[str], **kwargs) -> Optional[torch.nn.Module]:
    """Build model by name from MODEL_REGISTRY with given kwargs."""
    if not name:
        return None
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}") from exc
    return model_cls(**kwargs)


def _load_models(cfg, input_channels: int) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
    """Create teacher (optional) and student (always) models based on the config."""
    log.info("Building models...")
    ts_cfg = cfg.training.teacher_student
    teacher_name = ts_cfg.teacher_model_name if ts_cfg.train_student else None

    # the student model is the model to be trained (either standalone or via KD)
    student_name = cfg.model.name

    common = {"input_channels": input_channels, "num_classes": cfg.model.args.classes}
    teacher = _build_model(teacher_name, **common)
    student = _build_model(student_name, **common)
    return teacher, student


# -------------------- Main -------------------- #
def main(gpu_id: str, experiment_folder: str) -> None:
    # Make CUDA selection visible to child processes (if any)
    # Note: multi-GPU strings like "0,1" are respected by CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # If experiment_number is used to pick orchestrator, wire it here.
    orchestrator_path = Path("configs") / experiment_folder / "Experiment_Orchestrator_Config.toml"

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

            print_experiment_overview(cfg, input_channels, extra_channels, log)

            # Models
            model_teacher, model_student = _load_models(cfg, input_channels)

            if cfg.training.teacher_student.train_student:
                print_model_summary(cfg, model_teacher, input_channels, "Teacher", experiment_dir, log)

            print_model_summary(cfg, model_student, input_channels, "Student", experiment_dir, log)


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
                trainer = Trainer(cfg, experiment_dir=experiment_dir, config_path=cfg_path)
                trainer.train(
                    teacher=model_teacher,
                    student=model_student,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    adapt_kd_proj_flag=False,
                )

            else:
                if model_student is None:
                    raise RuntimeError(f"Pruning requires student model to be defined.")

                prefix_path = cfg.training.base_path_ckpts
                ckpt_path = cfg.training.pruning.path_checkpoint_to_prune

                pruning_orchestrator = PruningOrchestrator(model=model_student,
                                                           ckpt_path=prefix_path + ckpt_path,
                                                           strict=True)

                pruning_orchestrator.run(
                    cfg=cfg,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    input_channels=input_channels,
                    experiment_dir=experiment_dir,
                    cfg_path=cfg_path,
                    model_teacher=model_teacher,
                )

            plot_teacher_student_metrics(
                cfg_path=str(experiment_dir / "config.toml"),
                project_root=".",
                export_png=True, export_pdf=False, png_dpi=600, use_latex=False,
            )

            # Wrap-up
            duration = time.perf_counter() - t0
            log.info("Experiment %s --> end training", experiment_name)
            log.info("\n" + "=" * 88)
            log.info("RUN END | Experiment=%s | Run ID=%s | Duration=%.1fs",
                     experiment_name, cfg.experiment.run, duration)
            log.info("=" * 88)

            git_commit_and_push(f"Experiment={cfg.experiment.name} Run={cfg.experiment.run}")

        except Exception as e:
            log.error("Error during Experiment --> %s", e, exc_info=True)


if __name__ == "__main__":
    args = _parse_args()
    main(args.gpu, args.exp)
