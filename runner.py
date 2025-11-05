import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple
import tomllib
from dotenv import load_dotenv
from torchinfo import summary
import argparse

# Framework Imports
from framework.dataset.load_for_training import load_hf_splits, build_loaders_from_hf
from framework.models import MODEL_REGISTRY
from framework.training.trainer import train
from framework.utils.logger import setup_logging, log_meta_info
from framework.utils.utils import set_seed, load_config_toml
from framework.utils.push_to_remote_repository import git_commit_and_push
from framework.utils.plot_repository import plot_teacher_student_metrics

log = logging.getLogger(__name__)
load_dotenv()

def _parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with specified GPU device.")
    parser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU device ID to use (e.g. '0', '1', or '0,1' for multiple GPUs)."
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="XY",
        help="Experiment Number"
    )
    return parser.parse_args()

def _load_manifest(p: Path) -> list[Path]:
    """
    Load the manifest from the given path.
    """
    with p.open("rb") as f:
        cfg = tomllib.load(f)
    return [Path(x) for x in cfg["experiments"]["paths"]]


def main(gpu_id: str, experiment_number: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # config_orchestrator_path = f"configs/Experiment_{experiment_number}/Experiment_Orchestrator_Config.toml"
    config_orchestrator_path = f"configs/Experiment_5/Experiment_Orchestrator_Config.toml"

    for CFG_PATH in _load_manifest(Path(config_orchestrator_path)):

        t0 = time.time()
        # validate .toml
        cfg = load_config_toml(CFG_PATH)

        # Set all possible seeds for reproduction
        seed = cfg.training.seed
        if seed is not None:
            set_seed(seed)

        # get run_id & seed, check metadata
        experiment_dir = _create_log_directory(cfg)
        experiment_name = cfg.experiment.name

        # Logging setup
        setup_logging(cfg.model_dump(), experiment_dir)
        log_meta_info(cfg.model_dump(), experiment_dir, seed)

        log.info(f"Loading experiment {CFG_PATH}")

        # Start Training
        try:
            extra_channels = len(cfg.data.indices.calculated_indices or [])
            input_channels = len(cfg.data.indices.selected_bands or []) + extra_channels

            _print_experiment_overview(cfg, input_channels, extra_channels)

            model_teacher, model_student = _load_models(cfg, input_channels)

            _print_model_summary(cfg, model_student, input_channels, "Student",
                                 experiment_dir) if cfg.training.teacher_student.train_student else None
            _print_model_summary(cfg, model_teacher, input_channels, "Teacher", experiment_dir)

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

            train(cfg, teacher=model_teacher, student=model_student, train_loader=train_loader, val_loader=val_loader,
                  experiment_dir=experiment_dir, config_path=CFG_PATH)

            plot_teacher_student_metrics(f"{experiment_dir}/config.toml", project_root=".", set_png_dpi=350)

            log.info(f"Experiment name: {experiment_name} --> end training")
            log.info("\n" + "=" * 88)
            log.info("RUN END    |  Experiment=%s  |  Run ID=%s  |  Duration=%.1fs",
                     experiment_name, cfg.experiment.run, time.time() - t0)
            log.info("=" * 88)

            git_commit_and_push(f"Experiment={cfg.experiment.name} Run={cfg.experiment.run}")


        except Exception as e:
            log.error(f"Experiment name: {experiment_name} --> error: {e}", exc_info=True)



def _create_log_directory(cfg) -> Path:
    experiment_log_dir = Path(f"experiments/{cfg.experiment.name}/Run_{cfg.experiment.run}")
    if os.path.exists(experiment_log_dir) and cfg.saving.overwrite_run == False:
        raise ValueError(f"There is already a directory named {experiment_log_dir}. Check if you updated the run ID to a new one. "
                         "If you want to overwrite the current existing directory please set the parameter in the config file.",
                         )

    elif os.path.exists(experiment_log_dir) and cfg.saving.overwrite_run == True:
        log.warning(
            "There is already a directory named '%s'. But you choose to overwrite it. --> creating empty directory",
            experiment_log_dir)
        shutil.rmtree(experiment_log_dir)

    else:
        log.info(f"Creating directory '{experiment_log_dir}'")

    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    return experiment_log_dir


def _print_experiment_overview(cfg, input_channels, extra_channels):
    log.info(f"Experiment name: {cfg.experiment.name}")
    log.info(f"Run ID: {cfg.experiment.run}")
    log.info("=" * 80)
    log.info(f"Input channels: {input_channels}")
    log.info(f"Extra channels: {extra_channels}")
    log.info(f"Dataset: {cfg.data.dataset_hf_repo_id}")
    log.info("=" * 80)
    log.info(f"Number of Workers: {cfg.data.num_workers}")
    log.info(f"Pin Memory: {cfg.data.pin_memory}")
    log.info(f"Batch size: {cfg.training.batch_size}")
    log.info("=" * 80)
    log.info(f"Dataset Name: {cfg.data.dataset_hf_repo_id}")
    log.info(f"Number of Classes: {cfg.model.args.classes}")
    log.info(f"Image Size: {cfg.data.image_size}x{cfg.data.image_size}")
    log.info(f"Selected Bands: {cfg.data.indices.selected_bands}")
    log.info(f"Calculated Indices: {cfg.data.indices.calculated_indices}")
    log.info(f"Data Augmentation: {cfg.data.augment}")
    log.info(f"Data Normalization: {cfg.data.normalize}")
    log.info("=" * 80)
    if cfg.training.teacher_student.train_student:
        log.info("Train with teacher-student knowledge distillation.")
        log.info(f"Teacher Name: {cfg.training.teacher_student.teacher_model_name}")
        log.info(f"Student Name: {cfg.model.name}")
        log.info(f"KD Type: {cfg.training.kd_type}")
        log.info(f"Hidden Rep weight: {cfg.training.hidden_rep_loss_weight}")
        log.info(f"CE Loss weight: {cfg.training.ce_loss_weight}")
        log.info(f"Soft Target weight: {cfg.training.soft_target_loss_weight}")
        log.info(f"T weight: {cfg.training.T_param}")
    else:
        log.info("No teacher knowledge distillation. -> Train teacher itself")
        log.info(f"Teacher Name: {cfg.model.name}")
    log.info("=" * 80)


def _print_model_summary(cfg, model, input_channels, description, experiment_dir):
    """
    This function prints the summary of the model and also saves the prints in a separate file.
    """
    batch_size = cfg.training.batch_size
    image_size = cfg.data.image_size
    model_info_string = summary(model, input_size=(batch_size, input_channels, image_size, image_size),
                                col_names=("input_size", "output_size", "num_params", "trainable"),
                                depth=7,
                                verbose=0, )
    summary_text = f"{description}:\n---------------------------\n{model_info_string}\n\n"
    log.info(summary_text)

    # Persist summary to file
    summary_file = os.path.join(experiment_dir, f"model_summary_{description}.log")

    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(summary_text)


def _build_model(name: str, **kwargs):
    try:
        model_cls = MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return model_cls(**kwargs)


def _load_models(cfg, input_channels) -> Tuple[object, Optional[object]]:
    """
    Create teacher (always) and student (optional) models based on the config.

    Returns:
        (teacher_model, student_model_or_None)
    """
    log.info("Build models")
    ts_cfg = cfg.training.teacher_student
    teacher_name = ts_cfg.teacher_model_name if ts_cfg.train_student else cfg.model.name
    student_name = cfg.model.name if ts_cfg.train_student else None

    common_args = {
        "input_channels": input_channels,
        "num_classes": cfg.model.args.classes,
    }

    teacher = _build_model(teacher_name, **common_args)
    student = _build_model(student_name, **common_args) if student_name else None
    return teacher, student


if __name__ == "__main__":
    args = _parse_args()
    main(args.gpu, args.exp)
