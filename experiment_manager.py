import logging
import os
import shutil
import time
from pathlib import Path
from typing import Optional, Tuple
import tomllib
from dotenv import load_dotenv
from torchinfo import summary
import torch

# Framework Imports
from framework.dataset.load_for_training import load_eurosat_hf_splits, build_loaders_from_hf
from framework.models import MODEL_REGISTRY
from framework.training.trainer import train
from framework.utils.logger import setup_logging, log_meta_info
from framework.utils.utils import set_seed, load_config_toml
from framework.utils.push_to_gitlab import git_commit_and_push
from framework.utils.plots_2 import plot_teacher_student_metrics

log = logging.getLogger(__name__)
load_dotenv()


import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

def _load_manifest_full(p: Path) -> tuple[list[Path], dict]:
    with p.open("rb") as f:
        cfg = tomllib.load(f)
    exp_paths = [Path(x) for x in cfg["experiments"]["paths"]]
    orch = cfg.get("orchestrator", {}) or {}
    # defaults
    orch_cfg = {
        "max_parallel": int(orch.get("max_parallel", 3)),
        "skip_existing": bool(orch.get("skip_existing", False)),
        "shuffle": bool(orch.get("shuffle", False)),
        "cap_mem_frac": float(orch.get("cap_mem_frac", 0.0)),
        "stagger_start_seconds": float(orch.get("stagger_start_seconds", 0.0)),
        "defer_git_push": bool(orch.get("defer_git_push", False)),
        "filters": list(orch.get("filters", [])),
    }
    return exp_paths, orch_cfg

def _already_done(cfg_path: Path) -> bool:
    try:
        cfg = load_config_toml(cfg_path)
        exp_dir = Path(f"experiments/{cfg.experiment.name}/Run_{cfg.experiment.run}")
        return (exp_dir / "metrics.json").exists() or (exp_dir / "DONE").exists()
    except Exception:
        return False


def run_one_experiment(CFG_PATH: Path,
                       cap_mem_frac: float = 0.0,
                       stagger_start_seconds: float = 0.0,
                       defer_git_push: bool = False) -> tuple[str, bool, str]:
    try:
        if stagger_start_seconds > 0:
            time.sleep(stagger_start_seconds * (0.5 + 0.5 * random.random()))
        cfg = load_config_toml(CFG_PATH)

        # optional CUDA cap
        try:
            if cap_mem_frac > 0:
                torch.cuda.set_per_process_memory_fraction(cap_mem_frac, device=0)
        except Exception:
            pass

        seed = cfg.training.seed
        if seed is not None:
            set_seed(seed)

        experiment_dir = _create_log_directory(cfg)
        experiment_name = cfg.experiment.name
        setup_logging(cfg.model_dump(), experiment_dir)
        log_meta_info(cfg.model_dump(), experiment_dir, seed)

        extra_channels = len(cfg.data.indices.calculated_indices or [])
        input_channels = len(cfg.data.indices.selected_bands or []) + extra_channels
        _print_experiment_overview(cfg, input_channels, extra_channels)

        model_teacher, model_student = _load_models(cfg, input_channels)
        if cfg.training.teacher_student.train_student:
            _print_model_summary(cfg, model_student, input_channels, "Student", experiment_dir)
        _print_model_summary(cfg, model_teacher, input_channels, "Teacher", experiment_dir)

        train_ds, val_ds, test_ds = load_eurosat_hf_splits(cfg.data.dataset_hf_repo_id)
        train_loader, val_loader = build_loaders_from_hf(
            train_ds, val_ds,
            cfg.training.batch_size,
            cfg.data.image_size,
            cfg.data.indices.calculated_indices,
            cfg.data.augment,
            normalize=cfg.data.normalize,
            stats_samples=0,
            selected_bands=cfg.data.indices.selected_bands,
            num_workers=cfg.data.num_workers,   # consider lowering when parallel
            pin_memory=cfg.data.pin_memory,
        )

        t0 = time.time()
        train(cfg, teacher=model_teacher, student=model_student,
              train_loader=train_loader, val_loader=val_loader,
              experiment_dir=experiment_dir, config_path=CFG_PATH)

        plot_teacher_student_metrics(f"{experiment_dir}/config.toml")
        if not defer_git_push:
            git_commit_and_push(f"Experiment={cfg.experiment.name} Run={cfg.experiment.run}")

        Path(experiment_dir, "DONE").write_text("ok")
        return experiment_name, True, f"OK in {time.time() - t0:.1f}s"
    except Exception as e:
        try:
            log.error(f"Experiment failed: {CFG_PATH} error: {e}", exc_info=True)
        except Exception:
            pass
        return str(CFG_PATH), False, repr(e)

def main():
    manifest_path = Path("configs/Experiment_3/Experiment_Orchestrator_Config.toml")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    cfg_paths, orch = _load_manifest_full(manifest_path)

    # filter by substrings on cfg path or experiment name
    if orch["filters"]:
        def keep(p: Path) -> bool:
            txt = str(p)
            try:
                c = load_config_toml(p)
                txt += f"::{c.experiment.name}"
            except Exception:
                pass
            return any(f in txt for f in orch["filters"])
        cfg_paths = [p for p in cfg_paths if keep(p)]

    if orch["skip_existing"]:
        cfg_paths = [p for p in cfg_paths if not _already_done(p)]

    if orch["shuffle"]:
        random.shuffle(cfg_paths)

    if not cfg_paths:
        print("No experiments to run after filtering.")
        return

    print(f"Scheduling {len(cfg_paths)} experiments with max_parallel={orch['max_parallel']}")

    results = []
    commit_msgs = []

    with ProcessPoolExecutor(max_workers=orch["max_parallel"], mp_context=mp.get_context("spawn")) as ex:
        futs = {
            ex.submit(
                run_one_experiment,
                p,
                orch["cap_mem_frac"],
                orch["stagger_start_seconds"],
                orch["defer_git_push"],
            ): p
            for p in cfg_paths
        }
        for i, fut in enumerate(as_completed(futs), 1):
            p = futs[fut]
            try:
                name, ok, msg = fut.result()
                status = "DONE" if ok else "FAIL"
                print(f"[{i}/{len(cfg_paths)}] [{status}] {name} -> {msg}")
                results.append((name, ok, msg))
                if ok and orch["defer_git_push"]:
                    commit_msgs.append(f"Experiment={name} completed")
            except Exception as e:
                print(f"[ERROR] {p}: {e!r}")
                results.append((str(p), False, repr(e)))

    if orch["defer_git_push"] and commit_msgs:
        for msg in commit_msgs:
            try:
                git_commit_and_push(msg)
            except Exception as e:
                print(f"[git] push failed: {e!r}")

    ok_count = sum(1 for _, ok, _ in results if ok)
    print(f"Finished. Success {ok_count}/{len(results)}.")


def _create_log_directory(cfg) -> Path:
    experiment_log_dir = Path(f"experiments/{cfg.experiment.name}/Run_{cfg.experiment.run}")
    if os.path.exists(experiment_log_dir) and cfg.saving.overwrite_run == False:
        raise ValueError("There is already a directory named '%s'. Check if you updated the run ID to a new one. "
                         "If you want to overwrite the current existing directory please set the parameter in the config file.",
                         experiment_log_dir)

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
    log.info("\n" + "=" * 80)
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
    main()
