import logging
from logging import Logger
from pathlib import Path
from typing import Optional

import torch
from torchinfo import summary

from framework.config_parser.schemasBase import AppConfig

section_logger = logging.getLogger("section")
section_logger.propagate = False  # don’t pass to root logger
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(message)s"))  # only the message
section_logger.addHandler(handler)
section_logger.setLevel(logging.INFO)


def _format_run_spec(config: dict) -> str:
    """
    Build string from config to include in run_id.
    """
    m = config.get("model", {}) or {}
    a = m.get("args", {}) or {}
    t = config.get("training", {}) or {}

    parts = []
    mv = (m.get("model_version") or "model").lower()
    parts.append(mv)
    if "context_length" in a: parts.append(f"c{a['context_length']}")
    if "emb_dim" in a:        parts.append(f"e{a['emb_dim']}")
    if "n_heads" in a:        parts.append(f"h{a['n_heads']}")
    if "n_layers" in a:       parts.append(f"l{a['n_layers']}")
    if "batch_size" in t:     parts.append(f"bs{t['batch_size']}")
    if "learning_rate" in t:  parts.append(f"lr{(t['learning_rate'])}")
    return "_".join(parts)


def setup_logging(config: dict, experiment_dir: Path, save_to_file: bool = True):
    """
    Setup logging based on validated config (cfg.model_dump()).
      - logging.default_level
      - logging.levels (per-module overrides)
      - log file saved as logs/{run_id}.log
    :return: run_id
    """
    # Global log level
    raw_level = config.get("logging", {}).get("default_level", logging.INFO)
    log_level_str = raw_level.upper() if isinstance(raw_level, str) else "INFO"
    log_level = getattr(logging, log_level_str, logging.INFO)

    # Always include console output
    handlers = [logging.StreamHandler()]

    # Conditionally include file output
    if save_to_file:
        log_file = Path(experiment_dir / f"Experiment_log.log")
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] [%(name)s] (%(module)s.%(funcName)s:%(lineno)d) :: %(message)s",
        handlers=handlers,
        force=True,
    )

    logging.getLogger().info(
        f"Logging initialized. Level={log_level_str}, SaveToFile={save_to_file}, LogFile={log_file if save_to_file else 'N/A'}"
    )

    # per-module overrides
    module_levels = config.get("logging", {}).get("levels", {})
    for module, level_str in module_levels.items():
        level = getattr(logging, level_str.upper(), None)
        if level is None:
            logging.warning(f"[Logger] Unknown level '{level_str}' for module '{module}', skipping.")
            continue
        if level != log_level:
            logging.getLogger(module).setLevel(level)
            logging.getLogger().info(f"[Logger] Set {module} to {level_str.upper()}")


def log_meta_info(config: dict, experiment_dir: Path, seed: Optional[int] = None) -> None:
    """Log run metadata (ID, device, seed, paths) in one line."""
    log = logging.getLogger(__name__)
    exp_name = (config.get("experiment", {}) or {}).get("name", "default")
    exp_run = (config.get("experiment", {}) or {}).get("run", "default")
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    seed_str = str(seed) if seed is not None else "not-set"
    log.info(
        "Experiment=%s | Run ID=%s | Device=%s | Seed=%s | Experiment Directory=%s",
        exp_name, exp_run, device, seed_str, experiment_dir
    )


def log_section(title: str, char: str = "-") -> None:
    section_logger.info("\n%s\n%s\n%s", char * 72, title, char * 72)


def log_config_summary(cfg: AppConfig, logger: logging.Logger, level: int = logging.INFO) -> None:
    """Compact summary of model/training settings"""
    m = cfg.model
    args = m.args.model_dump()
    ordered = ("vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "drop_rate", "qkv_bias")
    msg = " ".join(f"{k}={args[k]!r}" for k in ordered if k in args)
    logger.log(level, "Model Name: %s", m.name)
    logger.log(level, "Args: %s", msg)
    t = cfg.training
    logger.log(
        level,
        "Training: epochs=%s batch_size=%s lr=%s log_interval=%s",
        t.epochs, t.batch_size, t.learning_rate, t.log_interval,
    )


def print_model_summary(cfg, model: Optional[torch.nn.Module], input_channels: int,
                        description: str, experiment_dir: Path, log: Logger) -> None:
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


def print_experiment_overview(cfg, input_channels: int, extra_channels: int, log: Logger) -> None:
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
