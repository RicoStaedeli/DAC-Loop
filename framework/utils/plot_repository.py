import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional, List
from framework.utils.utils import load_config_toml
import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Enable LaTeX rendering for text and set font style
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 12,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "legend.title_fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize":16,
    "figure.dpi": 400
})

def _load_metrics(path: str) -> pd.DataFrame:
    """Load a JSONL metrics file (one JSON object per line) and return a DataFrame with epoch and metrics."""
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]  # one JSON object per line

    df = pd.DataFrame(data)

    required = {"epoch", "train_loss"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    return df.sort_values("epoch").reset_index(drop=True)


def _first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _prepare_series(cfg, project_root: Path, cfg_path: Path):
    """
    Resolve expected metric file locations and load any that exist.
    Returns a list of (label, dataframe) tuples.
    """
    # Teacher metrics
    teacher_path = project_root / cfg.training.teacher_student.teacher_model_path
    teacher_metrics = teacher_path.parent / "metrics.jsonl"
    log.info(f"Path Teacher Metrics: {teacher_metrics}")

    # Baseline student metrics (expects a full path to a jsonl file)
    student_baseline_path = project_root / cfg.training.teacher_student.baseline_student
    log.info(f"Path Student Baseline: {student_baseline_path}")

    # Current run's student (or teacher) metrics, co-located with cfg file
    student_ts_path = cfg_path.parent / "metrics.jsonl"
    log.info(f"Path new Student metrics: {student_ts_path}")

    series = []

    if teacher_metrics.exists():
        try:
            log.info(f"Loading teacher metrics from {teacher_metrics}")
            series.append(("Teacher", _load_metrics(str(teacher_metrics))))
        except Exception as e:
            log.warning(f"Could not load teacher metrics from {teacher_metrics}: {e}")

    if student_baseline_path.exists():
        try:
            log.info(f"Loading student baseline metrics from {student_baseline_path}")
            series.append(("Student Baseline", _load_metrics(str(student_baseline_path))))
        except Exception as e:
            log.warning(f"Could not load baseline student metrics from {student_baseline_path}: {e}")

    if student_ts_path.exists():
        try:
            label = "Student" if cfg.training.teacher_student.train_student else "Teacher"
            log.info(f"Loading {label.lower()} metrics from {student_ts_path}")
            series.append((label, _load_metrics(str(student_ts_path))))
        except Exception as e:
            log.warning(f"Could not load student metrics from {student_ts_path}: {e}")

    if not series:
        raise FileNotFoundError("No metrics files found.")

    return series


def _plot_loss(series, cfg, save_dir: Path, figsize=(12, 5)):
    """
    Plot training loss vs. epoch for all series.
    Saves to save_dir / 'loss_per_epoch.png'
    """
    fig, ax = plt.subplots(figsize=figsize)

    for label, df in series:
        # Make numeric, clean rows
        df = df.copy()
        df["train_loss"] = pd.to_numeric(df.get("train_loss"), errors="coerce")
        df["epoch"] = pd.to_numeric(df.get("epoch"), errors="coerce")
        df = df.dropna(subset=["train_loss", "epoch"])

        if df.empty:
            log.info(f"No valid train loss data for {label}")
            loss_label = f"{label} (No Values)"
            # Still draw nothing but keep label consistent
            continue

        # Find best (min) loss
        best_idx = df["train_loss"].idxmin()
        best_train_loss = float(df.loc[best_idx, "train_loss"])
        best_epoch = int(df.loc[best_idx, "epoch"])
        log.info(f"{label}: best train loss {best_train_loss:.4f} at epoch {best_epoch}")

        loss_label = f"{label} (best={best_train_loss:.3f}@{best_epoch})"

        ax.plot(
            df["epoch"].to_numpy(),
            df["train_loss"].to_numpy(),
            linewidth=1,
            linestyle="-",
            marker=None,
            label=loss_label,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.suptitle(rf"\textbf{{Loss for {cfg.experiment.name}, Run: {cfg.experiment.run}}}")
    if cfg.training.teacher_student.train_student:
        subtitle = (
            f"Model: {cfg.model.name}, "
            f"Teacher-Model: {cfg.training.teacher_student.teacher_model_name} "
            f"KD-Type: {cfg.training.kd_type}"
        )
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)

    out_path = save_dir / "loss_per_epoch.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_val_accuracy(series, cfg, save_dir: Path, figsize=(12, 5)):
    """
    Plot validation accuracy vs. epoch for all series that contain 'val_accuracy'.
    Saves to save_dir / 'val_accuracy_per_epoch.png'
    """
    fig, ax = plt.subplots(figsize=figsize)

    any_plotted = False
    for label, df in series:
        if "val_accuracy" not in df.columns:
            log.info(f"{label}: no 'val_accuracy' column present; skipping.")
            continue

        df = df.copy()
        df["val_accuracy"] = pd.to_numeric(df["val_accuracy"], errors="coerce")
        df["epoch"] = pd.to_numeric(df.get("epoch"), errors="coerce")
        df = df.dropna(subset=["val_accuracy", "epoch"])

        if df.empty:
            log.info(f"{label}: no valid val accuracy rows after cleaning; skipping.")
            continue

        best_idx = df["val_accuracy"].idxmax()
        best_acc = float(df.loc[best_idx, "val_accuracy"])
        best_epoch = int(df.loc[best_idx, "epoch"])
        log.info(f"{label}: best val accuracy {best_acc:.2f}% at epoch {best_epoch}")

        label_with_acc = rf"{label} (best={best_acc:.2f}\%@{best_epoch})"

        ax.plot(
            df["epoch"].to_numpy(),
            df["val_accuracy"].to_numpy(),
            linewidth=1,
            linestyle="-",
            marker=None,
            label=label_with_acc,
        )
        any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    fig.suptitle(rf"\textbf{{Accuracy for {cfg.experiment.name}, Run: {cfg.experiment.run}}}")
    if cfg.training.teacher_student.train_student:
        subtitle = (
            f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, "
            f"Model: {cfg.model.name}, "
            f"Teacher-Model: {cfg.training.teacher_student.teacher_model_name} "
            f"KD-Type: {cfg.training.kd_type}"
        )
    else:
        subtitle = (
            f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, "
            f"Model: {cfg.model.name}"
        )
    ax.set_title(subtitle)

    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    if any_plotted:
        ax.legend(frameon=False)

    out_path = save_dir / "val_accuracy_per_epoch.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_teacher_student_metrics(cfg_path, project_root=".", set_png_dpi=400, use_latex=True):
    """
    Plot loss and validation accuracy vs. epoch for teacher, student with TS, and student without TS.
    Each plot is saved separately. Legends include best values per series.
    All paths are resolved relative to the given project_root.

    Returns:
        dict with keys 'loss' and 'val_acc' pointing to saved PNG paths.
    """
    # Load config
    cfg = load_config_toml(cfg_path)

    # Global figure DPI (keeps your original behavior)
    rcParams.update({"figure.dpi": set_png_dpi})
    rcParams.update({"text.usetex": use_latex})

    # Resolve paths
    project_root = Path(project_root).expanduser().resolve()
    cfg_path = Path(cfg_path).expanduser().resolve()
    dir_config = cfg_path.parent

    # Load available series
    series = _prepare_series(cfg, project_root, cfg_path)

    # Prepare output dir
    save_dir = dir_config / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Make plots
    loss_path = _plot_loss(series, cfg, save_dir)
    val_acc_path = _plot_val_accuracy(series, cfg, save_dir)

    tv_acc_path = _plot_train_vs_val_accuracy(series, cfg, save_dir)
    tv_loss_path = _plot_train_vs_val_loss(series, cfg, save_dir)

    log.info(f"Plots generated save dir: {save_dir}")
    return {"loss": str(loss_path), "val_acc": str(val_acc_path)}


def _plot_train_vs_val_accuracy(series, cfg, save_dir: Path, figsize=(12, 5)):
    """
    Plot train_accuracy and val_accuracy vs. epoch for each available series (side-by-side on the same axes).
    Saves to save_dir / 'train_vs_val_accuracy_per_epoch.png'
    """
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        if not {"train_accuracy", "val_accuracy"}.issubset(df.columns):
            log.info(f"{label}: missing 'train_accuracy' or 'val_accuracy'; skipping.")
            continue

        d = df.copy()
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d["train_accuracy"] = pd.to_numeric(d.get("train_accuracy"), errors="coerce")
        d["val_accuracy"] = pd.to_numeric(d.get("val_accuracy"), errors="coerce")
        d = d.dropna(subset=["epoch", "train_accuracy", "val_accuracy"])

        if d.empty:
            log.info(f"{label}: no valid accuracy rows after cleaning; skipping.")
            continue

        # Best points for labels (no markers drawn to keep style consistent)
        best_train_idx = d["train_accuracy"].idxmax()
        best_train_acc = float(d.loc[best_train_idx, "train_accuracy"])
        best_train_epoch = int(d.loc[best_train_idx, "epoch"])

        best_val_idx = d["val_accuracy"].idxmax()
        best_val_acc = float(d.loc[best_val_idx, "val_accuracy"])
        best_val_epoch = int(d.loc[best_val_idx, "epoch"])

        log.info(
            f"{label}: best train acc {best_train_acc:.2f}% @ epoch {best_train_epoch}; "
            f"best val acc {best_val_acc:.2f}% @ epoch {best_val_epoch}"
        )

        # Lines
        ax.plot(
            d["epoch"].to_numpy(),
            d["train_accuracy"].to_numpy(),
            linewidth=1,
            linestyle="-",
            marker=None,
            label=rf"{label} Train (best={best_train_acc:.2f}\%@{best_train_epoch})",
        )
        ax.plot(
            d["epoch"].to_numpy(),
            d["val_accuracy"].to_numpy(),
            linewidth=1,
            linestyle="--",
            marker=None,
            label=rf"{label} Val (best={best_val_acc:.2f}\%@{best_val_epoch})",
        )

        any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    fig.suptitle(rf"\textbf{{Train vs. Val Accuracy for {cfg.experiment.name}, Run: {cfg.experiment.run}}}")
    if cfg.training.teacher_student.train_student:
        subtitle = (
            f"Model: {cfg.model.name}, "
            f"Teacher-Model: {cfg.training.teacher_student.teacher_model_name} "
            f"KD-Type: {cfg.training.kd_type}"
        )
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    if any_plotted:
        ax.legend(frameon=False)

    out_path = save_dir / "train_vs_val_accuracy_per_epoch.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plot_train_vs_val_loss(series, cfg, save_dir: Path, figsize=(12, 5)):
    """
    Plot train_loss and val_loss vs. epoch for each available series (side-by-side on the same axes).
    Saves to save_dir / 'train_vs_val_loss_per_epoch.png'
    """
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        # train_loss is already required by _load_metrics; val_loss may be missing
        if "train_loss" not in df.columns or "val_loss" not in df.columns:
            log.info(f"{label}: missing 'train_loss' or 'val_loss'; skipping.")
            continue

        d = df.copy()
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d["train_loss"] = pd.to_numeric(d.get("train_loss"), errors="coerce")
        d["val_loss"] = pd.to_numeric(d.get("val_loss"), errors="coerce")
        d = d.dropna(subset=["epoch", "train_loss", "val_loss"])

        if d.empty:
            log.info(f"{label}: no valid loss rows after cleaning; skipping.")
            continue

        # Best (min) points for labels
        best_train_idx = d["train_loss"].idxmin()
        best_train_loss = float(d.loc[best_train_idx, "train_loss"])
        best_train_epoch = int(d.loc[best_train_idx, "epoch"])

        best_val_idx = d["val_loss"].idxmin()
        best_val_loss = float(d.loc[best_val_idx, "val_loss"])
        best_val_epoch = int(d.loc[best_val_idx, "epoch"])

        log.info(
            f"{label}: best train loss {best_train_loss:.4f} @ epoch {best_train_epoch}; "
            f"best val loss {best_val_loss:.4f} @ epoch {best_val_epoch}"
        )

        # Lines
        ax.plot(
            d["epoch"].to_numpy(),
            d["train_loss"].to_numpy(),
            linewidth=1,
            linestyle="-",
            marker=None,
            label=rf"{label} Train (best={best_train_loss:.3f}\@{best_train_epoch})",
        )
        ax.plot(
            d["epoch"].to_numpy(),
            d["val_loss"].to_numpy(),
            linewidth=1,
            linestyle="--",
            marker=None,
            label=rf"{label} Val (best={best_val_loss:.3f}\@{best_val_epoch})",
        )

        any_plotted = True

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    fig.suptitle(rf"\textbf{{Train vs. Val Loss for {cfg.experiment.name}, Run: {cfg.experiment.run}}}")
    if cfg.training.teacher_student.train_student:
        subtitle = (
            f"Model: {cfg.model.name}, "
            f"Teacher-Model: {cfg.training.teacher_student.teacher_model_name} "
            f"KD-Type: {cfg.training.kd_type}"
        )
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    if any_plotted:
        ax.legend(frameon=False)

    out_path = save_dir / "train_vs_val_loss_per_epoch.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path