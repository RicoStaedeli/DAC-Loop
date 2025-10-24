import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional, List
from framework.utils.utils import load_config_toml
import os
import logging

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

def plot_teacher_student_metrics(cfg_path, project_root="."):
    """
    Plot loss and validation accuracy vs. epoch for teacher, student with TS, and student without TS.
    Each plot is saved separately. Legends include best values per series.
    All paths are resolved relative to the given project_root.
    """
    cfg = load_config_toml(cfg_path)

    # Always resolve absolute paths
    project_root = os.path.abspath(project_root)
    cfg_path = os.path.abspath(cfg_path)
    dir_config = os.path.dirname(cfg_path)

    #Path for teacher model metrics
    teacher_path = os.path.join(project_root, cfg.training.teacher_student.teacher_model_path)
    teacher_dir = os.path.dirname(teacher_path)
    teacher_metrics = os.path.join(teacher_dir, "metrics.jsonl")
    log.info(f"Path Teacher Metrics: {teacher_metrics}")

    # Path for metrics from baseline students
    student_baseline_path = os.path.join(project_root, cfg.training.teacher_student.baseline_student)
    log.info(f"Path Student Baseline: {student_baseline_path}")

    # path to student metrics
    ts_dir = os.path.dirname(cfg_path)
    student_ts_path = os.path.join(ts_dir, "metrics.jsonl")
    log.info(f"Path new Student metrics: {student_ts_path}")

    # Load available series
    series = []
    if os.path.exists(teacher_metrics):
        try:
            log.info(f"Loading teacher metrics from {teacher_metrics}")
            series.append(("Teacher", _load_metrics(teacher_metrics)))
        except:
            log.warning(f"Could not load teacher metrics from {teacher_metrics}")
    if os.path.exists(student_baseline_path):
        try:
            log.info(f"Loading student baseline metrics from {student_baseline_path}")
            series.append(("Student Baseline", _load_metrics(student_baseline_path)))
        except:
            log.warning(f"Could not load baseline student metrics from {student_baseline_path}")
    if os.path.exists(student_ts_path):
        try:
            log.info(f"Loading student metrics from {student_ts_path}")
            series.append(("Student" if cfg.training.teacher_student.train_student else "Teacher", _load_metrics(student_ts_path)))
        except:
            log.warning(f"Could not load student metrics from {student_ts_path}")

    if not series:
        raise FileNotFoundError("No metrics files found.")

    # ----- Plot LOSS (training) -----
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df in series:
        # Convert and drop invalid values
        df["train_loss"] = pd.to_numeric(df["train_loss"], errors="coerce")
        df = df.dropna(subset=["train_loss", "epoch"])

        # Find best loss and corresponding epoch
        best_idx = df["train_loss"].idxmin()
        best_train_loss = df.loc[best_idx, "train_loss"]
        best_epoch = df.loc[best_idx, "epoch"]

        log.info(f"{label}: best train loss {best_train_loss:.4f} at epoch {best_epoch}")

        # Label with both loss and epoch
        loss_label = f"{label} (best={best_train_loss:.3f}@{int(best_epoch)})"
        ax.plot(
            df["epoch"], df["train_loss"],
            linewidth=1,
            marker=None,
            linestyle="-",
            label=loss_label
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    fig.suptitle(r"\textbf{Validation loss over epochs}")  # main title
    if(cfg.training.teacher_student.train_student):
        ax.set_title(f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, Model: {cfg.model.name}, Teacher-Model: {cfg.training.teacher_student.teacher_model_name} KD-Type: {cfg.training.kd_type}")  # subtitle
    else:
        ax.set_title(f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, Model: {cfg.model.name}")  # subtitle
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)

    save_dir = f"{dir_config}/plots/"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}loss_per_epoch.png", bbox_inches="tight")
    plt.close()

    # ----- Plot VALIDATION ACCURACY -----
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, df in series:
        if "val_accuracy" in df.columns:
            # Convert and clean
            df["val_accuracy"] = pd.to_numeric(df["val_accuracy"], errors="coerce")
            df = df.dropna(subset=["val_accuracy", "epoch"])

            # Find best accuracy and corresponding epoch
            best_idx = df["val_accuracy"].idxmax()
            best_acc = df.loc[best_idx, "val_accuracy"]
            best_epoch = df.loc[best_idx, "epoch"]

            log.info(f"{label}: best val accuracy {best_acc:.2f}% at epoch {best_epoch}")

            # Label with both accuracy and epoch
            label_with_acc = rf"{label} (best={best_acc:.2f}\%@{int(best_epoch)})"
            ax.plot(df["epoch"], df["val_accuracy"],
                     linewidth=1,
                     marker=None,
                     linestyle="-",
                     label=label_with_acc)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    fig.suptitle(r"\textbf{Validation accuracy over epochs}")  # main title
    if(cfg.training.teacher_student.train_student):
        ax.set_title(f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, Model: {cfg.model.name}, Teacher-Model: {cfg.training.teacher_student.teacher_model_name} KD-Type: {cfg.training.kd_type}")
    else:
        ax.set_title(f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, Model: {cfg.model.name}")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    ax.legend(frameon=False)

    plt.savefig(f"{save_dir}val_accuracy_per_epoch.png", bbox_inches="tight")
    plt.close()

    log.info(f"Plots generated save dir: {save_dir}")