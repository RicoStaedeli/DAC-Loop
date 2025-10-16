import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
from framework.utils.utils import load_config_toml
import os

def _load_metrics(path: str) -> pd.DataFrame:
    print(f"Loading metrics from {path}")
    """Load a metrics JSON file and return a DataFrame with epoch and metrics."""
    with open(path, "r") as f:
        data = json.load(f)
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

    # Resolve paths relative to project root
    student_no_ts_path = os.path.join(project_root, cfg.training.teacher_student.baseline_student)
    teacher_path = os.path.join(project_root, cfg.training.teacher_student.teacher_model_path)
    teacher_dir = os.path.dirname(teacher_path)
    teacher_metrics = os.path.join(teacher_dir, "metrics.json")

    # Assume TS model metrics are in the same folder as the config
    ts_dir = os.path.dirname(cfg_path)
    student_ts_path = os.path.join(ts_dir, "metrics.json")

    # Load available series
    series = []
    if os.path.exists(teacher_metrics):
        series.append(("Teacher", _load_metrics(teacher_metrics), dict(marker="o")))
    if os.path.exists(student_ts_path):
        series.append(("Student (TS)", _load_metrics(student_ts_path), dict(marker="s")))
    if os.path.exists(student_no_ts_path):
        series.append(("Student (no TS)", _load_metrics(student_no_ts_path), dict(marker="^")))

    if not series:
        raise FileNotFoundError("No metrics files found.")

    # ----- Plot LOSS (training) -----
    plt.figure(figsize=(7, 4.5), dpi=200)
    for label, df, style in series:
        best_train_loss = pd.to_numeric(df["train_loss"], errors="coerce").dropna().min()
        print(f"best train loss {best_train_loss}")
        loss_label = f"{label} (best loss={best_train_loss:.3f})"
        plt.plot(
            df["epoch"], df["train_loss"],
            linewidth=1.8, markersize=3.5,
            label=loss_label, **style
        )
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.title("Training loss over epochs", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    save_dir = f"{ts_dir}/plots/"
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(f"{save_dir}loss_per_epoch.png", bbox_inches="tight")
    plt.close()

    # ----- Plot VALIDATION ACCURACY -----
    plt.figure(figsize=(7, 4.5), dpi=200)
    for label, df, style in series:
        if "val_accuracy" in df.columns:
            best_acc = df["val_accuracy"].max()
            label_with_acc = f"{label} (best={best_acc:.2f}%)"
            plt.plot(df["epoch"], df["val_accuracy"],
                     linewidth=1.8, markersize=3.5,
                     linestyle="--",
                     label=label_with_acc, **style)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Validation Accuracy (%)", fontsize=11)
    plt.title("Validation accuracy over epochs", fontsize=12)
    plt.grid(True, linestyle=":", linewidth=0.7, alpha=0.8)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()

    plt.savefig(f"{save_dir}val_accuracy_per_epoch.png", bbox_inches="tight")
    plt.close()

    print(f"Plots generated save dir: {save_dir}")