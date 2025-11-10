import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from typing import Optional, List, Tuple
from framework.utils.utils import load_config_toml
import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---- Plot style tuned for small, print-ready figures ----
rcParams.update({
    # LaTeX typesetting (toggle at runtime via use_latex flag)
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif", "Times New Roman", "Times"],

    # Paper-friendly font sizes (points)
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "legend.title_fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.titlesize": 11,

    # Thin, crisp lines for small figures
    "axes.linewidth": 0.8,
    "lines.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,

    # Save/export defaults
    "savefig.dpi": 600,              # high-DPI for PNGs
    "figure.dpi": 150,               # canvas dpi (not critical if we pass dpi in savefig)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    # Ensure vector text in PDFs/PS (LaTeX will handle fonts)
    "pdf.fonttype": 42,  # TrueType
    "ps.fonttype": 42,

    # Layout
    "figure.constrained_layout.use": True,
})

# Recommended default: single-column figure for most journals (~3.25 in)
DEFAULT_FIGSIZE: Tuple[float, float] = (6, 3.5)  # (width, height) inches

def plot_teacher_student_metrics(
    cfg_path,
    project_root=".",
    # Export controls
    export_pdf: bool = True,
    export_png: bool = True,
    png_dpi: int = 600,
    use_latex: bool = True,
    # Figure geometry
    figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
    # Optional columns width presets (ignored if figsize provided)
    # single ~3.25in, 1.5-column ~5in, double ~6.75-7in
):
    """
    Generate compact, paper-ready plots (PDF + optional high-DPI PNG).
    Returns dict with saved paths.
    """
    # Load config
    cfg = load_config_toml(cfg_path)

    # Runtime toggles
    rcParams.update({
        "text.usetex": use_latex,
        "savefig.dpi": png_dpi,
    })

    # Resolve paths
    project_root = Path(project_root).expanduser().resolve()
    cfg_path = Path(cfg_path).expanduser().resolve()
    dir_config = cfg_path.parent

    # Load available series
    series = _prepare_series(cfg, project_root, cfg_path)

    # Prepare output dir
    save_dir = dir_config / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Make plots (compact figsize propagated everywhere)
    loss_path_base = save_dir / "loss_per_epoch"
    val_acc_path_base = save_dir / "val_accuracy_per_epoch"
    tv_acc_path_base = save_dir / "train_vs_val_accuracy_per_epoch"
    tv_loss_path_base = save_dir / "train_vs_val_loss_per_epoch"

    loss_paths = _plot_loss(series, cfg, loss_path_base, figsize=figsize,
                            export_pdf=export_pdf, export_png=export_png, png_dpi=png_dpi)
    val_acc_paths = _plot_val_accuracy(series, cfg, val_acc_path_base, figsize=figsize,
                                       export_pdf=export_pdf, export_png=export_png, png_dpi=png_dpi)
    _ = _plot_train_vs_val_accuracy(series, cfg, tv_acc_path_base, figsize=figsize,
                                    export_pdf=export_pdf, export_png=export_png, png_dpi=png_dpi)
    _ = _plot_train_vs_val_loss(series, cfg, tv_loss_path_base, figsize=figsize,
                                export_pdf=export_pdf, export_png=export_png, png_dpi=png_dpi)

    log.info(f"Plots generated save dir: {save_dir}")
    # Return PNG paths if present else PDF
    return {"loss": str(loss_paths.get("png") or loss_paths.get("pdf")),
            "val_acc": str(val_acc_paths.get("png") or val_acc_paths.get("pdf"))}


def _load_metrics(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]
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
    teacher_path = project_root / cfg.training.teacher_student.teacher_model_path
    teacher_metrics = teacher_path.parent / "metrics.jsonl"
    log.info(f"Path Teacher Metrics: {teacher_metrics}")

    student_baseline_path = project_root / cfg.training.teacher_student.baseline_student
    log.info(f"Path Student Baseline: {student_baseline_path}")

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


# ---------- Save helper: export as vector PDF + high-DPI PNG ----------
def _save_figure(fig, path_base: Path, export_pdf: bool, export_png: bool, png_dpi: int) -> dict:
    saved = {}
    # Vector (best for papers)
    if export_pdf:
        pdf_path = path_base.with_suffix(".pdf")
        fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
        saved["pdf"] = pdf_path
    # High-DPI raster for previews or venues requesting PNG
    if export_png:
        png_path = path_base.with_suffix(".png")
        fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches="tight")
        saved["png"] = png_path
    plt.close(fig)
    return saved


def _apply_axes_style(ax, cfg, title_prefix: str):
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.8)
    ax.tick_params(length=3, width=0.6, which="both")
    ax.set_xlabel("Epoch")
    # Titles: bold main title small caps via LaTeX, compact subtitle
    if hasattr(cfg, "experiment"):
        fig = ax.get_figure()
        fig.suptitle(rf"\textbf{{{title_prefix} for {cfg.experiment.name}, Run: {cfg.experiment.run}}}")
        # Keep axes title short to save vertical space
    ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.2, loc="best")


def _plot_loss(series, cfg, path_base: Path, figsize=DEFAULT_FIGSIZE,
               export_pdf=True, export_png=True, png_dpi=600):
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        d = df.copy()
        d["train_loss"] = pd.to_numeric(d.get("train_loss"), errors="coerce")
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d = d.dropna(subset=["train_loss", "epoch"])
        if d.empty:
            log.info(f"No valid train loss data for {label}")
            continue

        best_idx = d["train_loss"].idxmin()
        best_train_loss = float(d.loc[best_idx, "train_loss"])
        best_epoch = int(d.loc[best_idx, "epoch"])
        loss_label = f"{label} (best={best_train_loss:.3f}@{best_epoch})"

        ax.plot(d["epoch"].to_numpy(), d["train_loss"].to_numpy(),
                linestyle="-", marker=None, label=loss_label)
        any_plotted = True

    ax.set_ylabel("Loss")
    if hasattr(cfg, "training") and getattr(cfg.training.teacher_student, "train_student", False):
        subtitle = (f"Model: {cfg.model.name}, "
                    f"Teacher: {cfg.training.teacher_student.teacher_model_name}"
                    f", KD: {cfg.training.teacher_student.kd_type}") if hasattr(cfg.training.teacher_student, "kd_type") else ""
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    if any_plotted:
        _apply_axes_style(ax, cfg, "Loss")

    return _save_figure(fig, path_base, export_pdf, export_png, png_dpi)


def _plot_val_accuracy(series, cfg, path_base: Path, figsize=DEFAULT_FIGSIZE,
                       export_pdf=True, export_png=True, png_dpi=600):
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        if "val_accuracy" not in df.columns:
            continue
        d = df.copy()
        d["val_accuracy"] = pd.to_numeric(d["val_accuracy"], errors="coerce")
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d = d.dropna(subset=["val_accuracy", "epoch"])
        if d.empty:
            continue

        best_idx = d["val_accuracy"].idxmax()
        best_acc = float(d.loc[best_idx, "val_accuracy"])
        best_epoch = int(d.loc[best_idx, "epoch"])
        label_with_acc = rf"{label} (best={best_acc:.2f}\%@{best_epoch})"

        ax.plot(d["epoch"].to_numpy(), d["val_accuracy"].to_numpy(),
                linestyle="-", marker=None, label=label_with_acc)
        any_plotted = True

    ax.set_ylabel("Accuracy")
    if hasattr(cfg, "training") and getattr(cfg.training.teacher_student, "train_student", False):
        subtitle = (f"Model: {cfg.model.name}, "
                    f"Teacher: {cfg.training.teacher_student.teacher_model_name} "
                    f", KD: {cfg.training.teacher_student.kd_type}") if hasattr(cfg.training.teacher_student, "kd_type") else ""

    else:
        subtitle = (
            f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, "
            f"Model: {cfg.model.name}"
        )
    ax.set_title(subtitle)

    if any_plotted:
        _apply_axes_style(ax, cfg, "Accuracy")

    return _save_figure(fig, path_base, export_pdf, export_png, png_dpi)


def _plot_train_vs_val_accuracy(series, cfg, path_base: Path, figsize=DEFAULT_FIGSIZE,
                                export_pdf=True, export_png=True, png_dpi=600):
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        if not {"train_accuracy", "val_accuracy"}.issubset(df.columns):
            continue

        d = df.copy()
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d["train_accuracy"] = pd.to_numeric(d.get("train_accuracy"), errors="coerce")
        d["val_accuracy"] = pd.to_numeric(d.get("val_accuracy"), errors="coerce")
        d = d.dropna(subset=["epoch", "train_accuracy", "val_accuracy"])
        if d.empty:
            continue

        bt_idx = d["train_accuracy"].idxmax()
        bt, bt_ep = float(d.loc[bt_idx, "train_accuracy"]), int(d.loc[bt_idx, "epoch"])
        bv_idx = d["val_accuracy"].idxmax()
        bv, bv_ep = float(d.loc[bv_idx, "val_accuracy"]), int(d.loc[bv_idx, "epoch"])

        ax.plot(d["epoch"].to_numpy(), d["train_accuracy"].to_numpy(),
                linestyle="-", marker=None,
                label=rf"{label} Train (best={bt:.2f}\%@{bt_ep})")
        ax.plot(d["epoch"].to_numpy(), d["val_accuracy"].to_numpy(),
                linestyle="--", marker=None,
                label=rf"{label} Val (best={bv:.2f}\%@{bv_ep})")
        any_plotted = True

    ax.set_ylabel("Accuracy")
    if getattr(cfg.training.teacher_student, "train_student", False):
        subtitle = (f"Model: {cfg.model.name}, "
                    f"Teacher: {cfg.training.teacher_student.teacher_model_name} "
                    f", KD: {cfg.training.teacher_student.kd_type}") if hasattr(cfg.training.teacher_student, "kd_type") else ""
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    if any_plotted:
        _apply_axes_style(ax, cfg, "Train vs. Val Accuracy")

    return _save_figure(fig, path_base, export_pdf, export_png, png_dpi)


def _plot_train_vs_val_loss(series, cfg, path_base: Path, figsize=DEFAULT_FIGSIZE,
                            export_pdf=True, export_png=True, png_dpi=600):
    fig, ax = plt.subplots(figsize=figsize)
    any_plotted = False

    for label, df in series:
        if "train_loss" not in df.columns or "val_loss" not in df.columns:
            continue

        d = df.copy()
        d["epoch"] = pd.to_numeric(d.get("epoch"), errors="coerce")
        d["train_loss"] = pd.to_numeric(d.get("train_loss"), errors="coerce")
        d["val_loss"] = pd.to_numeric(d.get("val_loss"), errors="coerce")
        d = d.dropna(subset=["epoch", "train_loss", "val_loss"])
        if d.empty:
            continue

        bt_idx = d["train_loss"].idxmin()
        bt, bt_ep = float(d.loc[bt_idx, "train_loss"]), int(d.loc[bt_idx, "epoch"])
        bv_idx = d["val_loss"].idxmin()
        bv, bv_ep = float(d.loc[bv_idx, "val_loss"]), int(d.loc[bv_idx, "epoch"])

        ax.plot(d["epoch"].to_numpy(), d["train_loss"].to_numpy(),
                linestyle="-", marker=None,
                label=rf"{label} Train (best={bt:.3f}\@{bt_ep})")
        ax.plot(d["epoch"].to_numpy(), d["val_loss"].to_numpy(),
                linestyle="--", marker=None,
                label=rf"{label} Val (best={bv:.3f}\@{bv_ep})")
        any_plotted = True

    ax.set_ylabel("Loss")
    if getattr(cfg.training.teacher_student, "train_student", False):
        subtitle = (f"Model: {cfg.model.name}, "
                    f"Teacher: {cfg.training.teacher_student.teacher_model_name} "
                    f", KD: {cfg.training.teacher_student.kd_type}") if hasattr(cfg.training.teacher_student,"kd_type") else ""

    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(subtitle)

    if any_plotted:
        _apply_axes_style(ax, cfg, "Train vs. Val Loss")

    return _save_figure(fig, path_base, export_pdf, export_png, png_dpi)