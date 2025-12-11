import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import rcParams
from matplotlib.animation import FuncAnimation, PillowWriter

from framework.utils.utils import load_config_toml

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
    "savefig.dpi": 600,  # high-DPI for PNGs
    "figure.dpi": 150,  # canvas dpi (not critical if we pass dpi in savefig)
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
                    f", KD: {cfg.training.teacher_student.kd_type}" if hasattr(cfg.training.teacher_student,
                                                                               "kd_type") else f", KD: {cfg.training.kd_type}" if hasattr(
            cfg.training, "kd_type") else "")
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(rf"\textbf{{{subtitle}}}")

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
                    f", KD: {cfg.training.teacher_student.kd_type}" if hasattr(cfg.training.teacher_student,
                                                                               "kd_type") else f", KD: {cfg.training.kd_type}" if hasattr(
            cfg.training, "kd_type") else "")

    else:
        subtitle = (
            f"Experiment: {cfg.experiment.name}, Run: {cfg.experiment.run}, "
            f"Model: {cfg.model.name}"
        )
    ax.set_title(rf"\textbf{{{subtitle}}}")

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
                    f", KD: {cfg.training.teacher_student.kd_type}" if hasattr(cfg.training.teacher_student,
                                                                               "kd_type") else f", KD: {cfg.training.kd_type}" if hasattr(
            cfg.training, "kd_type") else "")
    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(rf"\textbf{{{subtitle}}}")

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
                    f", KD: {cfg.training.teacher_student.kd_type}" if hasattr(cfg.training.teacher_student,
                                                                               "kd_type") else f", KD: {cfg.training.kd_type}" if hasattr(
            cfg.training, "kd_type") else "")

    else:
        subtitle = f"Model: {cfg.model.name}"
    ax.set_title(rf"\textbf{{{subtitle}}}")

    if any_plotted:
        _apply_axes_style(ax, cfg, "Train vs. Val Loss")

    return _save_figure(fig, path_base, export_pdf, export_png, png_dpi)


def make_overview_per_experiment(
        base_dir="experiments/results/Experiment_3",
        run_indices=range(1, 25),
        hpad=40,  # padding between Loss and ValAcc columns
        vpad=60,  # padding between rows
        margin=60,  # outer border
        caption_width=400,  # left gutter for "Run i"
        out_path="experiments/results/Experiment_3/overview_all_plots.pdf",
        add_captions=False,
        final_scale=0.6,  # 1.0 = no final resize; <1.0 downscales once at the end
        set_dpi=400,  # logical DPI / resolution hint
        compression: bool = False
):
    """
    Create an overview of Loss / Val Acc per run and save as a single image.
    The output format is inferred from `out_path` extension:
    - .png -> PNG (lossless, with DPI metadata)
    - .pdf -> PDF (single-page bitmap)

    If compression=True:
      - Only val_accuracy_per_epoch_and_iteration.png is used
      - Overview consists of a single "Val Acc" column
    """
    rows = []
    labels = []

    for i in run_indices:
        pdir = os.path.join(base_dir, f"Run_{i}", "plots")

        if compression:
            # ✅ Only val accuracy with iteration, no loss file expected
            acc_p = os.path.join(pdir, "val_accuracy_per_epoch_and_iteration.png")
            if not os.path.exists(acc_p):
                continue

            acc_im = Image.open(acc_p).convert("RGB")
            row_h = acc_im.size[1]

            rows.append((acc_im, row_h))  # (image, height)
            labels.append(f"Run {i}")

        else:
            # ✅ Original behavior: require both loss + val accuracy per epoch
            loss_p = os.path.join(pdir, "loss_per_epoch.png")
            acc_p = os.path.join(pdir, "val_accuracy_per_epoch.png")
            if not (os.path.exists(loss_p) and os.path.exists(acc_p)):
                continue

            loss_im = Image.open(loss_p).convert("RGB")
            acc_im = Image.open(acc_p).convert("RGB")

            # Equalize row height by padding the shorter one (no resampling)
            h1, h2 = loss_im.size[1], acc_im.size[1]
            row_h = max(h1, h2)

            def pad_to_h(img, target_h):
                if img.size[1] == target_h:
                    return img
                padded = Image.new("RGB", (img.size[0], target_h), "white")
                padded.paste(img, (0, 0))
                return padded

            loss_im = pad_to_h(loss_im, row_h)
            acc_im = pad_to_h(acc_im, row_h)

            rows.append((loss_im, acc_im, row_h))  # (loss, acc, height)
            labels.append(f"Run {i}")

    if not rows:
        raise FileNotFoundError("No runs with plots found.")

    # --- Compute canvas size ---
    if compression:
        # Single-column (Val Acc only)
        acc_w = rows[0][0].size[0]
        col_w_total = acc_w
    else:
        # Two columns: Loss + Val Acc
        loss_w = rows[0][0].size[0]
        acc_w = rows[0][1].size[0]
        col_w_total = loss_w + hpad + acc_w

    nrows = len(rows)
    canvas_w = margin + (caption_width if add_captions else 0) + col_w_total + margin
    canvas_h = margin + sum(r[-1] for r in rows) + vpad * (nrows - 1) + margin

    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 30)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    # --- Column headers ---
    x0 = margin + (caption_width if add_captions else 0)
    header_y = max(10, margin // 2)

    if compression:
        # Only "Val Acc" centered in the single column
        draw.text((x0 + acc_w // 2 - 60, header_y), "Val Acc", font=font_bold, fill="black")
    else:
        # "Loss" and "Val Acc" headers for two columns
        draw.text((x0 + loss_w // 2 - 40, header_y), "Loss", font=font_bold, fill="black")
        draw.text((x0 + loss_w + hpad + acc_w // 2 - 60, header_y), "Val Acc", font=font_bold, fill="black")

    # --- Paste rows ---
    y = margin
    if compression:
        for (acc_im, row_h), label in zip(rows, labels):
            if add_captions:
                cap_x = margin
                cap_y = y + row_h // 2 - 12
                draw.text((cap_x, cap_y), label, font=font_bold, fill="black")

            x = x0
            canvas.paste(acc_im, (x, y))
            y += row_h + vpad
    else:
        for (loss_im, acc_im, row_h), label in zip(rows, labels):
            if add_captions:
                cap_x = margin
                cap_y = y + row_h // 2 - 12
                draw.text((cap_x, cap_y), label, font=font_bold, fill="black")

            x = x0
            canvas.paste(loss_im, (x, y))
            x += loss_w + hpad
            canvas.paste(acc_im, (x, y))
            y += row_h + vpad

    # Optional single downscale at the end (best quality if needed)
    if final_scale != 1.0:
        new_w = int(canvas.size[0] * final_scale)
        new_h = int(canvas.size[1] * final_scale)
        canvas = canvas.resize((new_w, new_h), Image.Resampling.LANCZOS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ext = os.path.splitext(out_path)[1].lower()

    if ext == ".png":
        canvas.save(
            out_path,
            format="PNG",
            optimize=True,
            compress_level=3,
            dpi=(set_dpi, set_dpi),
        )
    elif ext == ".pdf":
        canvas.save(
            out_path,
            format="PDF",
            resolution=set_dpi,
        )
    else:
        raise ValueError(f"Unsupported output extension for overview: {ext}")

    log.info(f"Saved overview: {out_path}")


################################################################################################################
############################ Plots with different experiments ##################################################
################################################################################################################
def _normalize_models_arg(
        models: Dict[str, Union[str, Dict[str, str]]]
) -> List[Tuple[str, str]]:
    """Accepts {name: path} OR {key: {'name': name, 'path': path}}."""
    pairs: List[Tuple[str, str]] = []
    for k, v in models.items():
        if isinstance(v, str):
            pairs.append((k, v))
        elif isinstance(v, dict):
            name = v.get("name", k)
            path = v.get("path")
            if not path:
                continue
            pairs.append((name, path))
    return pairs


def _load_valacc_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "epoch" in obj and "val_accuracy" in obj:
                rows.append({"epoch": obj["epoch"], "val_accuracy": obj["val_accuracy"]})
    if not rows:
        return pd.DataFrame(columns=["epoch", "val_accuracy"])
    df = pd.DataFrame(rows)
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["val_accuracy"] = pd.to_numeric(df["val_accuracy"], errors="coerce")
    df = df.dropna(subset=["epoch", "val_accuracy"]).sort_values("epoch").reset_index(drop=True)
    return df


def _is_fractional_accuracy(series: pd.Series) -> bool:
    """Heuristic: treat as fraction if values mostly <= 1.5."""
    if series.empty:
        return True
    s = series.dropna()
    if s.empty:
        return True
    return (s.quantile(0.95) <= 1.5)  # robust to a stray outlier


def _fmt_acc(value: float, as_percent: bool) -> str:
    if as_percent:
        return f"{value:.2f}\\%"  # LaTeX-friendly percent
    return f"{value:.4f}"


def plot_val_accuracy_over_epochs(
        models: Dict[str, Union[str, Dict[str, str]]],
        out_path_base: Union[str, Path] = "val_accuracy_over_epochs",
        *,
        title: str = "Validation Accuracy over Epochs",
        use_latex: bool = True,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
        export_pdf: bool = True,
        export_png: bool = True,
        png_dpi: int = 600,
) -> dict:
    """
    Plot val_accuracy vs. epoch for multiple runs and save as PDF/PNG.
    """
    rcParams.update({"text.usetex": use_latex, "savefig.dpi": png_dpi})

    pairs = _normalize_models_arg(models)
    if not pairs:
        raise ValueError("No valid models provided. Expected {name: path} or {key: {'name','path'}}.")

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.8)
    ax.tick_params(length=3, width=0.6, which="both")
    ax.set_xlabel("Epoch")

    as_percent = None
    any_plotted = False

    for label, path in pairs:
        if not os.path.exists(path):
            log.info(f"-> Skipping missing file: {path}")
            continue
        df = _load_valacc_jsonl(path)
        if df.empty:
            log.info(f"No usable val_accuracy in: {path}")
            continue

        if as_percent is None:
            as_percent = _is_fractional_accuracy(df["val_accuracy"])

        y = df["val_accuracy"].to_numpy()
        if as_percent:
            y = y * 100.0

        best_idx = int(df["val_accuracy"].idxmax())
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val = float(df.loc[best_idx, "val_accuracy"])
        best_disp = best_val * 100.0 if as_percent else best_val

        disp_label = rf"{label} (best={_fmt_acc(best_disp, as_percent)}@{best_epoch})"
        ax.plot(df["epoch"].to_numpy(), y, linestyle="-", marker=None, label=disp_label)
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("No series plotted (no files found or they lacked val_accuracy/epoch).")

    ax.set_ylabel("Validation Accuracy (%)" if as_percent else "Validation Accuracy")
    ax.set_title(rf"\textbf{{{title}}}")
    ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.2, loc="best")

    out_base = Path(out_path_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_figure(fig, out_base, export_pdf, export_png, png_dpi)
    return saved

def plot_compression_experiment(
        base_dir: Union[str, Path],
        cfg_path: Union[str, Path],
        *,
        title: str = "Compression Experiment – Validation Accuracy per Iteration",
        use_latex: bool = True,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
        export_pdf: bool = True,
        export_png: bool = True,
        png_dpi: int = 600,
) -> dict:
    """
    Plot val_accuracy vs. epoch for each compression iteration in a single figure.

    Expects files named 'metrics_iteration_*.jsonl' in `base_dir`, e.g.:
        metrics_iteration_1.jsonl
        metrics_iteration_2.jsonl
        ...

    Each curve's label contains the best validation accuracy and the epoch where
    it occurs, e.g. "iter 1 (best=92.50\\%@37)".
    """
    # Runtime plotting options
    rcParams.update({"text.usetex": use_latex, "savefig.dpi": png_dpi})

    base_dir = Path(base_dir)
    metric_files = sorted(base_dir.glob("metrics_iteration_*.jsonl"))

    cfg = load_config_toml(cfg_path)

    if not metric_files:
        raise FileNotFoundError(f"No metrics_iteration_*.jsonl files found in {base_dir}")

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.8)
    ax.tick_params(length=3, width=0.6, which="both")
    ax.set_xlabel("Epoch")

    as_percent: Optional[bool] = None
    any_plotted = False

    for path in metric_files:
        df = _load_valacc_jsonl(path)
        if df.empty:
            log.warning(f"No usable val_accuracy data in {path}, skipping.")
            continue

        # Decide once whether to treat accuracy as fraction or percent
        if as_percent is None:
            as_percent = _is_fractional_accuracy(df["val_accuracy"])

        y = df["val_accuracy"].to_numpy()
        if as_percent:
            y = y * 100.0

        # Find best epoch by validation accuracy
        best_idx = int(df["val_accuracy"].idxmax())
        best_epoch = int(df.loc[best_idx, "epoch"])
        best_val = float(df.loc[best_idx, "val_accuracy"])
        best_disp = best_val * 100.0 if as_percent else best_val

        # Extract iteration number from file name: metrics_iteration_3.jsonl -> 3
        stem = path.stem  # "metrics_iteration_3"
        iter_str = stem.split("metrics_iteration_")[-1]
        try:
            iteration = int(iter_str)
        except ValueError:
            iteration = iter_str  # fallback if name is weird

        label = rf"iter {iteration} (best={_fmt_acc(best_disp, as_percent)}@{best_epoch})"

        ax.plot(df["epoch"].to_numpy(), y, linestyle="-", marker=None, label=label)
        any_plotted = True

    if not any_plotted:
        raise RuntimeError("No series plotted for compression experiment (all files empty?).")

    ax.set_ylabel("Validation Accuracy (%)" if as_percent else "Validation Accuracy")
    try:
        exp_name = cfg.experiment.name
        run_id = cfg.experiment.run
        model_name = cfg.model.name
        title = rf"\textbf{{Experiment: {exp_name}, Run: {run_id}, Model: {model_name}}}"
    except Exception:
        title = r"\textbf{Compression Experiment}"

    ax.set_title(rf"\textbf{{{title}}}")
    ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.2, loc="best")

    out_base = Path(base_dir/"plots"/"val_accuracy_per_epoch_and_iteration")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_figure(fig, out_base, export_pdf, export_png, png_dpi)
    return saved

def plot_dac_compression_curves(
        json_path: Union[str, Path],
        out_path_base: Union[str, Path] = "dac_compression_curves",
        *,
        total_parameters: int,
        title: str = "Effect of DAC on Validation Accuracy vs. Compression",
        use_latex: bool = True,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
        export_pdf: bool = True,
        export_png: bool = True,
        png_dpi: int = 600,
        show_step_markers: bool = False,  # <-- NEW
) -> dict:
    """
    Plot validation accuracy with/without DAC as a function of compression.

    Expects a JSON file like:
    [
      {
        "Parameter": 11191917,
        "Acc_without_DAC": 91.24,
        "Acc_with_DAC": null
      },
      {
        "Parameter": 5878933,
        "Acc_without_DAC": null,
        "Acc_with_DAC": 92.86
      },
      ...
    ]

    Compression is computed as:
        Compression (%) = 100 * (1 - Parameter / total_parameters)

    All entries with a value in Acc_without_DAC form the "w/o DAC" curve.
    All entries with a value in Acc_with_DAC form the "w/ DAC" curve.

    If show_step_markers is True, then on both curves we:
      - interpolate accuracies at 0, 10, 20, ..., 100 % compression
        (only within the min/max compression range of each curve),
      - add markers at those points, plus the final real point,
      - and print the accuracy next to each marker.
    """

    if total_parameters <= 0:
        raise ValueError(f"total_parameters must be positive, got {total_parameters}")

    # Runtime plotting options
    rcParams.update({
        "text.usetex": use_latex,
        "savefig.dpi": png_dpi,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
    })

    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list of records in {json_path}")

    df = pd.DataFrame(data)

    required_cols = {"Parameter", "Acc_without_DAC", "Acc_with_DAC"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{json_path} is missing required columns: {missing}")

    # Ensure numeric
    df["Parameter"] = pd.to_numeric(df["Parameter"], errors="coerce")
    df["Acc_without_DAC"] = pd.to_numeric(df["Acc_without_DAC"], errors="coerce")
    df["Acc_with_DAC"] = pd.to_numeric(df["Acc_with_DAC"], errors="coerce")

    # Drop rows with invalid Parameter
    df = df.dropna(subset=["Parameter"])
    if df.empty:
        raise ValueError(f"No valid 'Parameter' values found in {json_path}")

    # Compute compression from parameter counts
    df["Compression"] = 100.0 * (1.0 - df["Parameter"] / float(total_parameters))

    # Split into two curves: w/o DAC and w/ DAC
    df_no = df.dropna(subset=["Acc_without_DAC"]).copy()
    df_yes = df.dropna(subset=["Acc_with_DAC"]).copy()

    if df_no.empty:
        raise ValueError(f"No rows with 'Acc_without_DAC' found in {json_path}")
    if df_yes.empty:
        raise ValueError(f"No rows with 'Acc_with_DAC' found in {json_path}")

    # Sort each curve by compression
    df_no = df_no.sort_values("Compression").reset_index(drop=True)
    df_yes = df_yes.sort_values("Compression").reset_index(drop=True)

    # Decide once whether to treat accuracy as fraction or percent
    all_acc = pd.concat(
        [df_no["Acc_without_DAC"], df_yes["Acc_with_DAC"]],
        ignore_index=True
    ).dropna()

    if all_acc.empty:
        raise ValueError(f"No valid accuracy values found in {json_path}")

    # In your example (80–95 range) this will evaluate to "already percent"
    as_fraction = _is_fractional_accuracy(all_acc)

    y_no = df_no["Acc_without_DAC"].to_numpy()
    y_yes = df_yes["Acc_with_DAC"].to_numpy()

    # If accuracies are fractional in [0,1], convert to percent
    if as_fraction:
        y_no = y_no * 100.0
        y_yes = y_yes * 100.0

    x_no = df_no["Compression"].to_numpy()
    x_yes = df_yes["Compression"].to_numpy()

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.8)
    ax.tick_params(length=3, width=0.6, which="both")
    ax.set_xlabel("Compression (\\%)")
    ax.set_ylabel("Validation Accuracy (\\%)")

    # Plot the two curves independently
    (line_no,) = ax.plot(x_no, y_no, linestyle="-", marker=None, label="without DAC")
    (line_yes,) = ax.plot(x_yes, y_yes, linestyle="-", marker=None, label="with DAC")

    # Optional markers/labels at 10%-steps + final point
    if show_step_markers:
        # 0, 10, 20, ..., 100
        ticks_x = np.arange(0.0, 101.0, 10.0)

        # ---- WITH DAC ----
        mask_yes = (ticks_x >= x_yes.min()) & (ticks_x <= x_yes.max())
        x_label_pts_yes = ticks_x[mask_yes]
        y_label_pts_yes = np.interp(x_label_pts_yes, x_yes, y_yes)

        # ensure final real point is included
        x_label_pts_yes = np.append(x_label_pts_yes, x_yes[-1])
        y_label_pts_yes = np.append(y_label_pts_yes, y_yes[-1])

        # ---- WITHOUT DAC ----
        mask_no = (ticks_x >= x_no.min()) & (ticks_x <= x_no.max())
        x_label_pts_no = ticks_x[mask_no]
        y_label_pts_no = np.interp(x_label_pts_no, x_no, y_no)

        # ensure final real point is included
        x_label_pts_no = np.append(x_label_pts_no, x_no[-1])
        y_label_pts_no = np.append(y_label_pts_no, y_no[-1])

        # --- Static markers/labels for WITHOUT DAC (similar offsets to GIF) ---
        for j, (xi, yi) in enumerate(zip(x_label_pts_no, y_label_pts_no)):
            # marker
            ax.plot(
                xi, yi,
                marker="o",
                markersize=4,
                color=line_no.get_color(),
                label="_nolegend_",
            )

            # label offsets
            txt_x = xi + 0.8
            txt_y = yi + 0.2
            ha = "left"
            va = "bottom"

            if j == 0:
                txt_x = xi + 1.0
                txt_y = yi - 0.15

            if j == len(x_label_pts_no) - 1:
                txt_x = xi + 0.9
                txt_y = yi - 0.5

            ax.text(
                txt_x,
                txt_y,
                f"{yi:.2f}",
                fontsize=8,
                ha=ha,
                va=va,
                color=line_no.get_color(),
            )

        # --- Static markers/labels for WITH DAC (matching GIF style) ---
        index_last_point_yes = len(x_label_pts_yes) - 1
        for j, (xi, yi) in enumerate(zip(x_label_pts_yes, y_label_pts_yes)):
            ax.plot(
                xi, yi,
                marker="o",
                markersize=4,
                color=line_yes.get_color(),
                label="_nolegend_",
            )

            if j == index_last_point_yes:
                txt_x = xi + 1.0
                txt_y = yi - 0.5
                ha = "left"
                va = "bottom"
            else:
                txt_x = xi + 1.0
                txt_y = yi + 0.2
                ha = "left"
                va = "bottom"

            ax.text(
                txt_x,
                txt_y,
                f"{yi:.2f}",
                fontsize=8,
                ha=ha,
                va=va,
                color=line_yes.get_color(),
            )

    # Show every 10% step on the x-axis
    ax.set_xticks(list(range(0, 101, 10)))
    ax.set_xlim(0, 100)
    ax.set_ylim(87, 94)

    ax.set_title(rf"\textbf{{{title}}}")
    ax.legend(frameon=False, handlelength=1.2, borderaxespad=0.2, loc="best")

    out_base = Path(out_path_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = _save_figure(fig, out_base, export_pdf, export_png, png_dpi)
    return saved



def animate_only_dac_curve(
        json_path: Union[str, Path],
        out_path_base: Union[str, Path] = "dac_compression_animation",
        *,
        total_parameters: int,
        title: str = "Effect of DAC on Validation Accuracy vs. Compression",
        use_latex: bool = True,
        figsize: Tuple[float, float] = DEFAULT_FIGSIZE,
        fps: int = 5,
) -> str:
    """
    Creates ONLY a .gif where the 'with DAC' curve grows step-by-step.

    - X ticks every 10% compression
    - Y-axis max = 94
    - At each 10% step along BOTH curves (interpolated),
      draw a marker and show the accuracy label.
    - "without DAC" is static (always present in the gif).
    - "with DAC" is animated (markers/labels appear as the curve grows).
    """

    if total_parameters <= 0:
        raise ValueError(f"total_parameters must be positive, got {total_parameters}")

    rcParams.update({
        "text.usetex": use_latex,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.titlesize": 13,
    })

    json_path = Path(json_path)
    data = json.loads(json_path.read_text())
    df = pd.DataFrame(data)

    required_cols = {"Parameter", "Acc_without_DAC", "Acc_with_DAC"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{json_path} is missing required columns: {missing}")

    # numeric
    df["Parameter"] = pd.to_numeric(df["Parameter"], errors="coerce")
    df["Acc_without_DAC"] = pd.to_numeric(df["Acc_without_DAC"], errors="coerce")
    df["Acc_with_DAC"] = pd.to_numeric(df["Acc_with_DAC"], errors="coerce")
    df = df.dropna(subset=["Parameter"])

    # compression
    df["Compression"] = 100.0 * (1.0 - df["Parameter"] / float(total_parameters))

    df_no = df.dropna(subset=["Acc_without_DAC"]).copy()
    df_yes = df.dropna(subset=["Acc_with_DAC"]).copy()

    if df_no.empty or df_yes.empty:
        raise ValueError("Need both Acc_without_DAC and Acc_with_DAC data.")

    df_no = df_no.sort_values("Compression").reset_index(drop=True)
    df_yes = df_yes.sort_values("Compression").reset_index(drop=True)

    # fractional vs percent (simple heuristic)
    all_acc = pd.concat(
        [df_no["Acc_without_DAC"], df_yes["Acc_with_DAC"]],
        ignore_index=True
    ).dropna()
    as_fraction = False
    if not all_acc.empty:
        mn, mx = all_acc.min(), all_acc.max()
        as_fraction = (0.0 <= mn) and (mx <= 1.0)

    y_no = df_no["Acc_without_DAC"].to_numpy()
    y_yes = df_yes["Acc_with_DAC"].to_numpy()
    if as_fraction:
        y_no = y_no * 100.0
        y_yes = y_yes * 100.0

    x_no = df_no["Compression"].to_numpy()
    x_yes = df_yes["Compression"].to_numpy()

    # -------- precompute labels at 10% compression steps --------
    ticks_x = np.arange(0.0, 101.0, 10.0)

    # --- WITH DAC label points ---
    mask_yes = (ticks_x >= x_yes.min()) & (ticks_x <= x_yes.max())
    x_label_pts_yes = ticks_x[mask_yes]
    y_label_pts_yes = np.interp(x_label_pts_yes, x_yes, y_yes)

    # ensure final real point is included
    x_label_pts_yes = np.append(x_label_pts_yes, x_yes[-1])
    y_label_pts_yes = np.append(y_label_pts_yes, y_yes[-1])

    # --- WITHOUT DAC label points ---
    mask_no = (ticks_x >= x_no.min()) & (ticks_x <= x_no.max())
    x_label_pts_no = ticks_x[mask_no]
    y_label_pts_no = np.interp(x_label_pts_no, x_no, y_no)

    # ensure final real point is included
    x_label_pts_no = np.append(x_label_pts_no, x_no[-1])
    y_label_pts_no = np.append(y_label_pts_no, y_no[-1])

    # -------- plot & animate --------
    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.8)
    ax.set_xlabel("Compression (\\%)")
    ax.set_ylabel("Validation Accuracy (\\%)")
    ax.set_ylim(87, 94)                 # max y = 94
    ax.set_xticks(ticks_x)             # tick every 10% on x-axis
    ax.set_xlim(0, 100)

    # static baseline (without DAC)
    (line_no,) = ax.plot(x_no, y_no, linestyle="-", lw=1, label="without DAC")

    # ---- static markers/labels for WITHOUT DAC ----
    ax._no_labels = []
    ax._no_markers = []

    for j, (xi, yi) in enumerate(zip(x_label_pts_no, y_label_pts_no)):
        # marker (no legend entry)
        (m,) = ax.plot(
            xi, yi,
            marker="o",
            markersize=4,
            color=line_no.get_color(),
            label="_nolegend_",
        )
        ax._no_markers.append(m)

        # baseline labels slightly BELOW the point to reduce collisions
        txt_x = xi + 0.8
        txt_y = yi + 0.2
        ha = "left"
        va = "bottom"

        if j == 0:
            txt_x = xi + 1.0
            txt_y = yi - 0.15

        # tweak final point if desired
        if j == len(x_label_pts_no) - 1:
            txt_x = xi + 0.9
            txt_y = yi - 0.5

        txt = ax.text(
            txt_x,
            txt_y,
            f"{yi:.2f}",
            fontsize=8,
            ha=ha,
            va=va,
            color=line_no.get_color(),
        )
        ax._no_labels.append(txt)

    # animated curve (with DAC)
    (line_yes,) = ax.plot([], [], linestyle="-", lw=1, label="with DAC")

    ax.set_title(rf"\textbf{{{title}}}")
    ax.legend(
        frameon=False,
        loc="lower left",
    )

    # storage for DAC label + marker artists (animated ones)
    ax._dac_labels = []
    ax._dac_markers = []

    def init():
        line_yes.set_data([], [])
        # clear DAC labels & markers only
        for t in ax._dac_labels:
            t.remove()
        for m in ax._dac_markers:
            m.remove()
        ax._dac_labels = []
        ax._dac_markers = []
        return (line_yes,)

    index_last_point = len(x_label_pts_yes) - 1

    def update(frame: int):
        # grow the DAC line
        line_yes.set_data(x_yes[:frame + 1], y_yes[:frame + 1])

        # clear previous DAC labels & markers
        for t in ax._dac_labels:
            t.remove()
        for m in ax._dac_markers:
            m.remove()
        ax._dac_labels = []
        ax._dac_markers = []

        # current max compression reached by animation
        current_x_max = x_yes[frame]

        # show markers/labels for all 10%-steps that are <= current_x_max
        visible_mask = x_label_pts_yes <= current_x_max
        xs = x_label_pts_yes[visible_mask]
        ys = y_label_pts_yes[visible_mask]



        for j, (xi, yi) in enumerate(zip(xs, ys)):
            # marker
            (m,) = ax.plot(
                xi, yi,
                marker="o",
                markersize=4,
                color=line_yes.get_color(),
                label="_nolegend_",
            )
            ax._dac_markers.append(m)

            # --- special label offset for the FINAL visible point ---
            if j == index_last_point:
                txt_x = xi + 1.0
                txt_y = yi - 0.5
                ha = "left"
                va = "bottom"
            else:
                txt_x = xi + 1.0
                txt_y = yi + 0.2
                ha = "left"
                va = "bottom"

            txt = ax.text(
                txt_x,
                txt_y,
                f"{yi:.2f}",
                fontsize=8,
                ha=ha,
                va=va,
                color=line_yes.get_color(),
            )
            ax._dac_labels.append(txt)

        return (line_yes, *ax._dac_labels, *ax._dac_markers)

    n_frames = len(x_yes)
    if n_frames == 0:
        raise ValueError("No 'with DAC' points available to animate.")

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
        repeat=False,
        save_count=n_frames,
    )

    out_base = Path(out_path_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    gif_path = out_base.with_suffix(".gif")

    writer = PillowWriter(fps=fps)
    anim.save(gif_path, writer=writer)

    # ---- Post-process loop count ----
    from PIL import Image, ImageSequence

    im = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(im)]
    durations = [f.info.get("duration", int(1000 / fps)) for f in frames]

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=None,   # repeat only one time
        disposal=2
    )
    # -------------------------------

    plt.close(fig)
    return str(gif_path)