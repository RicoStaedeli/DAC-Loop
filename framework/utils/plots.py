# utils/plots.py
from __future__ import annotations

import os
import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# ---------------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------------
class ExperimentNotFound(Exception):
    pass


# ---------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------
def _pick_run_dir(exp_dir: Path, run_id: Optional[int]) -> Path:
    if not exp_dir.exists():
        raise ExperimentNotFound(f"Experiment directory not found: {exp_dir}")

    if run_id is not None:
        rd = exp_dir / f"Run_{run_id}"
        if not rd.exists():
            raise ExperimentNotFound(f"Run_{run_id} not found under {exp_dir}")
        return rd

    run_dirs = []
    for p in exp_dir.glob("Run_*"):
        m = re.fullmatch(r"Run_(\d+)", p.name)
        if m:
            run_dirs.append((int(m.group(1)), p))
    if not run_dirs:
        raise ExperimentNotFound(f"No Run_* folders found under {exp_dir}")
    run_dirs.sort(key=lambda t: t[0])
    return run_dirs[-1][1]


def _load_toml(path: Path) -> Dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_metrics(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_log_file(run_dir: Path, experiment_name: str, run_id: int) -> Optional[Path]:
    candidate = run_dir / f"{experiment_name}_{run_id}.log"
    if candidate.exists():
        return candidate
    logs = list(run_dir.glob("*.log"))
    if len(logs) == 1:
        return logs[0]
    return None


def _normalize_accuracy_percent(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure accuracy columns are in percent if they look like [0,1]."""
    for c in [c for c in df.columns if "accuracy" in c.lower()]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].dropna().size > 0 and df[c].dropna().max() is not None and df[c].dropna().max() <= 1.0:
            df[c] = df[c] * 100.0
    return df


def _role_alias(df: pd.DataFrame, role: str) -> pd.DataFrame:
    """
    Map generic columns in df to role-specific names.
    role ∈ {"teacher","student"}.
    """
    df = df.rename(columns={
        "loss_batch": "val_loss",
        "token_accuracy": "val_accuracy",
        "train_loss_epoch": "train_loss",
        "val_token_acc": "val_accuracy",
        "token_acc": "val_accuracy",
    })
    df = _normalize_accuracy_percent(df)

    if role == "teacher":
        if "train_loss" in df and "teacher_train_loss" not in df: df["teacher_train_loss"] = df["train_loss"]
        if "val_loss"   in df and "teacher_val_loss"   not in df: df["teacher_val_loss"]   = df["val_loss"]
        if "train_accuracy" in df and "teacher_train_accuracy" not in df: df["teacher_train_accuracy"] = df["train_accuracy"]
        if "val_accuracy"   in df and "teacher_val_accuracy"   not in df: df["teacher_val_accuracy"]   = df["val_accuracy"]
    else:  # student
        if "train_loss" in df and "student_train_loss" not in df: df["student_train_loss"] = df["train_loss"]
        if "val_loss"   in df and "student_val_loss"   not in df: df["student_val_loss"]   = df["val_loss"]
        if "train_accuracy" in df and "student_train_accuracy" not in df: df["student_train_accuracy"] = df["train_accuracy"]
        if "val_accuracy"   in df and "student_val_accuracy"   not in df: df["student_val_accuracy"]   = df["val_accuracy"]
    return df



def load_experiment(
    base_dir: Path | str = Path("."),
    experiments_root: str = "experiments",  # default to lowercase to match typical trees
    experiment_name: str = "",
    run_id: Optional[int] = None,
) -> Dict[str, Any]:
    base_dir = Path(base_dir)
    exp_dir = base_dir / experiments_root / experiment_name
    if not exp_dir.exists():
        # case-insensitive fallback: try lower/upper
        alt = base_dir / experiments_root.lower() / experiment_name
        if alt.exists():
            exp_dir = alt
        else:
            alt2 = base_dir / experiments_root.upper() / experiment_name
            if alt2.exists():
                exp_dir = alt2
    run_dir = _pick_run_dir(exp_dir, run_id)

    m = re.fullmatch(r"Run_(\d+)", run_dir.name)
    resolved_run_id = int(m.group(1)) if m else -1

    config_path = run_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.toml at {config_path}")
    config = _load_toml(config_path)
    do_not_log_steps = bool(config.get("training", {}).get("do_not_log_steps", False))

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")
    metrics = _load_metrics(metrics_path)

    log_path: Optional[Path] = None
    log_text: Optional[str] = None
    if not do_not_log_steps:
        log_path = _find_log_file(run_dir, experiment_name, resolved_run_id)
        if log_path is not None:
            log_text = log_path.read_text(encoding="utf-8")

    return {
        "paths": {
            "run_dir": str(run_dir),
            "config": str(config_path),
            "metrics": str(metrics_path),
            "log": str(log_path) if log_path else None,
        },
        "config": config,
        "do_not_log_steps": do_not_log_steps,
        "metrics": metrics,
        "log_text": log_text,
        "resolved": {"experiment_name": experiment_name, "run_id": resolved_run_id},
    }


# ---------------------------------------------------------------------------------
# In-memory registries
# ---------------------------------------------------------------------------------
EXPERIMENTS: List[Dict[str, Any]] = []
LOG_FILES: List[str] = []
CUSTOM_LABELS: Dict[str, str] = {}
CONFIGS: Dict[str, dict] = {}
METRICS_PATHS: Dict[str, str] = {}
RUN_DIRS: Dict[str, str] = {}
PLOT_DIRS: Dict[str, str] = {}

# ---------------------------------------------------------------------------------
# Log parsing regexes & fns (classification-friendly)
# ---------------------------------------------------------------------------------
STEP_LINE_RE = re.compile(
    r"epoch=(?P<epoch>\d+)\s+"
    r"step=(?P<step>\d+)\s*/\s*(?P<step_total>\d+)\s+"
    r"loss_avg=(?P<loss_avg>[-+eE0-9\.]+)\s+"
    r"loss_step_\d+=(?P<loss_step>[-+eE0-9\.]+)"
    r"(?:\s+ppl[≈=](?P<ppl>[-+eE0-9\.]+))?",
    flags=re.IGNORECASE,
)

TIMESTAMP_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")

EPOCH_CLASSIF_RE = re.compile(
    r"Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    r"train_loss\s+(?P<train_loss>[-+eE0-9\.]+)\s+train_accuracy\s+(?P<train_acc>[-+eE0-9\.]+)%\s*\|\s*"
    r"val_loss\s+(?P<val_loss>[-+eE0-9\.]+)\s+val_accuracy\s+(?P<val_acc>[-+eE0-9\.]+)%",
    flags=re.IGNORECASE,
)

TEACHER_STUDENT_LINE_RE = re.compile(
    r"(?:student_(?:loss|acc(?:uracy)?)\s+(?P<student_val>[-+eE0-9\.]+)%?)|"
    r"(?:teacher_(?:loss|acc(?:uracy)?)\s+(?P<teacher_val>[-+eE0-9\.]+)%?)|"
    r"(?:kd_(?:loss|ce)\s+(?P<kd_loss>[-+eE0-9\.]+))|"
    r"(?:distill_(?:loss|ce)\s+(?P<distill_loss>[-+eE0-9\.]+))|"
    r"(?:hard_loss\s+(?P<hard_loss>[-+eE0-9\.]+))|"
    r"(?:soft_loss\s+(?P<soft_loss>[-+eE0-9\.]+))|"
    r"(?:alpha\s*[:=]\s*(?P<alpha>[-+eE0-9\.]+))|"
    r"(?:temperature\s*[:=]\s*(?P<temperature>[-+eE0-9\.]+))",
    flags=re.IGNORECASE,
)

GRAD_NORM_RE = re.compile(r"(?:grad(?:ient)?\s*[_\s]?norm)[:=]\s*(?P<gn>[-+eE0-9\.]+)")


def parse_timestamp(line: str) -> Optional[datetime]:
    m = TIMESTAMP_RE.search(line)
    if not m:
        return None
    return datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")


def _safe_sort(rows_or_df, primary: str, fallback: Optional[str] = None) -> pd.DataFrame:
    df = rows_or_df if isinstance(rows_or_df, pd.DataFrame) else pd.DataFrame(rows_or_df)
    if df.empty:
        return df
    if primary == "timestamp" and not np.issubdtype(df[primary].dtype, np.datetime64):
        df[primary] = pd.to_datetime(df[primary], errors="coerce")
    if primary in df.columns:
        return df.sort_values(primary).reset_index(drop=True)
    if fallback and fallback in df.columns:
        return df.sort_values(fallback).reset_index(drop=True)
    return df.reset_index(drop=True)


def parse_step_line(line: str) -> Optional[dict]:
    m = STEP_LINE_RE.search(line)
    if not m:
        return None
    gd = m.groupdict()
    loss_step = float(gd["loss_step"])
    loss_avg = float(gd["loss_avg"])
    ppl = float(gd["ppl"]) if gd.get("ppl") else np.nan
    return {
        "epoch": int(gd["epoch"]),
        "step": int(gd["step"]),
        "step_total": int(gd["step_total"]),
        "loss": loss_step if np.isfinite(loss_step) else loss_avg,
        "loss_avg": loss_avg,
        "loss_step": loss_step,
        "ppl": ppl,
    }


# ---------------------------------------------------------------------------------
# Config-aware role inference & path helpers
# ---------------------------------------------------------------------------------
def _iter_kv(d, prefix=""):
    """Yield (key_path_lower, value) pairs for nested dicts/lists/tuples."""
    if isinstance(d, dict):
        for k, v in d.items():
            kp = f"{prefix}.{k}" if prefix else str(k)
            yield from _iter_kv(v, kp)
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            kp = f"{prefix}[{i}]"
            yield from _iter_kv(v, kp)
    else:
        yield (str(prefix).lower(), d)


def infer_run_role(cfg: dict) -> Optional[str]:
    """
    Decide if this run is a 'teacher' or 'student'.
    Priority:
      1) Explicit distillation.role
      2) training.teacher_student.train_student flag
      3) Heuristics (is_teacher / teacher_model_* / student_model_* ...)
      4) Fallback: None (caller will default to 'student')
    """
    # 1) explicit role
    dist = (cfg.get("distillation") or {})
    role = str(dist.get("role") or "").lower()
    if role in ("teacher", "student"):
        return role

    # 2) template switch
    ts_cfg = ((cfg.get("training") or {}).get("teacher_student") or {})
    ts_flag = ts_cfg.get("train_student")
    if ts_flag in (True, "true", 1, "1"):
        return "student"
    if ts_flag in (False, "false", 0, "0"):
        # If the block exists and train_student is explicitly false, treat as teacher pretrain
        return "teacher"

    # 3) heuristics
    teacher_hits, student_hits = 0, 0

    def walk(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                yield from walk(v, f"{prefix}.{k}" if prefix else k)
        elif isinstance(d, (list, tuple)):
            for i, v in enumerate(d):
                yield from walk(v, f"{prefix}[{i}]")
        else:
            yield (str(prefix).lower(), d)

    for k, v in walk(cfg):
        k_low = k.lower()

        # boolean-style flags
        if any(t in k_low for t in ["is_teacher", "teacher_run", "teacher_mode", "use_as_teacher"]):
            if v in (True, "true", "True", 1, "1"): teacher_hits += 1
        if any(t in k_low for t in ["is_student", "student_run", "student_mode", "use_as_student"]):
            if v in (True, "true", "True", 1, "1"): student_hits += 1

        # model references
        if any(t in k_low for t in ["teacher_model", "teacher_ckpt", "teacher_name"]):
            teacher_hits += 1
        if any(t in k_low for t in ["student_model", "student_ckpt", "student_name"]):
            student_hits += 1

    if teacher_hits > student_hits and teacher_hits > 0:
        return "teacher"
    if student_hits > teacher_hits and student_hits > 0:
        return "student"

    # last resort: model name
    model_name = str((cfg.get("model") or {}).get("name") or "").lower()
    if "teacher" in model_name:
        return "teacher"

    return None


def _guess_teacher_metrics_path(cfg: dict, base_dir: Path) -> Optional[Path]:
    """
    Find teacher_model_path in cfg (any nesting), then replace the last segment
    with 'metrics.json' as requested. Resolve relative to base_dir.
    """
    teacher_path_str: Optional[str] = None
    for k, v in _iter_kv(cfg):
        if isinstance(v, str) and k.endswith("teacher_model_path"):
            teacher_path_str = v
            break
    if not teacher_path_str:
        return None

    p = Path(teacher_path_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    mp = (p.parent / "metrics.json") if p.suffix else (p / "metrics.json")
    return mp if mp.exists() else None


def _guess_baseline_metrics_path(cfg: dict, base_dir: Path) -> Optional[Path]:
    """
    Find 'baseline_student' in cfg (any nesting). If it points to a directory,
    append 'metrics.json'. Resolve relative to base_dir. Return path if exists.
    """
    baseline_str: Optional[str] = None
    for k, v in _iter_kv(cfg):
        if isinstance(v, str) and k.endswith("baseline_student"):
            baseline_str = v
            break
    if not baseline_str:
        return None

    p = Path(baseline_str)
    if not p.is_absolute():
        p = (base_dir / p).resolve()

    mp = p if p.suffix == ".json" else (p / "metrics.json")
    return mp if mp.exists() else None


def parse_logs(paths: List[Union[str, Path]]) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for path in paths:
        p = Path(path)
        label = CUSTOM_LABELS.get(str(p), p.stem)

        steps_rows, epoch_rows, best_rows = [], [], []
        extra_rows: List[Dict[str, Any]] = []

        if not p.exists():
            out[label] = {"steps": pd.DataFrame(), "epochs": pd.DataFrame(),
                          "extra": pd.DataFrame(), "best": pd.DataFrame(), "config": {}}
            continue

        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                ts = parse_timestamp(line)

                ps = parse_step_line(line)
                if ps:
                    gn = np.nan
                    if "norm" in line:
                        gnm = GRAD_NORM_RE.search(line)
                        if gnm:
                            gn = float(gnm.group("gn"))
                    steps_rows.append({"timestamp": ts, **ps, "grad_norm": gn})
                    continue

                m = EPOCH_CLASSIF_RE.search(line)
                if m:
                    epoch_rows.append({
                        "timestamp": ts,
                        "epoch": int(m.group("epoch")),
                        "train_loss": float(m.group("train_loss")),
                        "train_accuracy": float(m.group("train_acc")),
                        "val_loss": float(m.group("val_loss")),
                        "val_accuracy": float(m.group("val_acc")),
                    })
                    continue

                # teacher/student extras
                if any(tok in line.lower() for tok in
                       ["teacher_", "student_", "kd_", "distill_", "hard_loss", "soft_loss", "alpha", "temperature"]):
                    d: Dict[str, Any] = {"timestamp": ts}
                    for m2 in TEACHER_STUDENT_LINE_RE.finditer(line):
                        gd = {k: v for k, v in m2.groupdict().items() if v is not None}
                        for k, v in gd.items():
                            try:
                                d[k] = float(v)
                            except Exception:
                                pass
                    if len(d) > 1:
                        extra_rows.append(d)
                        continue

        steps_df = _safe_sort(steps_rows, "timestamp", fallback="step")
        epoch_df = _safe_sort(epoch_rows, "epoch")
        extra_df = _safe_sort(extra_rows, "timestamp")
        out[label] = {
            "config": {},
            "steps": steps_df,
            "epochs": epoch_df,
            "extra": extra_df,
            "best": pd.DataFrame()
        }
    return out


# ---------------------------------------------------------------------------------
# Colors & formatting
# ---------------------------------------------------------------------------------
parsed: Dict[str, Dict[str, object]] = {}

# Axis/aux colors
COL_TIME   = "#4c566a"   # time axis (left)
COL_KD     = "#7f7f7f"   # KD components
COL_BEST   = "#17becf"   # best marker
RIGHT_NEUTRAL = "#222222"

# Role-consistent colors
ROLE_STUDENT_COLOR = "#1f77b4"  # blue
ROLE_TEACHER_COLOR = "#ff7f0e"  # orange
BASELINE_STUDENT_COLOR = "#2ca02c"  # green


def _colorize_axes(ax_left, ax_right, col_left, col_right):
    ax_left.yaxis.label.set_color(col_left)
    ax_left.tick_params(axis="y", colors=col_left)
    if "left" in ax_left.spines:
        ax_left.spines["left"].set_color(col_left)

    ax_right.yaxis.label.set_color(col_right)
    ax_right.tick_params(axis="y", colors=col_right)
    if "right" in ax_right.spines:
        ax_right.spines["right"].set_color(col_right)


# ---------------------------------------------------------------------------------
# Metrics loading & formatting helpers
# ---------------------------------------------------------------------------------
def load_metrics_df(label_or_path):
    """
    Load metrics JSON and normalize columns.
    Critically: map generic columns to *teacher_* or *student_* according to the
    role inferred from CONFIGS for that label/path.
    """
    # Resolve path and label
    if label_or_path in METRICS_PATHS:
        label = label_or_path
        p = METRICS_PATHS[label]
    else:
        p = METRICS_PATHS.get(label_or_path, label_or_path)
        # best-effort reverse lookup for label
        label = None
        for k, v in METRICS_PATHS.items():
            if v == p:
                label = k
                break

    df = pd.read_json(p)

    # Normalize common names
    rename_map = {
        "loss_batch": "val_loss",
        "token_accuracy": "val_accuracy",
        "train_loss_epoch": "train_loss",
        "val_token_acc": "val_accuracy",
        "token_acc": "val_accuracy",
    }
    df = df.rename(columns=rename_map)

    # Ensure percent scale for accuracies in [0,1]
    for c in [c for c in df.columns if "accuracy" in c.lower()]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].max() is not None and pd.notna(df[c].max()) and df[c].max() <= 1.0:
            df[c] = df[c] * 100.0

    # Decide role for mapping generics
    cfg = CONFIGS.get(label or "", {}) or {}
    role = infer_run_role(cfg) or "student"  # default to 'student' if unknown

    # Promote/alias columns to role-specific names
    def ensure_role_alias(src: str, dst_student: str, dst_teacher: str):
        if src not in df.columns:
            return
        if role == "teacher" and dst_teacher not in df.columns:
            df[dst_teacher] = df[src]
        if role == "student" and dst_student not in df.columns:
            df[dst_student] = df[src]

    ensure_role_alias("train_loss", "student_train_loss", "teacher_train_loss")
    ensure_role_alias("val_loss", "student_val_loss", "teacher_val_loss")
    ensure_role_alias("train_accuracy", "student_train_accuracy", "teacher_train_accuracy")
    ensure_role_alias("val_accuracy", "student_val_accuracy", "teacher_val_accuracy")

    # If explicit student_/teacher_ columns exist already, keep them as-is.
    return df


def _fmt_hours(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "n/a"
    return f"{seconds/3600.0:.2f}h"


def _best_epoch_min(df, col: str):
    s = pd.to_numeric(df[col], errors="coerce")
    if s.dropna().empty:
        return (None, None)
    i = s.idxmin()
    return int(df.loc[i, "epoch"]), float(s.loc[i])


def _best_epoch_max_with_tie_last(ep_df, col: str, tol: float = 1e-9):
    s = pd.to_numeric(ep_df[col], errors="coerce")
    if s.dropna().empty:
        return (None, None)
    m = float(s.max())
    idx = s.index[(np.abs(s - m) <= tol)][-1]
    return int(ep_df.loc[idx, "epoch"]), float(s.loc[idx])


def _get_steps_for_label(label: str) -> pd.DataFrame:
    steps = parsed.get(label, {}).get("steps", pd.DataFrame())
    if isinstance(steps, list):
        steps = pd.DataFrame(steps)
    return steps if not steps.empty else pd.DataFrame()


def _steps_with_fractional_epoch(steps_df: pd.DataFrame) -> pd.DataFrame:
    if steps_df.empty or "epoch" not in steps_df.columns or "step_total" not in steps_df.columns:
        return steps_df
    df = steps_df.copy()
    st = df["step_total"].replace(0, 1)
    frac = (df["step"] - 1) / st
    df["epoch_frac"] = df["epoch"].astype(float) + frac
    return df


def experiment_name_from_log_or_paths(log_path: Optional[str], run_dir: str) -> str:
    if log_path:
        s = Path(log_path).stem
        s = re.sub(r"^\d{4}-\d{2}-\d{2}[_-]\d{2}-\d{2}-\d{2}_", "", s)
        return s
    return Path(run_dir).name


def hp_tail_from_config(cfg: dict) -> str:
    m = (cfg.get("model") or {})
    t = (cfg.get("training") or {})
    fields = {
        "model": m.get("name"),
        "epochs": t.get("epochs"),
        "bs": t.get("batch_size"),
        "lr": t.get("learning_rate"),
        "role": infer_run_role(cfg) or "student",
    }
    parts = [f"{k}={v}" for k, v in fields.items() if v is not None]
    return ", ".join(parts)


def exp_id_from_config(label: str, *, default: str = "") -> str:
    cfg = CONFIGS.get(label, {}) or {}
    exp = cfg.get("experiment") or {}
    name = str(exp.get("name") or "").strip()
    run  = exp.get("run")
    if name and run is not None:
        return f"{name}_{int(run)}"
    return default or label


# ---------------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------------
def _sync_left_ticks_to_right_grid(ax_left, ax_right):
    ax_left.autoscale(False)
    ax_right.autoscale(False)
    ax_right.figure.canvas.draw()
    r_ticks = ax_right.get_yticks()
    y2min, y2max = ax_right.get_ylim()
    if not np.isfinite([y2min, y2max]).all() or y2max <= y2min:
        return
    frac = (r_ticks - y2min) / (y2max - y2min)
    y1min, y1max = ax_left.get_ylim()
    l_ticks = y1min + frac * (y1max - y1min)
    ax_left.yaxis.set_major_locator(FixedLocator(l_ticks))
    ax_left.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))


def _save_fig(fig, stem: str, basename: str, folder: str, dpi: int = 180):
    Path(folder).mkdir(parents=True, exist_ok=True)
    fname = re.sub(r"[^-a-zA-Z0-9_.]+", "_", stem).strip("_") + f"_{basename}.png"
    fig.savefig(Path(folder) / fname, dpi=dpi, facecolor="white", edgecolor="white",
                bbox_inches="tight", pad_inches=0.25)
    print(f"Saved: {Path(folder) / fname}")


# ---------------------------------------------------------------------------------
# Plot functions (Teacher–Student aware)
# ---------------------------------------------------------------------------------
def plot_losses_teacher_student(label, dfm, cfg, run_id_str, folder):
    assert "epoch" in dfm.columns
    epochs = dfm["epoch"].astype(int).to_numpy()
    e_min, e_max = int(epochs.min()), int(epochs.max())

    fig = plt.figure(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
    ax_time = fig.add_subplot(111)
    twin = ax_time.twinx()

    # Left y: time/epoch
    if "time_per_epoch" in dfm.columns and dfm["time_per_epoch"].notna().any():
        t = dfm["time_per_epoch"] / 60.0
        tmin, tmax = float(t.min()), float(t.max())
        pad_y = max(1e-6, 0.02 * (tmax - tmin + 1e-9))
        ax_time.set_ylim(tmin - pad_y, tmax + pad_y)
        ax_time.plot(dfm["epoch"], t, marker="o", label="Time/epoch (min)", color=COL_TIME)
        ax_time.set_ylabel("Time per epoch (min)")
        ax_time.tick_params(axis="y", colors=COL_TIME)

    # Right y: losses
    plotted_student = False
    plotted_teacher = False

    if "student_val_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["student_val_loss"],
                  marker="s", color=ROLE_STUDENT_COLOR, label="Student Val Loss")
        plotted_student = True
    if "student_train_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["student_train_loss"],
                  marker="^", linestyle="--", alpha=0.9,
                  color=ROLE_STUDENT_COLOR, label="Student Train Loss")
        plotted_student = True

    if "teacher_val_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["teacher_val_loss"],
                  marker="o", color=ROLE_TEACHER_COLOR, label="Teacher Val Loss")
        plotted_teacher = True
    if "teacher_train_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["teacher_train_loss"],
                  marker="v", linestyle="--", alpha=0.9,
                  color=ROLE_TEACHER_COLOR, label="Teacher Train Loss")
        plotted_teacher = True

    # Baseline student (if present)
    if "baseline_student_val_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["baseline_student_val_loss"],
                  marker="D", linestyle="-.", alpha=0.9,
                  color=BASELINE_STUDENT_COLOR, label="Baseline Student Val Loss")
    if "baseline_student_train_loss" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["baseline_student_train_loss"],
                  marker="x", linestyle="--", alpha=0.9,
                  color=BASELINE_STUDENT_COLOR, label="Baseline Student Train Loss")

    # KD components
    for kd_col in ["kd_loss", "distill_loss", "soft_loss", "hard_loss"]:
        if kd_col in dfm.columns and dfm[kd_col].notna().any():
            twin.plot(dfm["epoch"], dfm[kd_col],
                      linestyle=":", color=COL_KD, label=kd_col.replace("_", " ").title())

    # Axis cosmetics for right y depending on what we plotted
    if any(c in dfm.columns for c in [
        "baseline_student_val_loss", "baseline_student_train_loss"
    ]):
        right_col = RIGHT_NEUTRAL
    elif plotted_student and not plotted_teacher:
        right_col = ROLE_STUDENT_COLOR
    elif plotted_teacher and not plotted_student:
        right_col = ROLE_TEACHER_COLOR
    else:
        right_col = RIGHT_NEUTRAL

    twin.set_ylabel("Loss")
    twin.yaxis.grid(True, alpha=0.6, zorder=-1)
    _colorize_axes(ax_time, twin, COL_TIME, right_col)

    # X formatting
    pad = 0.15
    ax_time.set_xlim(e_min - pad, e_max + pad)
    ax_time.set_xticks(range(e_min, e_max + 1))
    ax_time.set_xlabel("Epoch")
    ax_time.xaxis.grid(True, alpha=0.6, zorder=-1)

    # Best marker for student val loss (if present)
    if "student_val_loss" in dfm.columns:
        be, bl = _best_epoch_min(dfm, "student_val_loss")
        if be is not None:
            twin.scatter([be], [bl], s=160, marker="*", color=COL_BEST,
                         edgecolors="white", linewidths=0.8,
                         label=f"Best Student Val Loss: {bl:.3f} @ {be}", zorder=5)

    # Legend & titles
    handles, labels = [], []
    for a in (ax_time, twin):
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    ax_time.legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    exp_id = exp_id_from_config(label, default=experiment_name_from_log_or_paths(None, RUN_DIRS.get(label, "")))
    subtitle = hp_tail_from_config(cfg)
    total_time = _fmt_hours(float(dfm.get("time_per_epoch", pd.Series(dtype=float)).sum()))
    fig.suptitle(f"{exp_id}: Losses vs Epoch", fontsize=14, fontweight="bold", y=0.94)
    fig.text(0.5, 0.875, subtitle, ha="center", fontsize=9)
    fig.text(0.5, 0.03, f"Total train time: {total_time}", ha="center", fontsize=8)

    _sync_left_ticks_to_right_grid(ax_time, twin)
    _save_fig(fig, stem=exp_id, basename="losses_teacher_student", folder=folder)


def _choose_accuracy_role_and_columns(dfm: pd.DataFrame, cfg: dict):
    """
    Decide which single role to plot ('teacher' or 'student') and
    return the column names for (val_acc_col, train_acc_col).
    """
    has_t = ("teacher_val_accuracy" in dfm.columns) or ("teacher_train_accuracy" in dfm.columns)
    has_s = ("student_val_accuracy" in dfm.columns) or ("student_train_accuracy" in dfm.columns)

    inferred = infer_run_role(cfg) or None

    if has_t and not has_s:
        role = "teacher"
        val_acc = "teacher_val_accuracy" if "teacher_val_accuracy" in dfm.columns else None
        trn_acc = "teacher_train_accuracy" if "teacher_train_accuracy" in dfm.columns else None
    elif has_s and not has_t:
        role = "student"
        val_acc = "student_val_accuracy" if "student_val_accuracy" in dfm.columns else None
        trn_acc = "student_train_accuracy" if "student_train_accuracy" in dfm.columns else None
    elif has_t and has_s:
        role = inferred or "teacher"
        if role == "teacher":
            val_acc = "teacher_val_accuracy" if "teacher_val_accuracy" in dfm.columns else None
            trn_acc = "teacher_train_accuracy" if "teacher_train_accuracy" in dfm.columns else None
        else:
            val_acc = "student_val_accuracy" if "student_val_accuracy" in dfm.columns else None
            trn_acc = "student_train_accuracy" if "student_train_accuracy" in dfm.columns else None
    else:
        # Neither role-specific exists → use generics, mapped to inferred/default role
        role = inferred or "student"
        val_acc = "val_accuracy" if "val_accuracy" in dfm.columns else None
        trn_acc = "train_accuracy" if "train_accuracy" in dfm.columns else None

    return role, val_acc, trn_acc


def plot_accuracy_teacher_student(label, dfm, cfg, folder):
    # KD mode? -> show both roles if possible
    ts_cfg = ((cfg.get("training") or {}).get("teacher_student") or {})
    kd_mode = bool(ts_cfg.get("train_student")) or \
              (("teacher_val_accuracy" in dfm.columns or "teacher_train_accuracy" in dfm.columns) and
               ("student_val_accuracy" in dfm.columns or "student_train_accuracy" in dfm.columns))

    epochs = dfm["epoch"].astype(int).to_numpy()
    e_min, e_max = int(epochs.min()), int(epochs.max())

    fig = plt.figure(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
    ax_time = fig.add_subplot(111)
    twin = ax_time.twinx()

    # Left: time per epoch
    if "time_per_epoch" in dfm.columns and dfm["time_per_epoch"].notna().any():
        t = dfm["time_per_epoch"] / 60.0
        tmin, tmax = float(t.min()), float(t.max())
        pad_y = max(1e-6, 0.02 * (tmax - tmin + 1e-9))
        ax_time.set_ylim(tmin - pad_y, tmax + pad_y)
        ax_time.plot(dfm["epoch"], t, marker="o", color=COL_TIME, label="Time/epoch (min)")
        ax_time.set_ylabel("Time per epoch (min)")
        ax_time.tick_params(axis="y", colors=COL_TIME)

    def plot_role(role: str):
        color = ROLE_TEACHER_COLOR if role == "teacher" else ROLE_STUDENT_COLOR
        vcol = f"{role}_val_accuracy" if f"{role}_val_accuracy" in dfm.columns else None
        tcol = f"{role}_train_accuracy" if f"{role}_train_accuracy" in dfm.columns else None
        if vcol is not None:
            twin.plot(dfm["epoch"], dfm[vcol], marker="s",
                      label=f"{role.capitalize()} Val Acc (%)", color=color)
            be, ba = _best_epoch_max_with_tie_last(
                dfm[["epoch", vcol]].rename(columns={vcol: "acc"}), "acc"
            )
            if be is not None and role == "student":
                twin.scatter([be], [ba], s=160, marker="*", color=COL_BEST,
                             edgecolors="white", linewidths=0.8,
                             label=f"Best Student Val Acc: {ba:.2f}% @ {be}", zorder=5)
        if tcol is not None:
            twin.plot(dfm["epoch"], dfm[tcol], marker="^", linestyle="--", alpha=0.9,
                      label=f"{role.capitalize()} Train Acc (%)", color=color)
        return color, (vcol is not None) or (tcol is not None)

    if kd_mode:
        # Plot BOTH roles (teacher + student)
        col_t, drawn_t = plot_role("teacher")
        col_s, drawn_s = plot_role("student")
        right_col = RIGHT_NEUTRAL if (drawn_t and drawn_s) else (col_t if drawn_t else col_s)
    else:
        # Single-role fallback decided from config/columns
        role = infer_run_role(cfg) or ("teacher" if "teacher_val_accuracy" in dfm.columns or "teacher_train_accuracy" in dfm.columns else "student")
        col, _ = plot_role(role)
        right_col = col

    # Baseline student accuracy
    baseline_drawn = False
    if "baseline_student_val_accuracy" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["baseline_student_val_accuracy"],
                  marker="D", linestyle="-.", alpha=0.9,
                  label="Baseline Student Val Acc (%)", color=BASELINE_STUDENT_COLOR)
        baseline_drawn = True
    if "baseline_student_train_accuracy" in dfm.columns:
        twin.plot(dfm["epoch"], dfm["baseline_student_train_accuracy"],
                  marker="x", linestyle="--", alpha=0.9,
                  label="Baseline Student Train Acc (%)", color=BASELINE_STUDENT_COLOR)
        baseline_drawn = True
    if baseline_drawn:
        right_col = RIGHT_NEUTRAL

    twin.set_ylabel("Accuracy (%)")
    twin.yaxis.grid(True, alpha=0.6, zorder=-1)
    _colorize_axes(ax_time, twin, COL_TIME, right_col)

    # X formatting
    pad = 0.15
    ax_time.set_xlim(e_min - pad, e_max + pad)
    ax_time.set_xticks(range(e_min, e_max + 1))
    ax_time.set_xlabel("Epoch")
    ax_time.xaxis.grid(True, alpha=0.6, zorder=-1)

    # Legend (only what was drawn)
    handles, labels_ = [], []
    for a in (ax_time, twin):
        h, l = a.get_legend_handles_labels()
        handles += h; labels_ += l
    ax_time.legend(handles, labels_, loc="lower right", fontsize=9, frameon=True)

    # Titles / subtitles
    exp_id = exp_id_from_config(label, default=experiment_name_from_log_or_paths(None, RUN_DIRS.get(label, "")))
    subtitle = hp_tail_from_config(cfg)
    total_time = _fmt_hours(float(dfm.get("time_per_epoch", pd.Series(dtype=float)).sum()))
    fig.suptitle(f"{exp_id}: Accuracy vs Epoch (Teacher–Student)", fontsize=14, fontweight="bold", y=0.94)
    fig.text(0.5, 0.875, subtitle, ha="center", fontsize=9)
    fig.text(0.5, 0.03, f"Total train time: {total_time}", ha="center", fontsize=8)

    _sync_left_ticks_to_right_grid(ax_time, twin)
    _save_fig(fig, stem=exp_id, basename="accuracy_teacher_student", folder=folder)



def plot_gradnorm_with_time(label, dfm, cfg, folder):
    has_avg = "avg_gradient_norm" in dfm.columns
    has_last = "gradient_norm_last_step" in dfm.columns

    for col in ("avg_gradient_norm", "gradient_norm_last_step"):
        if col in dfm.columns:
            dfm[col] = pd.to_numeric(dfm[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    if not has_avg and not has_last and _get_steps_for_label(label).empty:
        print(f"{label}: no gradient norm data.")
        return

    epochs = dfm["epoch"].astype(int).to_numpy()
    e_min, e_max = int(epochs.min()), int(epochs.max())

    fig = plt.figure(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
    ax_time = fig.add_subplot(111)
    twin = ax_time.twinx()

    if "time_per_epoch" in dfm.columns and dfm["time_per_epoch"].notna().any():
        t = dfm["time_per_epoch"] / 60.0
        tmin, tmax = float(t.min()), float(t.max())
        pad_y = max(1e-6, 0.02 * (tmax - tmin + 1e-9))
        ax_time.set_ylim(tmin - pad_y, tmax + pad_y)
        ax_time.plot(dfm["epoch"], t, marker="o", color=COL_TIME, label="Time/epoch (min)")
        ax_time.set_ylabel("Time per epoch (min)")
        ax_time.tick_params(axis="y", colors=COL_TIME)

    plotted = False
    if has_avg:
        twin.plot(dfm["epoch"], dfm["avg_gradient_norm"], marker="s", color=ROLE_STUDENT_COLOR, label="Avg grad norm")
        plotted = True
    if has_last:
        twin.plot(dfm["epoch"], dfm["gradient_norm_last_step"], marker="^", linestyle="--",
                  color=ROLE_STUDENT_COLOR, alpha=0.85, label="Last-step grad norm")
        plotted = True

    steps_df = _get_steps_for_label(label)
    if not steps_df.empty and "grad_norm" in steps_df.columns:
        steps_df = steps_df.copy()
        steps_df["grad_norm"] = pd.to_numeric(steps_df["grad_norm"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        steps_df = steps_df.dropna(subset=["grad_norm"])
        steps_df = _steps_with_fractional_epoch(steps_df)
        if len(steps_df) > 4000:
            thinned = []
            for _, sub in steps_df.groupby("epoch", sort=True):
                n = len(sub)
                stride = max(1, n // max(1, int(np.ceil(n * 4000 / len(steps_df)))))
                thinned.append(sub.iloc[::stride])
            steps_df = pd.concat(thinned, ignore_index=True)

        if not steps_df.empty:
            twin.plot(steps_df["epoch_frac"], steps_df["grad_norm"], linewidth=0.9, alpha=0.3,
                      label="Grad norm (step)")
            plotted = True

    if not plotted:
        print(f"{label}: gradient norms present but nothing finite to plot.")
        return

    twin.set_ylabel("Gradient Norm")
    twin.yaxis.grid(True, alpha=0.6, zorder=-1)
    _colorize_axes(ax_time, twin, COL_TIME, ROLE_STUDENT_COLOR)

    pad = 0.15
    ax_time.set_xlim(e_min - pad, e_max + pad)
    ax_time.set_xticks(range(e_min, e_max + 1))
    ax_time.set_xlabel("Epoch")
    ax_time.xaxis.grid(True, alpha=0.6, zorder=-1)

    handles, labels = [], []
    for a in (ax_time, twin):
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    ax_time.legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    exp_id = exp_id_from_config(label, default=experiment_name_from_log_or_paths(None, RUN_DIRS.get(label, "")))
    subtitle = hp_tail_from_config(cfg)
    total_time = _fmt_hours(float(dfm.get("time_per_epoch", pd.Series(dtype=float)).sum()))
    fig.suptitle(f"{exp_id}: Gradient Norm vs Epoch", fontsize=14, fontweight="bold", y=0.94)
    fig.text(0.5, 0.89, subtitle, ha="center", fontsize=9)
    fig.text(0.5, 0.03, f"Total train time: {total_time}", ha="center", fontsize=8)

    _sync_left_ticks_to_right_grid(ax_time, twin)
    _save_fig(fig, stem=exp_id, basename="grad_norm", folder=folder)


def plot_mse_optional(label, dfm, cfg, folder):
    mse_col = None
    for c in ["MSE", "mse", "student_mse", "teacher_mse"]:
        if c in dfm.columns:
            mse_col = c
            break
    if mse_col is None:
        return

    epochs = dfm["epoch"].astype(int).to_numpy()
    e_min, e_max = int(epochs.min()), int(epochs.max())

    fig = plt.figure(figsize=(11.5, 4.8))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.14)
    ax_time = fig.add_subplot(111)
    twin = ax_time.twinx()

    if "time_per_epoch" in dfm.columns and dfm["time_per_epoch"].notna().any():
        t = dfm["time_per_epoch"] / 60.0
        tmin, tmax = float(t.min()), float(t.max())
        pad_y = max(1e-6, 0.02 * (tmax - tmin + 1e-9))
        ax_time.set_ylim(tmin - pad_y, tmax + pad_y)
        ax_time.plot(dfm["epoch"], t, marker="o", color=COL_TIME, label="Time/epoch (min)")
        ax_time.set_ylabel("Time per epoch (min)")
        ax_time.tick_params(axis="y", colors=COL_TIME)

    twin.plot(dfm["epoch"], dfm[mse_col], marker="s", color=ROLE_STUDENT_COLOR, label=mse_col.upper())
    twin.set_ylabel(mse_col.upper())
    twin.yaxis.grid(True, alpha=0.6, zorder=-1)
    _colorize_axes(ax_time, twin, COL_TIME, ROLE_STUDENT_COLOR)

    pad = 0.15
    ax_time.set_xlim(e_min - pad, e_max + pad)
    ax_time.set_xticks(range(e_min, e_max + 1))
    ax_time.set_xlabel("Epoch")
    ax_time.xaxis.grid(True, alpha=0.6, zorder=-1)

    handles, labels = [], []
    for a in (ax_time, twin):
        h, l = a.get_legend_handles_labels()
        handles += h; labels += l
    ax_time.legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    exp_id = exp_id_from_config(label, default=experiment_name_from_log_or_paths(None, RUN_DIRS.get(label, "")))
    subtitle = hp_tail_from_config(cfg)
    total_time = _fmt_hours(float(dfm.get("time_per_epoch", pd.Series(dtype=float)).sum()))
    fig.suptitle(f"{exp_id}: {mse_col.upper()} vs Epoch", fontsize=14, fontweight="bold", y=0.94)
    fig.text(0.5, 0.89, subtitle, ha="center", fontsize=9)
    fig.text(0.5, 0.03, f"Total train time: {total_time}", ha="center", fontsize=8)

    _sync_left_ticks_to_right_grid(ax_time, twin)
    _save_fig(fig, stem=exp_id, basename="MSE", folder=folder)


# ---------------------------------------------------------------------------------
# Entry point used by your training script
# ---------------------------------------------------------------------------------
def save_training_plots(
    experiment_dir: Path,
    experiment_name: str,
    run_id: int | None,
    *,
    base_dir: Path | str = Path("."),
    experiments_root: str = "experiments",  # default to lowercase
) -> None:
    """
    Generates plots and saves them under:
        <experiments>/<experiment_name>/Run_<id>/plots/
    """
    experiment_dir = Path(experiment_dir)
    if experiment_dir.name.startswith("Run_"):
        run_dir = experiment_dir
    else:
        # Robust path selection with case fallback
        base_dir = Path(base_dir)
        exp_root = base_dir / experiments_root / experiment_name
        if not exp_root.exists():
            low = base_dir / experiments_root.lower() / experiment_name
            upp = base_dir / experiments_root.upper() / experiment_name
            exp_root = low if low.exists() else (upp if upp.exists() else exp_root)
        run_dir = _pick_run_dir(exp_root, run_id)

    m = re.fullmatch(r"Run_(\d+)", run_dir.name)
    resolved_run_id = int(m.group(1)) if m else int(run_id or -1)

    config_path = run_dir / "config.toml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.toml at {config_path}")
    cfg = _load_toml(config_path)

    # --------- clear registries ----------
    RUN_DIRS.clear(); PLOT_DIRS.clear(); METRICS_PATHS.clear()
    LOG_FILES.clear(); CUSTOM_LABELS.clear()
    EXPERIMENTS.clear(); CONFIGS.clear()
    # -------------------------------------

    CONFIGS[experiment_name] = cfg
    do_not_log_steps = bool(cfg.get("training", {}).get("do_not_log_steps", False))

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json at {metrics_path}")

    label = experiment_name

    plot_dir = os.path.join(str(run_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    PLOT_DIRS[label] = plot_dir
    RUN_DIRS[label] = str(run_dir)
    METRICS_PATHS[label] = str(metrics_path)

    log_path: Optional[Path] = None
    if not do_not_log_steps:
        log_path = _find_log_file(run_dir, experiment_name, resolved_run_id)
        if log_path:
            LOG_FILES.append(str(log_path))
            CUSTOM_LABELS[str(log_path)] = label

    EXPERIMENTS.append({
        "label": label,
        "run_dir": str(run_dir),
        "metrics": str(metrics_path),
        "log": str(log_path) if log_path else None,
        "status": "READY" if metrics_path else "MISSING",
    })

    # Parse logs (step/epoch extras)
    global parsed
    parsed = parse_logs(LOG_FILES)

    # Load metrics (pass LABEL so we can read CONFIGS to infer role)
    dfm = load_metrics_df(label).sort_values("epoch")

    # --- Teacher merge: resolve metrics.json from teacher_model_path
    teacher_metrics_path = _guess_teacher_metrics_path(cfg, Path(base_dir))
    print(f"[plots] Teacher metrics path: {teacher_metrics_path}")
    if teacher_metrics_path is not None:
        try:
            df_teacher = pd.read_json(teacher_metrics_path)
            df_teacher = _role_alias(df_teacher, role="teacher")

            # keep epoch and teacher_* fields only (avoid collisions)
            keep_t = ["epoch"] + [c for c in df_teacher.columns if c.startswith("teacher_")]
            # optional KD components if they exist in teacher metrics
            kd_cols = [c for c in ["kd_loss", "distill_loss", "soft_loss", "hard_loss"] if c in df_teacher.columns]
            keep_t += kd_cols

            df_teacher = df_teacher[sorted(set(keep_t))].copy()

            # ensure student df has role-mapped fields too (in case only generics existed)
            dfm = _role_alias(dfm, role="student")

            # outer-merge on epoch to align possibly different ranges
            dfm = pd.merge(dfm, df_teacher, on="epoch", how="outer").sort_values("epoch").reset_index(drop=True)

            # fill KD extras from teacher if student doesn't have them
            for kc in ["kd_loss", "distill_loss", "soft_loss", "hard_loss"]:
                if kc in df_teacher.columns and kc not in dfm.columns:
                    dfm[kc] = df_teacher[kc]

        except Exception as e:
            print(f"Warning: could not merge teacher metrics from {teacher_metrics_path}: {e}")
    else:
        print("Info: teacher_model_path not found in config or teacher metrics.json missing.")

    # --- Baseline student merge: resolve metrics.json from baseline_student
    baseline_metrics_path = _guess_baseline_metrics_path(cfg, Path(base_dir))
    print(f"[plots] Baseline student metrics path: {baseline_metrics_path}")
    if baseline_metrics_path is not None:
        try:
            df_base = pd.read_json(baseline_metrics_path)
            # map generics -> student_* first, then prefix with baseline_*
            df_base = _role_alias(df_base, role="student")

            # keep epoch + the student_* columns, then rename to baseline_student_*
            keep_b = ["epoch"] + [c for c in df_base.columns if c.startswith("student_")]
            df_base = df_base[sorted(set(keep_b))].copy()
            rename_map = {c: c.replace("student_", "baseline_student_") for c in df_base.columns if c.startswith("student_")}
            df_base = df_base.rename(columns=rename_map)

            # ensure current dfm has student_* aliases
            dfm = _role_alias(dfm, role="student")

            # outer merge on epoch to align ranges
            dfm = pd.merge(dfm, df_base, on="epoch", how="outer").sort_values("epoch").reset_index(drop=True)

        except Exception as e:
            print(f"Warning: could not merge baseline metrics from {baseline_metrics_path}: {e}")
    else:
        print("Info: baseline_student not found in config or metrics.json missing.")

    # If logs have epoch summaries, prefer merging any missing fields into dfm
    ep_from_logs = parsed.get(label, {}).get("epochs", pd.DataFrame())
    if not ep_from_logs.empty:
        dfm = dfm.merge(ep_from_logs.drop(columns=["timestamp"]), on="epoch", how="outer", suffixes=("", "_log"))
        for c in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
            if c not in dfm.columns and f"{c}_log" in dfm.columns:
                dfm[c] = dfm[f"{c}_log"]
        dfm = dfm.drop(columns=[c for c in dfm.columns if c.endswith("_log")], errors="ignore")

    # Merge teacher/student extras from logs if exist
    extra_df = parsed.get(label, {}).get("extra", pd.DataFrame())
    if not extra_df.empty and "timestamp" in extra_df.columns:
        extra_df = extra_df.drop(columns=["timestamp"])
        for c in extra_df.columns:
            if c not in dfm.columns:
                dfm[c] = pd.Series(dtype=float)
        dfm = dfm.reset_index(drop=True)
        extra_df = extra_df.reset_index(drop=True)
        for c in extra_df.columns:
            dfm[c] = dfm[c].fillna(pd.Series(extra_df[c]).reindex_like(dfm).ffill())

    # Plots
    plot_losses_teacher_student(label, dfm, cfg, str(resolved_run_id), plot_dir)
    plot_accuracy_teacher_student(label, dfm, cfg, plot_dir)
    plot_gradnorm_with_time(label, dfm, cfg, plot_dir)
    plot_mse_optional(label, dfm, cfg, plot_dir)
