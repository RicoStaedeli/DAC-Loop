import json
import logging
import math
import re
import tomllib
from pathlib import Path
from typing import Dict, List, Optional, Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml

log = logging.getLogger(__name__)


# ======================================================================
# Shared helpers
# ======================================================================

def _extract_run_number(name: str) -> int:
    """
    Extract integer suffix from folder names like:
      Run_1 → 1
      Run_12 → 12
    Returns 0 if not found.
    """
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0


def _ensure_parent_dir(path: Path) -> None:
    """Ensure the parent directory of a path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_toml(path: Path) -> Dict:
    """Read a TOML file into a dict."""
    with path.open("rb") as f:
        return tomllib.load(f)


def _build_markdown_from_rows(headers: List[str], rows: List[Dict[str, str]]) -> str:
    """
    Build a markdown table from headers and list of dict rows.
    Missing keys are treated as empty strings.
    """
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = []
    for row in rows:
        row_values = [str(row.get(h, "")) for h in headers]
        row_lines.append("| " + " | ".join(row_values) + " |")

    return "\n".join([header_line, separator_line] + row_lines) + "\n"


def _get_best_val_record_from_jsonl(metrics_path: Path, val_key: str = "val_accuracy") -> Dict:
    """
    Scan a JSONL metrics file and return the full record (dict) for the epoch
    with the best validation value for 'val_key'.

    Expected keys per JSON line:
      - train_loss
      - train_accuracy
      - val_loss
      - val_accuracy
      - val_MSE
      - epoch
      - time_per_epoch
    """
    best_record = None
    best_val = None

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            if val_key not in record:
                continue

            val = record[val_key]
            if val is None:
                continue

            if (best_val is None) or (val > best_val):
                best_val = val
                best_record = record

    if best_record is None:
        raise ValueError(
            f"Could not find any valid '{val_key}' entries in metrics file: {metrics_path}"
        )

    return best_record


def _get_best_val_accuracy_from_jsonl(
        metrics_path: Path,
        val_key: str = "val_accuracy",
) -> tuple[float, int]:
    """
    Return (best_val_accuracy, best_epoch) from a JSONL metrics file.
    Uses _get_best_val_record_from_jsonl under the hood.
    """
    best_record = _get_best_val_record_from_jsonl(metrics_path, val_key)
    best_val = best_record.get(val_key)
    best_epoch = best_record.get("epoch")

    if best_val is None or best_epoch is None:
        raise ValueError(
            f"Could not find any valid (epoch, {val_key}) pairs in metrics file: {metrics_path}"
        )

    return best_val, best_epoch


def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)


def _fmt_float(x, nd: int = 2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)


def _fmt_percent(x, nd: int = 2):
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return str(x)


# ======================================================================
# Baseline experiments
# ======================================================================

def summarize_baseline_experiment_group_to_markdown(
        experiment_dir: str,
        output_md_path: str,
) -> None:
    """
    Summarize one or more *baseline* runs in an experiment directory into a
    single markdown table.

    Expected structures:

      Case 1: multiple runs
        experiments/results/Experiment_1/
          Run_1/
            metrics.jsonl
            config.toml
          Run_2/
            metrics.jsonl
            config.toml
          ...

      Case 2: single run, no Run_X subfolders
        experiments/results/Experiment_1/
          metrics.jsonl
          config.toml
    """
    experiment_path = Path(experiment_dir)
    output_path = Path(output_md_path)

    if not experiment_path.is_dir():
        raise NotADirectoryError(f"{experiment_dir} is not a directory")

    # All subdirectories (Run_1, Run_2, ...)
    run_dirs = [d for d in experiment_path.iterdir() if d.is_dir()]

    # If there are no subdirs but we do have metrics.jsonl/config.toml,
    # treat the folder itself as a single run.
    if not run_dirs and (experiment_path / "metrics.jsonl").is_file():
        run_dirs = [experiment_path]

    run_dirs = sorted(run_dirs, key=lambda d: _extract_run_number(d.name))

    rows: List[Dict[str, str]] = [
        _summarize_single_baseline_run(run_dir) for run_dir in run_dirs
    ]

    if not rows:
        raise RuntimeError(f"No valid baseline runs found in {experiment_dir}")

    headers = [
        "run_name",
        "experiment_name",
        "run_id",
        "model",
        "dataset",
        "augmentation",
        "indices_config",
        "epochs",
        "learning_rate",
        "batch_size",
        "weight_decay",
        "scheduler",
        "seed",
        "best_epoch",
        "best_train_loss",
        "best_val_loss",
        "best_train_accuracy (%)",
        "best_val_accuracy (%)",
        "best_val_MSE",
    ]

    markdown_content = _build_markdown_from_rows(headers, rows)
    _ensure_parent_dir(output_path)
    output_path.write_text(markdown_content, encoding="utf-8")


def _summarize_single_baseline_run(run_dir: Path) -> Dict[str, str]:
    """
    Given a baseline run directory (e.g. experiments/results/Experiment_1/Run_1), extract:
      - config.toml
      - metrics.jsonl

    and return a dict with the values needed for one markdown row.
    """
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.toml"

    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl found in {run_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.toml found in {run_dir}")

    # --- Metrics: take the record with best val_accuracy ---
    best_record = _get_best_val_record_from_jsonl(metrics_path)
    best_epoch = best_record.get("epoch", "")
    best_train_loss = best_record.get("train_loss", "")
    best_train_accuracy = best_record.get("train_accuracy", "")
    best_val_loss = best_record.get("val_loss", "")
    best_val_accuracy = best_record.get("val_accuracy", "")
    best_val_mse = best_record.get("val_MSE", "")
    best_time_per_epoch = best_record.get("time_per_epoch", "")

    # --- Config ---
    config = _read_toml(config_path)
    training_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    data_cfg = config.get("data", {})
    experiment_cfg = config.get("experiment", {})

    model_name = model_cfg.get("name", "")
    dataset = data_cfg.get("dataset_hf_repo_id", "")
    epochs = training_cfg.get("epochs", "")
    lr = training_cfg.get("learning_rate", "")
    batch_size = training_cfg.get("batch_size", "")
    weight_decay = training_cfg.get("weight_decay", "")
    scheduler = training_cfg.get("scheduler", "")
    seed = training_cfg.get("seed", "")
    experiment_name = experiment_cfg.get("name", "")
    run_id = experiment_cfg.get("run", "")
    indices_cfg = data_cfg.get("indices", {})
    augmentation = data_cfg.get("augment", "")

    calculated_indices = indices_cfg.get("calculated_indices")
    selected_bands = indices_cfg.get("selected_bands")

    def _to_list_str(v):
        if v is None or v == "":
            return ""
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(map(str, v)) + "]"
        return str(v)

    parts = []
    if calculated_indices is not None:
        parts.append(f"calc={_to_list_str(calculated_indices)}")
    if selected_bands is not None:
        parts.append(f"bands={_to_list_str(selected_bands)}")
    indices_summary = " - ".join(parts) if parts else ""

    def _fmt(v, fmt):
        if v == "" or v is None:
            return ""
        return fmt.format(v)

    return {
        "run_name": run_dir.name,
        "experiment_name": str(experiment_name),
        "run_id": str(run_id),
        "model": str(model_name),
        "dataset": str(dataset),
        "augmentation": str(augmentation),
        "indices_config": indices_summary,
        "epochs": str(epochs),
        "learning_rate": str(lr),
        "batch_size": str(batch_size),
        "weight_decay": str(weight_decay),
        "scheduler": str(scheduler),
        "seed": str(seed),
        "best_epoch": str(best_epoch),
        "best_train_loss": _fmt(best_train_loss, "{:.6f}"),
        "best_train_accuracy (%)": _fmt(best_train_accuracy, "{:.4f}"),
        "best_val_loss": _fmt(best_val_loss, "{:.6f}"),
        "best_val_accuracy (%)": _fmt(best_val_accuracy, "{:.4f}"),
        "best_val_MSE": _fmt(best_val_mse, "{:.6f}"),
        "time_per_epoch (s)": _fmt(best_time_per_epoch, "{:.4f}"),
    }


# ======================================================================
# Pruning / compression experiments
# ======================================================================

def extract_relative_metrics_from_file(log_path: str) -> Dict[str, Optional[float]]:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    def find_indices(needle: str) -> List[int]:
        pat = re.compile(re.escape(needle), re.IGNORECASE)
        return [i for i, ln in enumerate(lines) if pat.search(ln)]

    anchors = {
        "before_pruning": find_indices("BEFORE pruning."),
        "pruned_before_ft": find_indices("PRUNED model before fine-tuning."),
        "pruned_finetuned": find_indices("Pruned + fine-tuned."),
    }

    acc_re = re.compile(r"\bAcc\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*%")
    sparsity_re = re.compile(
        r"Overall\s+sparsity\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%",
        re.IGNORECASE,
    )

    def parse_accuracy(s: str) -> Optional[float]:
        m = acc_re.search(s)
        return float(m.group(1)) if m else None

    def parse_sparsity(s: str) -> Optional[float]:
        m = sparsity_re.search(s)
        return float(m.group(1)) if m else None

    metrics: Dict[str, Optional[float]] = {
        "Accuracy Before Pruning (%)": None,
        "Sparsity (Pruned, Before Fine-Tune) (%)": None,
        "Accuracy After Pruning (Before Fine-Tune) (%)": None,
        "Sparsity After Pruning (Post Fine-Tune) (%)": None,
        "Accuracy Pruned + Fine-Tuned (%)": None,
    }

    # 1) after BEFORE pruning.
    for idx in anchors["before_pruning"]:
        if idx + 1 < len(lines):
            v = parse_accuracy(lines[idx + 1])
            if v is not None:
                metrics["Accuracy Before Pruning (%)"] = v
                break

    # 2) before PRUNED model before fine-tuning. -> sparsity
    for idx in anchors["pruned_before_ft"]:
        if idx - 1 >= 0:
            v = parse_sparsity(lines[idx - 1])
            if v is not None:
                metrics["Sparsity (Pruned, Before Fine-Tune) (%)"] = v
                break

    # 3) after PRUNED model before fine-tuning. -> accuracy
    for idx in anchors["pruned_before_ft"]:
        if idx + 1 < len(lines):
            v = parse_accuracy(lines[idx + 1])
            if v is not None:
                metrics["Accuracy After Pruning (Before Fine-Tune) (%)"] = v
                break

    # 4) before Pruned + fine-tuned. -> sparsity
    for idx in anchors["pruned_finetuned"]:
        if idx - 1 >= 0:
            v = parse_sparsity(lines[idx - 1])
            if v is not None:
                metrics["Sparsity After Pruning (Post Fine-Tune) (%)"] = v
                break

    # 5) after Pruned + fine-tuned. -> accuracy
    for idx in anchors["pruned_finetuned"]:
        if idx + 1 < len(lines):
            v = parse_accuracy(lines[idx + 1])
            if v is not None:
                metrics["Accuracy Pruned + Fine-Tuned (%)"] = v
                break

    return metrics


def extract_params_last_block(log_path: str) -> Dict[str, Optional[float]]:
    """
    Looks for lines that START with:
      - 'Total params:'
      - 'Trainable params:'
      - 'Estimated Total Size'
    Groups rows by encountering 'Total params:'.
    Returns the LAST group's values (common in per-run logs).
    """
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    int_re = re.compile(r"[-+]?\d[\d,]*")
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?")

    def lstarts(line: str, prefix: str) -> bool:
        return line.lstrip().startswith(prefix)

    def extract_int(s: str) -> Optional[int]:
        m = int_re.search(s)
        return int(m.group(0).replace(",", "")) if m else None

    def extract_float(s: str) -> Optional[float]:
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    rows: List[Dict[str, Optional[float]]] = []
    current: Optional[Dict[str, Optional[float]]] = None

    for raw in lines:
        line = raw.rstrip("\n")

        if lstarts(line, "Total params:"):
            if current and any(v is not None for v in current.values()):
                rows.append(current)
            current = {
                "Total Params": extract_int(line),
                "Trainable Params": None,
                "Estimated Total Size (MB)": None,
            }
            continue

        if lstarts(line, "Trainable params:"):
            if current is None:
                current = {
                    "Total Params": None,
                    "Trainable Params": None,
                    "Estimated Total Size (MB)": None,
                }
            current["Trainable Params"] = extract_int(line)
            continue

        if lstarts(line, "Estimated Total Size"):
            if current is None:
                current = {
                    "Total Params": None,
                    "Trainable Params": None,
                    "Estimated Total Size (MB)": None,
                }
            current["Estimated Total Size (MB)"] = extract_float(line)
            continue

    if current and any(v is not None for v in current.values()):
        rows.append(current)

    if rows:
        return rows[-1]
    return {
        "Total Params": None,
        "Trainable Params": None,
        "Estimated Total Size (MB)": None,
    }


def df_to_markdown_stable(df: pd.DataFrame) -> str:
    """
    Convert a DataFrame to a Markdown table string with stable formatting.
    (Integer formatting, percentages, etc. same as before.)
    """
    # Identify columns
    int_cols = [
        c
        for c in df.columns
        if "Params" in c and "(%)" not in c and "Size (MB)" not in c
    ]
    float2_cols = [c for c in df.columns if ("Accuracy" in c or "Sparsity" in c)]
    percent_cols = [
        "Accuracy Before Pruning (%)",
        "Sparsity (Pruned, Before Fine-Tune) (%)",
        "Accuracy After Pruning (Before Fine-Tune) (%)",
        "Sparsity After Pruning (Post Fine-Tune) (%)",
        "Accuracy Pruned + Fine-Tuned (%)",
        "Reduction Student to pruned Student (%)",
        "Reduction Teacher to pruned Student (%)",
    ]
    float4_cols = [c for c in df.columns if "Size (MB)" in c]

    df_disp = pd.DataFrame(index=df.index, columns=df.columns, dtype=object)

    for c in df.columns:
        if c in int_cols:
            df_disp[c] = df[c].apply(_fmt_int)
        elif c in percent_cols:
            df_disp[c] = df[c].apply(lambda v: _fmt_percent(v, 2))
        elif c in float2_cols:
            df_disp[c] = df[c].apply(lambda v: _fmt_float(v, 2))
        elif c in float4_cols:
            df_disp[c] = df[c].apply(lambda v: _fmt_float(v, 4))
        else:
            df_disp[c] = df[c].apply(
                lambda v: "" if (isinstance(v, float) and math.isnan(v)) else str(v)
            )

    headers = list(df_disp.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, r in df_disp.iterrows():
        lines.append("| " + " | ".join(r[h] for h in headers) + " |")

    return "\n".join(lines)


def print_markdown_stable(df: pd.DataFrame) -> None:
    """Backwards-compatible helper: print the Markdown table to stdout."""
    print(df_to_markdown_stable(df))




TOTAL_PARAMS_RE = re.compile(r"Total params:\s*([\d,]+)")


def _extract_total_params(log_path: Path) -> Optional[int]:
    """
    Given a path to a model summary log file, extract the integer after 'Total params:'.
    Returns None if the file does not exist or the pattern is not found.
    """
    if not log_path.is_file():
        return None

    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TOTAL_PARAMS_RE.search(line)
        if match:
            # Remove thousands separators
            return int(match.group(1).replace(",", ""))
    return None


def _get_val_acc_from_candidates(d: Dict[str, Any], *candidate_keys: str) -> Optional[float]:
    """
    Try several possible keys (e.g. 'after_pruning', 'pruned') to find a dict
    containing 'val_acc'. Returns the float or None.
    """
    for key in candidate_keys:
        sub = d.get(key)
        if isinstance(sub, dict) and "val_acc" in sub:
            return sub["val_acc"]
    return None



RUN_MARKERS = ["🔴", "🟢", "🔵", "🟡", "🟣", "🟠"]

def marker_for_run(run_id: int) -> str:
    # run_id is assumed to start at 1
    return RUN_MARKERS[(run_id - 1) % len(RUN_MARKERS)]

def insert_run_separators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert a bold separator row after each run group.
    """
    rows = []
    columns = df.columns

    unique_runs = df["Run ID"].unique()

    for run in unique_runs:
        block = df[df["Run ID"] == run]
        rows.append(block)

        # separator row (one per run)
        sep = {col: "**———**" for col in columns}
        rows.append(pd.DataFrame([sep]))

    # concatenate everything
    out = pd.concat(rows, ignore_index=True)
    return out

def summarize_pruning_experiment_multiple_iterations(
    experiment_dir: str,
    num_of_runs: Optional[int] = None,
    output_md_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a combined pruning/compression table across multiple runs and iterations.

    For each Run_X:
      - Read metrics_pruning.jsonl
      - For each line (iteration):
          * extract accuracies before/after pruning and after fine-tuning
          * load the corresponding model summary logs to get total params

    Final DataFrame columns:

      | Run ID | Iteration | Acc. before Pruning | Acc. after Pruning |
      | Acc. after fine tuning | Parameter Before pruning | Parameter after pruning |
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.is_dir():
        raise NotADirectoryError(f"{experiment_dir} is not a directory")

    combined_rows: List[Dict[str, Any]] = []

    # determine run directories
    if num_of_runs is not None:
        run_dirs = [
            experiment_path / f"Run_{i}"
            for i in range(1, num_of_runs + 1)
            if (experiment_path / f"Run_{i}").is_dir()
        ]
    else:
        run_dirs = [d for d in experiment_path.iterdir() if d.is_dir()]

    # You already have this helper somewhere:
    run_dirs = sorted(run_dirs, key=lambda d: _extract_run_number(d.name))

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_pruning.jsonl"
        if not metrics_path.is_file():
            # skip runs without metrics
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

                iteration = record.get("iteration")
                if iteration is None:
                    continue

                # --- accuracies (nested dicts) ---
                acc_before = _get_val_acc_from_candidates(record, "before_pruning")
                acc_after_pruning = _get_val_acc_from_candidates(
                    record,
                    "after_pruning",
                    "pruned",
                    "after_prune",
                )
                acc_after_ft = _get_val_acc_from_candidates(
                    record,
                    "after_finetuning",
                    "after_fine_tuning",
                    "pruned_and_finetuned",
                    "pruned_finetuned",
                )

                # round to 2 decimals if present
                acc_before = round(acc_before, 2) if acc_before is not None else None
                acc_after_pruning = (
                    round(acc_after_pruning, 2) if acc_after_pruning is not None else None
                )
                acc_after_ft = round(acc_after_ft, 2) if acc_after_ft is not None else None

                # --- parameter counts from log files (per iteration) ---
                before_log = run_dir / f"model_summary_Student - Before Pruning iteration {iteration}.log"
                after_log = run_dir / f"model_summary_Student - Pruned and Fine-tuned iteration {iteration}.log"

                params_before = _extract_total_params(before_log)
                params_after = _extract_total_params(after_log)

                row: Dict[str, Any] = {
                    "Run ID": _extract_run_number(run_dir.name),
                    "Iteration": iteration,
                    "Acc. before Pruning": acc_before,
                    "Acc. after Pruning": acc_after_pruning,
                    "Acc. after fine tuning": acc_after_ft,
                    "Parameter Before pruning": params_before,
                    "Parameter after pruning": params_after,
                }

                combined_rows.append(row)

    if not combined_rows:
        raise RuntimeError(f"No valid runs or iterations found in {experiment_dir}")

    df = pd.DataFrame(combined_rows)

    if "Parameter Before pruning" in df.columns and "Parameter after pruning" in df.columns:
        df["Param Reduction (%)"] = (
                100 * (1 - df["Parameter after pruning"] / df["Parameter Before pruning"])
        ).round(2)

    markdown_parts: List[str] = []

    # ensure runs are in numeric order
    for run_id in sorted(df["Run ID"].unique()):
        run_df = df[df["Run ID"] == run_id].copy()

        # sort iterations within each run (optional but usually nice)
        if "Iteration" in run_df.columns:
            run_df = run_df.sort_values("Iteration")

        # optional: drop Run ID column for the subtables, since it's in the title
        run_df_for_table = run_df.drop(columns=["Run ID"])

        markdown_parts.append(f"### {marker_for_run(int(run_id))} Run {int(run_id)}\n")
        markdown_parts.append(df_to_markdown_stable(run_df_for_table))
        markdown_parts.append("")  # blank line after each table

    markdown = "\n".join(markdown_parts)

    if output_md_path is not None:
        out_path = Path(output_md_path)
        _ensure_parent_dir(out_path)
        out_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)

    return df


def summarize_pruning_experiment(
    experiment_dir: str,
    num_of_runs: Optional[int] = None,
    output_md_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build a combined pruning/compression table across multiple runs and iterations.

    For each Run_X:
      - Read metrics_pruning.jsonl
      - For each line (iteration):
          * extract accuracies before/after pruning and after fine-tuning
          * load the corresponding model summary logs to get total params

    Final DataFrame columns:

      | Run ID | Iteration | Acc. before Pruning | Acc. after Pruning |
      | Acc. after fine tuning | Parameter Before pruning | Parameter after pruning |
    """
    experiment_path = Path(experiment_dir)

    if not experiment_path.is_dir():
        raise NotADirectoryError(f"{experiment_dir} is not a directory")

    combined_rows: List[Dict[str, Any]] = []

    # determine run directories
    if num_of_runs is not None:
        run_dirs = [
            experiment_path / f"Run_{i}"
            for i in range(1, num_of_runs + 1)
            if (experiment_path / f"Run_{i}").is_dir()
        ]
    else:
        run_dirs = [d for d in experiment_path.iterdir() if d.is_dir()]

    # You already have this helper somewhere:
    run_dirs = sorted(run_dirs, key=lambda d: _extract_run_number(d.name))

    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics_pruning.jsonl"
        if not metrics_path.is_file():
            # skip runs without metrics
            continue

        with metrics_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # skip malformed lines
                    continue

                iteration = record.get("iteration")
                if iteration is None:
                    continue

                # --- accuracies (nested dicts) ---
                acc_before = _get_val_acc_from_candidates(record, "before_pruning")
                acc_after_pruning = _get_val_acc_from_candidates(
                    record,
                    "after_pruning",
                    "pruned",
                    "after_prune",
                )
                acc_after_ft = _get_val_acc_from_candidates(
                    record,
                    "after_finetuning",
                    "after_fine_tuning",
                    "pruned_and_finetuned",
                    "pruned_finetuned",
                )

                # round to 2 decimals if present
                acc_before = round(acc_before, 2) if acc_before is not None else None
                acc_after_pruning = (
                    round(acc_after_pruning, 2) if acc_after_pruning is not None else None
                )
                acc_after_ft = round(acc_after_ft, 2) if acc_after_ft is not None else None

                # --- parameter counts from log files (per iteration) ---
                before_log = run_dir / f"model_summary_Student - Before Pruning iteration {iteration}.log"
                after_log = run_dir / f"model_summary_Student - Pruned and Fine-tuned iteration {iteration}.log"

                params_before = _extract_total_params(before_log)
                params_after = _extract_total_params(after_log)

                row: Dict[str, Any] = {
                    "Run ID": _extract_run_number(run_dir.name),
                    "Iteration": iteration,
                    "Acc. before Pruning": acc_before,
                    "Acc. after Pruning": acc_after_pruning,
                    "Acc. after fine tuning": acc_after_ft,
                    "Parameter Before pruning": params_before,
                    "Parameter after pruning": params_after,
                }

                combined_rows.append(row)

    if not combined_rows:
        raise RuntimeError(f"No valid runs or iterations found in {experiment_dir}")

    df = pd.DataFrame(combined_rows)

    if "Parameter Before pruning" in df.columns and "Parameter after pruning" in df.columns:
        df["Param Reduction (%)"] = (
            100 * (1 - df["Parameter after pruning"] / df["Parameter Before pruning"])
        ).round(2)

    markdown = df_to_markdown_stable(df)

    if output_md_path is not None:
        out_path = Path(output_md_path)
        _ensure_parent_dir(out_path)
        out_path.write_text(markdown + "\n", encoding="utf-8")
    else:
        print(markdown)

    return df


def _collect_run_points(df: pd.DataFrame, run: int):
    """Return list of (stage, params, acc) for a given run. Skips missing."""
    row = df.loc[df["Run"] == run]
    if row.empty:
        return []

    row = row.iloc[0]
    points = [
        (
            "Teacher",
            row.get("Teacher Total Params"),
            row.get("Accuracy Before Pruning (%)"),
        ),
        (
            "Student Before",
            row.get("Student Before Total Params"),
            row.get("Accuracy Before Pruning (%)"),
        ),
        (
            "Student P+FT",
            row.get("Student P+FT Total Params"),
            row.get("Accuracy Pruned + Fine-Tuned (%)"),
        ),
    ]
    return [
        (s, p, a)
        for (s, p, a) in points
        if pd.notna(p) and pd.notna(a)
    ]


def save_size_accuracy_per_run(df: pd.DataFrame, out_dir: str, log_x: bool = False):
    """
    For each run, save a plot of Accuracy (%) vs Parameters.
    - X axis: parameters (optionally log scale)
    - Y axis: accuracy (%)
    - Points: Teacher, Student Before, Student P+FT
    - A line connects the stages in that order (Teacher -> Student Before -> Student P+FT)
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = sorted(df["Run"].dropna().astype(int).unique().tolist())

    for run in runs:
        pts = _collect_run_points(df, run)
        if not pts:
            continue

        stages, params, accs = zip(*pts)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(params, accs, marker="o")

        for s, p, a in pts:
            ax.annotate(
                s,
                (p, a),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

        ax.set_title(f"Run {run}: Accuracy vs. Parameters")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy (%)")

        if log_x:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style="plain", axis="x")

        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.3)

        try:
            teacher_p = next(p for s, p, _ in pts if s == "Teacher")
            student_b_p = next(p for s, p, _ in pts if s == "Student Before")
            student_p_p = next(p for s, p, _ in pts if s == "Student P+FT")
            teacher_a = next(a for s, _, a in pts if s == "Teacher")
            student_b_a = next(a for s, _, a in pts if s == "Student Before")
            student_p_a = next(a for s, _, a in pts if s == "Student P+FT")
            subtitle = (
                f"Teacher: {_fmt_int(teacher_p)} / {_fmt_percent(teacher_a)}    |    "
                f"Student Before: {_fmt_int(student_b_p)} / {_fmt_percent(student_b_a)}    |    "
                f"Student P+FT: {_fmt_int(student_p_p)} / {_fmt_percent(student_p_a)}"
            )
            ax.text(
                0.5,
                1.02,
                subtitle,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
            )
        except StopIteration:
            pass

        fig.tight_layout()
        plot_dir = out / f"Run_{run}/plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        out_path = plot_dir / "size_vs_accuracy.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    print(f"✅ Saved size-vs-accuracy plots to: {Path(out_dir).resolve()}")


# ======================================================================
# Teacher–student KD experiments
# ======================================================================

def _summarize_single_run(run_dir: Path) -> Dict[str, str]:
    """
    Given a run directory like experiments/results/Experiment_3/Run_1, extract:
      - config.toml
      - metrics.jsonl
      - teacher metrics
      - baseline student metrics

    Returns a dict with all values needed for one markdown row.
    """
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.toml"

    if not metrics_path.is_file():
        raise FileNotFoundError(f"No metrics.jsonl found in {run_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"No config.toml found in {run_dir}")

    # --- Student (this run) ---
    student_best_val_acc, student_best_epoch = _get_best_val_accuracy_from_jsonl(
        metrics_path
    )

    # --- Config ---
    config = _read_toml(config_path)
    training_cfg = config.get("training", {})
    teacher_student_cfg = training_cfg.get("teacher_student", {})

    teacher_model = teacher_student_cfg.get("teacher_model_name", "")
    student_model = config.get("model", {}).get("name", "")
    dataset = config.get("data", {}).get("dataset_hf_repo_id", "")

    def _get_param(key: str, default=""):
        if key in training_cfg:
            return training_cfg[key]
        if key in teacher_student_cfg:
            return teacher_student_cfg[key]
        return default

    kd_type = _get_param("kd_type")
    hidden_rep_loss_weight = _get_param("hidden_rep_loss_weight")
    ce_loss_weight = _get_param("ce_loss_weight")
    soft_target_loss_weight = _get_param("soft_target_loss_weight")
    T_param = _get_param("T_param")
    alpha_kd = _get_param("alpha_kd")
    beta_feat = _get_param("beta_feat")

    # --- Teacher ---
    teacher_model_path = teacher_student_cfg.get("teacher_model_path", "")
    if not teacher_model_path:
        raise KeyError(
            f"{config_path}: missing 'training.teacher_student.teacher_model_path'"
        )

    teacher_ckpt_path = Path(teacher_model_path)
    if not teacher_ckpt_path.is_absolute():
        teacher_ckpt_path = teacher_ckpt_path.resolve()

    teacher_metrics_path = teacher_ckpt_path.parent / "metrics.jsonl"
    if not teacher_metrics_path.is_file():
        raise FileNotFoundError(
            f"Teacher metrics file not found at expected location: {teacher_metrics_path}"
        )

    teacher_best_val_acc, _ = _get_best_val_accuracy_from_jsonl(teacher_metrics_path)

    # --- Baseline student ---
    baseline_student_path = teacher_student_cfg.get("baseline_student")
    if not baseline_student_path:
        raise KeyError(
            f"{config_path}: missing 'training.teacher_student.baseline_student'"
        )

    baseline_metrics_path = Path(baseline_student_path)
    if not baseline_metrics_path.is_absolute():
        baseline_metrics_path = baseline_metrics_path.resolve()

    if not baseline_metrics_path.is_file():
        raise FileNotFoundError(
            f"Baseline student metrics file not found at expected location: {baseline_metrics_path}"
        )

    baseline_best_val_acc, _ = _get_best_val_accuracy_from_jsonl(
        baseline_metrics_path
    )

    return {
        "run_name": run_dir.name,
        "teacher_model": teacher_model,
        "student_model": student_model,
        "dataset": dataset,
        "kd_type": kd_type,
        "hidden_rep_loss_weight": hidden_rep_loss_weight,
        "ce_loss_weight": ce_loss_weight,
        "soft_target_loss_weight": soft_target_loss_weight,
        "alpha_kd": alpha_kd,
        "beta_feat": beta_feat,
        "T_param": T_param,
        "teacher_best_val_accuracy (%)": f"{teacher_best_val_acc:.4f}",
        "student_best_val_accuracy (%)": f"{student_best_val_acc:.4f}",
        "baseline_student_best_val_accuracy (%)": f"{baseline_best_val_acc:.4f}",
        "improvement_over_baseline (%)": f"{student_best_val_acc - baseline_best_val_acc:.4f}",
    }


def summarize_experiment_group_to_markdown(
        experiment_dir: str,
        output_md_path: str,
) -> None:
    """
    Combine multiple runs inside an experiment directory into a single markdown table.

    Example:
        summarize_experiment_group_to_markdown(
            experiment_dir="experiments/results/Experiment_3",
            output_md_path="docs/results/experiment_3_summary.md",
        )
    """
    experiment_path = Path(experiment_dir)
    output_path = Path(output_md_path)

    if not experiment_path.is_dir():
        raise NotADirectoryError(f"{experiment_dir} is not a directory")

    run_dirs = sorted(
        (d for d in experiment_path.iterdir() if d.is_dir()),
        key=lambda d: _extract_run_number(d.name),
    )

    rows = [_summarize_single_run(run_dir) for run_dir in run_dirs]

    if not rows:
        raise RuntimeError(f"No valid runs found in {experiment_dir}")

    headers = [
        "run_name",
        "teacher_model",
        "student_model",
        "dataset",
        "kd_type",
        "hidden_rep_loss_weight",
        "ce_loss_weight",
        "soft_target_loss_weight",
        "alpha_kd",
        "beta_feat",
        "T_param",
        "teacher_best_val_accuracy (%)",
        "student_best_val_accuracy (%)",
        "baseline_student_best_val_accuracy (%)",
        "improvement_over_baseline (%)",
    ]

    markdown_content = _build_markdown_from_rows(headers, rows)
    _ensure_parent_dir(output_path)
    output_path.write_text(markdown_content, encoding="utf-8")


# ======================================================================
# Main
# ======================================================================

def main():
    # Setup logging
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info(f"Config loaded from {logger_cfg_path}")

    # ------------------------------------------------------------------
    # Tables for Baseline Experiments
    # ------------------------------------------------------------------
    summarize_baseline_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_1",
        output_md_path="experiments/results/Experiment_1/experiment_summary.md",
    )
    summarize_baseline_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_5",
        output_md_path="experiments/results/Experiment_5/experiment_summary.md",
    )
    summarize_baseline_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_10",
        output_md_path="experiments/results/Experiment_10/experiment_summary.md",
    )
    summarize_baseline_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_10a",
        output_md_path="experiments/results/Experiment_10a/experiment_summary.md",
    )
    # ------------------------------------------------------------------
    # Tables for Teacher - Student KD Experiments
    # ------------------------------------------------------------------
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_2",
        output_md_path="experiments/results/Experiment_2/experiment_summary.md",
    )

    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_3",
        output_md_path="experiments/results/Experiment_3/experiment_summary.md",
    )

    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_4",
        output_md_path="experiments/results/Experiment_4/experiment_summary.md",
    )

    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_4a",
        output_md_path="experiments/results/Experiment_4a/experiment_summary.md",
    )

    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_6",
        output_md_path="experiments/results/Experiment_6/experiment_summary.md",
    )
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_6a",
        output_md_path="experiments/results/Experiment_6a/experiment_summary.md",
    )
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/results/Experiment_6b",
        output_md_path="experiments/results/Experiment_6b/experiment_summary.md",
    )

    # ------------------------------------------------------------------
    # Tables for Pruning/Compression Experiments
    # ------------------------------------------------------------------
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_7",
        output_md_path="experiments/results/Experiment_7/experiment_summary.md",
    )
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_7a",
        output_md_path="experiments/results/Experiment_7a/experiment_summary.md",
    )
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_11",
        output_md_path="experiments/results/Experiment_11/experiment_summary.md",
    )
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_11a",
        output_md_path="experiments/results/Experiment_11a/experiment_summary.md",
    )
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_8",
        output_md_path="experiments/results/Experiment_8/experiment_summary.md",
    )
    summarize_pruning_experiment(
        experiment_dir="experiments/results/Experiment_12",
        output_md_path="experiments/results/Experiment_12/experiment_summary.md",
    )

    # ------------------------------------------------------------------
    # Tables for Compression Loop Experiments
    # ------------------------------------------------------------------
    summarize_pruning_experiment_multiple_iterations(
        experiment_dir="experiments/results/Experiment_9",
        output_md_path="experiments/results/Experiment_9/experiment_summary.md",
    )
    summarize_pruning_experiment_multiple_iterations(
        experiment_dir="experiments/results/Experiment_9a",
        output_md_path="experiments/results/Experiment_9a/experiment_summary.md",
    )
    summarize_pruning_experiment_multiple_iterations(
        experiment_dir="experiments/results/Experiment_13",
        output_md_path="experiments/results/Experiment_13/experiment_summary.md",
    )


if __name__ == "__main__":
    main()
