from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
import logging
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import tomllib
import json

log = logging.getLogger(__name__)

def main():
    # Setup logging
    logger_cfg_path = "configs/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info(f"Config loaded from {logger_cfg_path}")

    ################################################################################################################
    ############################ Tables for Teacher - Student KD Experiments    ####################################
    ################################################################################################################
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/Experiment_3",
        output_md_path="experiments/Experiment_3/results_summary.md",
    )
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/Experiment_4",
        output_md_path="experiments/Experiment_4/results_summary.md",
    )
    summarize_experiment_group_to_markdown(
        experiment_dir="experiments/Experiment_6",
        output_md_path="experiments/Experiment_6/results_summary.md",
    )

    ################################################################################################################
    ############################ Tables for Pruning/Compression Experiments     ####################################
    ################################################################################################################
    experiment_dir = "experiments/Experiment_7"
    build_combined_table(experiment_dir, 21)




#--------- 1) Extract relative metrics (returns dict, no printing) ---------
def extract_relative_metrics_from_file(log_path: str) -> Dict[str, Optional[float]]:
    text = Path(log_path).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    def find_indices(needle: str):
        pat = re.compile(re.escape(needle), re.IGNORECASE)
        return [i for i, ln in enumerate(lines) if pat.search(ln)]

    anchors = {
        "before_pruning": find_indices("BEFORE pruning."),
        "pruned_before_ft": find_indices("PRUNED model before fine-tuning."),
        "pruned_finetuned": find_indices("Pruned + fine-tuned."),
    }

    acc_re = re.compile(r"\bAcc\s*=?\s*([0-9]+(?:\.[0-9]+)?)\s*%")
    sparsity_re = re.compile(r"Overall\s+sparsity\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)

    def parse_accuracy(s: str):
        m = acc_re.search(s)
        return float(m.group(1)) if m else None

    def parse_sparsity(s: str):
        m = sparsity_re.search(s)
        return float(m.group(1)) if m else None

    metrics = {
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

# --------- 2) Extract model params (starts-with exact phrases; returns LAST block) ---------
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

    def extract_int(s: str):
        m = int_re.search(s)
        return int(m.group(0).replace(",", "")) if m else None

    def extract_float(s: str):
        m = float_re.search(s)
        return float(m.group(0)) if m else None

    rows: List[Dict[str, Optional[float]]] = []
    current = None

    for raw in lines:
        line = raw.rstrip("\n")

        if lstarts(line, "Total params:"):
            if current and any(v is not None for v in current.values()):
                rows.append(current)
            current = {"Total Params": extract_int(line),
                       "Trainable Params": None,
                       "Estimated Total Size (MB)": None}
            continue

        if lstarts(line, "Trainable params:"):
            if current is None:
                current = {"Total Params": None,
                           "Trainable Params": None,
                           "Estimated Total Size (MB)": None}
            current["Trainable Params"] = extract_int(line)
            continue

        if lstarts(line, "Estimated Total Size"):
            if current is None:
                current = {"Total Params": None,
                           "Trainable Params": None,
                           "Estimated Total Size (MB)": None}
            current["Estimated Total Size (MB)"] = extract_float(line)
            continue

    if current and any(v is not None for v in current.values()):
        rows.append(current)

    # Return the LAST block if present, else empty dict with keys
    if rows:
        return rows[-1]
    return {"Total Params": None, "Trainable Params": None, "Estimated Total Size (MB)": None}

def _fmt_int(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{int(x):,}"
    except Exception:
        return str(x)

def _fmt_float(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return str(x)

def _fmt_percent(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return str(x)

def print_markdown_stable(df: pd.DataFrame):
    # Identify columns
    int_cols     = [c for c in df.columns if "Params" in c and "(%)" not in c and "Size (MB)" not in c]
    float2_cols  = [c for c in df.columns if ("Accuracy" in c or "Sparsity" in c)]
    percent_cols = [
        "Accuracy Before Pruning (%)",
        "Sparsity (Pruned, Before Fine-Tune) (%)",
        "Accuracy After Pruning (Before Fine-Tune) (%)",
        "Sparsity After Pruning (Post Fine-Tune) (%)",
        "Accuracy Pruned + Fine-Tuned (%)",
        "Reduction Student to pruned Student (%)",
        "Reduction Teacher to pruned Student (%)",
    ]
    float4_cols  = [c for c in df.columns if "Size (MB)" in c]

    # Build a fully-string DataFrame for display
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
            df_disp[c] = df[c].apply(lambda v: "" if (isinstance(v, float) and math.isnan(v)) else str(v))

    # Manual Markdown printing (no external deps, no dtype surprises)
    headers = list(df_disp.columns)
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, r in df_disp.iterrows():
        print("| " + " | ".join(r[h] for h in headers) + " |")

# --------- 3) Combine across runs into one DataFrame and print Markdown ---------
def build_combined_table(experiment_dir, num_of_runs):
    combined_rows = []

    for i in range(1,num_of_runs + 1):
        row: Dict[str, Optional[float]] = {"Run": i}

        # Paths as given in your snippet
        base = f"{experiment_dir}/Run_{i}"

        # a) Relative metrics
        metrics_path = f"{base}/Experiment_log.log"
        try:
            metrics = extract_relative_metrics_from_file(metrics_path)
        except FileNotFoundError:
            metrics = {
                "Accuracy Before Pruning (%)": None,
                "Sparsity (Pruned, Before Fine-Tune) (%)": None,
                "Accuracy After Pruning (Before Fine-Tune) (%)": None,
                "Sparsity After Pruning (Post Fine-Tune) (%)": None,
                "Accuracy Pruned + Fine-Tuned (%)": None,
            }
        row.update(metrics)

        # b) Teacher params
        try:
            teacher = extract_params_last_block(f"{base}/model_summary_Teacher.log")
        except FileNotFoundError:
            teacher = {"Total Params": None, "Trainable Params": None, "Estimated Total Size (MB)": None}
        row.update({
            "Teacher Total Params": teacher.get("Total Params"),
            "Teacher Trainable Params": teacher.get("Trainable Params"),
            "Teacher Estimated Total Size (MB)": teacher.get("Estimated Total Size (MB)"),
        })

        # c) Student Pruned + Fine-tuned params
        try:
            student_pft = extract_params_last_block(f"{base}/model_summary_Student - Pruned and Fine-tuned.log")
        except FileNotFoundError:
            student_pft = {"Total Params": None, "Trainable Params": None, "Estimated Total Size (MB)": None}
        row.update({
            "Student P+FT Total Params": student_pft.get("Total Params"),
            "Student P+FT Trainable Params": student_pft.get("Trainable Params"),
            "Student P+FT Estimated Total Size (MB)": student_pft.get("Estimated Total Size (MB)"),
        })

        # d) Student Before Pruning params
        try:
            student_before = extract_params_last_block(f"{base}/model_summary_Student - Before Pruning.log")
        except FileNotFoundError:
            student_before = {"Total Params": None, "Trainable Params": None, "Estimated Total Size (MB)": None}
        row.update({
            "Student Before Total Params": student_before.get("Total Params"),
            "Student Before Trainable Params": student_before.get("Trainable Params"),
            "Student Before Estimated Total Size (MB)": student_before.get("Estimated Total Size (MB)"),
        })

        combined_rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(combined_rows)

    # Ensure the ratio column exists and is float
    df["Reduction Student to pruned Student (%)"] = (1 - df["Student P+FT Total Params"] / df["Student Before Total Params"]) * 100
    df["Reduction Teacher to pruned Student (%)"] = (1 - df["Student P+FT Total Params"] / df["Teacher Total Params"]) * 100

    # Print with stable formatting (no sci-notation, % kept)
    print_markdown_stable(df)

    return df

def _fmt_int(n):
    try:
        return f"{int(n):,}"
    except Exception:
        return str(n)

def _fmt_pct(x, nd=2):
    try:
        return f"{float(x):.{nd}f}%"
    except Exception:
        return str(x)

def _collect_run_points(df: pd.DataFrame, run: int):
    """Return list of (stage, params, acc) for a given run. Skips missing."""
    row = df.loc[df["Run"] == run]
    if row.empty:
        return []

    row = row.iloc[0]
    points = [
        ("Teacher",        row.get("Teacher Total Params"),          row.get("Accuracy Before Pruning (%)")),
        ("Student Before", row.get("Student Before Total Params"),   row.get("Accuracy Before Pruning (%)")),
        ("Student P+FT",   row.get("Student P+FT Total Params"),     row.get("Accuracy Pruned + Fine-Tuned (%)")),
    ]
    # keep only complete points
    clean = [(s, p, a) for (s, p, a) in points if pd.notna(p) and pd.notna(a)]
    return clean

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
            # nothing to plot for this run
            continue

        stages, params, accs = zip(*pts)

        fig, ax = plt.subplots(figsize=(8, 5))

        # connect points in their listed order
        ax.plot(params, accs, marker="o")  # default style/colors

        # add point labels
        for s, p, a in pts:
            ax.annotate(
                s,
                (p, a),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=9,
            )

        # axis labels & title
        ax.set_title(f"Run {run}: Accuracy vs. Parameters")
        ax.set_xlabel("Parameters")
        ax.set_ylabel("Accuracy (%)")

        # formatting
        if log_x:
            ax.set_xscale("log")
        else:
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=False))
            ax.ticklabel_format(style='plain', axis='x')

        ax.set_ylim(0, 100)
        ax.grid(True, linestyle="--", alpha=0.3)

        # optional helper text
        # show exact values under the title
        try:
            teacher_p   = next(p for s, p, _ in pts if s == "Teacher")
            student_b_p = next(p for s, p, _ in pts if s == "Student Before")
            student_p_p = next(p for s, p, _ in pts if s == "Student P+FT")
            # accuracy
            teacher_a   = next(a for s, _, a in pts if s == "Teacher")
            student_b_a = next(a for s, _, a in pts if s == "Student Before")
            student_p_a = next(a for s, _, a in pts if s == "Student P+FT")
            subtitle = (
                f"Teacher: {_fmt_int(teacher_p)} / {_fmt_pct(teacher_a)}    |    "
                f"Student Before: {_fmt_int(student_b_p)} / {_fmt_pct(student_b_a)}    |    "
                f"Student P+FT: {_fmt_int(student_p_p)} / {_fmt_pct(student_p_a)}"
            )
            ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", va="bottom", fontsize=9)
        except StopIteration:
            pass

        fig.tight_layout()
        out_path = out / f"Run_{run}/plots" /f"size_vs_accuracy.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)

    print(f"✅ Saved size-vs-accuracy plots to: {Path(out_dir).resolve()}")




################################################################################################################
############################ Tables for Teacher - Student Experiments    #######################################
################################################################################################################
def _get_best_val_accuracy_from_jsonl(metrics_path: Path, val_key: str = "val_accuracy"):
    """
    Scan a JSONL metrics file and return (best_val_accuracy, epoch).

    Assumes each line is a JSON object with keys:
      - val_key  (e.g. "val_accuracy")
      - "epoch"
    """
    best_val_acc = None
    best_epoch = None

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            val_acc = record.get(val_key)
            epoch = record.get("epoch")

            if val_acc is None or epoch is None:
                continue

            if (best_val_acc is None) or (val_acc > best_val_acc):
                best_val_acc = val_acc
                best_epoch = epoch

    if best_val_acc is None or best_epoch is None:
        raise ValueError(
            f"Could not find any valid (epoch, {val_key}) pairs in metrics file: {metrics_path}"
        )

    return best_val_acc, best_epoch

def _summarize_single_run(run_dir: Path):
    """
    Given a run directory like experiments/Experiment_3/Run_1, extract:
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
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    training_cfg = config.get("training", {})
    teacher_student_cfg = training_cfg.get("teacher_student", {})

    teacher_model = teacher_student_cfg.get("teacher_model_name","")
    student_model = config.get("model").get("name","")

    def _get_param(training_cfg, teacher_student_cfg, key, default=""):
        if key in training_cfg:
            return training_cfg[key]
        if key in teacher_student_cfg:
            return teacher_student_cfg[key]
        return default

    kd_type = _get_param(training_cfg, teacher_student_cfg, "kd_type")
    hidden_rep_loss_weight = _get_param(training_cfg, teacher_student_cfg, "hidden_rep_loss_weight")
    ce_loss_weight = _get_param(training_cfg, teacher_student_cfg, "ce_loss_weight")
    soft_target_loss_weight = _get_param(training_cfg, teacher_student_cfg, "soft_target_loss_weight")
    T_param = _get_param(training_cfg, teacher_student_cfg, "T_param")
    alpha_kd = _get_param(training_cfg, teacher_student_cfg, "alpha_kd")
    beta_feat = _get_param(training_cfg, teacher_student_cfg, "beta_feat")

    dataset = config.get("data").get("dataset_hf_repo_id","")

    # --- Teacher ---
    teacher_model_path = teacher_student_cfg.get("teacher_model_path","")
    if not teacher_model_path:
        raise KeyError(
            f"{config_path}: missing 'training.teacher_student.teacher_model_path'"
        )

    teacher_ckpt_path = Path(teacher_model_path)
    if not teacher_ckpt_path.is_absolute():
        # treat as relative to project root (cwd)
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

def _extract_run_number(name: str) -> int:
    """
    Extract integer suffix from folder names like:
      Run_1 → 1
      Run_12 → 12
    Returns 0 if not found.
    """
    import re
    m = re.search(r"(\d+)$", name)
    return int(m.group(1)) if m else 0

def summarize_experiment_group_to_markdown(
    experiment_dir: str,
    output_md_path: str,
) -> None:
    """
    Combine multiple runs inside an experiment directory into a single markdown table.

    Example:
        summarize_experiment_group_to_markdown(
            experiment_dir="experiments/Experiment_3",
            output_md_path="docs/results/experiment_3_summary.md",
        )

    Expected structure:
      experiments/Experiment_3/
        Run_1/
          metrics.jsonl
          config.toml
        Run_2/
          metrics.jsonl
          config.toml
        ...
    """
    experiment_path = Path(experiment_dir)
    output_path = Path(output_md_path)

    if not experiment_path.is_dir():
        raise NotADirectoryError(f"{experiment_dir} is not a directory")

    # --- numeric sort of Run_X folders ---
    run_dirs = sorted(
        (d for d in experiment_path.iterdir() if d.is_dir()),
        key=lambda d: _extract_run_number(d.name)
    )

    rows = []
    for run_dir in run_dirs:
        row = _summarize_single_run(run_dir)
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid runs found in {experiment_dir}")

    # --- Build markdown table ---
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

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"

    row_lines = []
    for row in rows:
        row_values = [str(row[h]) for h in headers]
        row_lines.append("| " + " | ".join(row_values) + " |")

    markdown_content = "\n".join([header_line, separator_line] + row_lines) + "\n"

    # --- Write output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_content, encoding="utf-8")



if __name__ == "__main__":
    main()