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

log = logging.getLogger(__name__)

def main():
    # Setup logging
    logger_cfg_path = "configs/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info(f"Config loaded from {logger_cfg_path}")


    experiment_dir = "experiments/Experiment_7"
    df = build_combined_table(experiment_dir, 20)
    # save_size_accuracy_per_run(df, experiment_dir, log_x=False)


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
    pct_col = "P+FT Params (%) of Teacher"
    int_cols     = [c for c in df.columns if "Params" in c and "(%)" not in c and "Size (MB)" not in c]
    float2_cols  = [c for c in df.columns if ("Accuracy" in c or "Sparsity" in c)]
    percent_cols = [
        "Accuracy Before Pruning (%)",
        "Sparsity (Pruned, Before Fine-Tune) (%)",
        "Accuracy After Pruning (Before Fine-Tune) (%)",
        "Sparsity After Pruning (Post Fine-Tune) (%)",
        "Accuracy Pruned + Fine-Tuned (%)",
        pct_col,  # the ratio column you already added
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
    df["P+FT Params (%) of Teacher"] = df["Student P+FT Total Params"] / df["Teacher Total Params"] * 100

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



if __name__ == "__main__":
    main()