from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
import logging
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Optional
import math

log = logging.getLogger(__name__)

def main():
    df = build_combined_table()


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
def build_combined_table():
    combined_rows = []

    for i in range(1,16):
        row: Dict[str, Optional[float]] = {"Run": i}

        # Paths as given in your snippet
        base = f"experiments/Experiment_7/Run_{i}"

        # a) Relative metrics
        metrics_path = f"{base}/Experiment_7_{i}.log"
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

# ---------- Run it ----------
# df = build_combined_table()

if __name__ == "__main__":
    main()