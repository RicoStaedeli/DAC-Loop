import json
from pathlib import Path
from typing import Tuple

from framework.utils.utils import load_config_toml

EXPERIMENTS_ROOT = Path("experiments")
DEFAULT_OUTPUT_PATH = Path("experiments/_generated_experiment_tables.md")


def _get_best_val_accuracy_from_jsonl(
        metrics_path: Path,
        val_key: str = "val_accuracy",
) -> Tuple[float, int]:
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


def _run_dir(experiment_name: str, run_idx: int) -> Path:
    return EXPERIMENTS_ROOT / experiment_name / f"Run_{run_idx}"


def _config_path(experiment_name: str, run_idx: int) -> Path:
    return _run_dir(experiment_name, run_idx) / "config.toml"


def _metrics_path(experiment_name: str, run_idx: int) -> Path:
    return _run_dir(experiment_name, run_idx) / "metrics.jsonl"


def _config_path_md(experiment_name: str, run_idx: int) -> str:
    return f"{experiment_name}/Run_{run_idx}/config.toml"


def _metrics_path_md(experiment_name: str, run_idx: int) -> str:
    return f"{experiment_name}/Run_{run_idx}/metrics.jsonl"


def _plots_path_md(experiment_name: str, run_idx: int) -> str:
    return f"{experiment_name}/Run_{run_idx}/plots"


# ---------- TABLE BUILDERS ----------


def _create_ex_desc_table_only_teacher(experiment_name: str, runs: int) -> str:
    """
    Create a markdown section (## title + table) for experiments without teacher-student setup.
    """
    lines: list[str] = []

    # Section title
    lines.append(f"## {experiment_name}")
    lines.append("")  # blank line before table (outside table is fine)

    # Table header (each item is one line)
    lines.append(
        "| Run | Config | Model | Dataset | Metrics | Plots | Bands | Best Accuracy |"
    )
    lines.append(
        "|-----|--------|-------|---------|---------|-------|-------|---------------|"
    )

    # Table body
    for run_idx in range(1, runs + 1):
        cfg_path = _config_path(experiment_name, run_idx)
        cfg = load_config_toml(str(cfg_path), validate=True)

        student_name = cfg.model.name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands


        metrics_path = _metrics_path(experiment_name, run_idx)
        print(f"Loading metrics from {metrics_path}")
        best_val_acc, best_epoch = _get_best_val_accuracy_from_jsonl(
            metrics_path,
            val_key="val_accuracy",
        )

        lines.append(
            f"| [{run_idx}] "
            f"| [Config.toml]({_config_path_md(experiment_name, run_idx)}) "
            f"| {student_name} "
            f"| `{dataset_name}` "
            f"| [metrics.json]({_metrics_path_md(experiment_name, run_idx)}) "
            f"| [plots]({_plots_path_md(experiment_name, run_idx)}) "
            f"| {indices} "
            f"| {best_val_acc:.2f}% @ {best_epoch} |"
        )

    lines.append("")  # blank line after table to separate from next section
    return "\n".join(lines)


def _create_ex_desc_table_teacher_student(experiment_name: str, runs: int) -> str:
    """
    Create a markdown section (## title + table) for teacher-student experiments.
    """
    lines: list[str] = []

    # Section title
    lines.append(f"## {experiment_name}")
    lines.append("")

    # Table header
    lines.append(
        "| Run | Config | Model | Model Teacher | Dataset | KD-Type "
        "| Metrics | Plots | Bands | Best Accuracy | Baseline Accuracy | Improvement to Baseline |"
    )
    lines.append(
        "|-----|--------|-------|---------------|---------|--------"
        "|---------|-------|-------|---------------|-------------------|-------------------------|"
    )

    # Table body
    for run_idx in range(1, runs + 1):
        cfg_path = _config_path(experiment_name, run_idx)
        cfg = load_config_toml(str(cfg_path), validate=True)

        student_name = cfg.model.name
        teacher_name = cfg.training.teacher_student.teacher_model_name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands

        ts_cfg = cfg.training.teacher_student
        kd_type = ts_cfg.kd_type

        result_metrics_path = _metrics_path(experiment_name, run_idx)
        baseline_metrics_path = Path(ts_cfg.baseline_student)

        # Result metrics
        if result_metrics_path.exists():
            try:
                best_val_acc, best_epoch = _get_best_val_accuracy_from_jsonl(
                    result_metrics_path,
                    val_key="val_accuracy",
                )
            except ValueError as e:
                print(f"[Warning] {e}")
                best_val_acc, best_epoch = 0.0, -1
        else:
            print(f"[Warning] Missing result metrics file: {result_metrics_path}")
            best_val_acc, best_epoch = 0.0, -1

        # Baseline metrics
        if baseline_metrics_path.exists():
            try:
                baseline_val_acc, _ = _get_best_val_accuracy_from_jsonl(
                    baseline_metrics_path,
                    val_key="val_accuracy",
                )
            except ValueError as e:
                print(f"[Warning] {e}")
                baseline_val_acc = 0.0
        else:
            print(f"[Warning] Missing baseline metrics file: {baseline_metrics_path}")
            baseline_val_acc = 0.0

        improvement = best_val_acc - baseline_val_acc

        lines.append(
            f"| [{run_idx}] "
            f"| [Config.toml]({_config_path_md(experiment_name, run_idx)}) "
            f"| {student_name} "
            f"| {teacher_name} "
            f"| `{dataset_name}` "
            f"| {kd_type} "
            f"| [metrics.json]({_metrics_path_md(experiment_name, run_idx)}) "
            f"| [plots]({_plots_path_md(experiment_name, run_idx)}) "
            f"| {indices} "
            f"| {best_val_acc:.2f}% @ {best_epoch} "
            f"| {baseline_val_acc:.2f}% "
            f"| {improvement:.2f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def _create_ex_desc_table_compression(experiment_name: str, runs: int) -> str:
    """
    Create a markdown section (## title + table) for teacher-student experiments.
    """
    lines: list[str] = []

    # Section title
    lines.append(f"## {experiment_name}")
    lines.append("")

    # Table header
    lines.append(
        "| Run | Config | Model | Model Teacher | Dataset | KD-Type  | Compression"
        "| Metrics | Plots | Bands | Best Accuracy | Baseline Accuracy | Improvement to Baseline |"
    )
    lines.append(
        "|-----|--------|-------|---------------|---------|--------|---------"
        "|---------|-------|-------|---------------|-------------------|-------------------------|"
    )

    # Table body
    for run_idx in range(1, runs + 1):
        cfg_path = _config_path(experiment_name, run_idx)
        cfg = load_config_toml(str(cfg_path), validate=True)
        student_name = cfg.model.name
        teacher_name = cfg.training.teacher_student.teacher_model_name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands
        pruning_method = cfg.training.pruning.schedule[0].method

        ts_cfg = cfg.training.teacher_student
        kd_type = ts_cfg.kd_type if ts_cfg.train_student else "N/A"

        result_metrics_path = _metrics_path(experiment_name, run_idx)
        baseline_metrics_path = Path(ts_cfg.baseline_student)

        # Result metrics
        if result_metrics_path.exists():
            try:
                best_val_acc, best_epoch = _get_best_val_accuracy_from_jsonl(
                    result_metrics_path,
                    val_key="val_accuracy",
                )
            except ValueError as e:
                print(f"[Warning] {e}")
                best_val_acc, best_epoch = 0.0, -1
        else:
            print(f"[Warning] Missing result metrics file: {result_metrics_path}")
            best_val_acc, best_epoch = 0.0, -1

        # Baseline metrics
        if baseline_metrics_path.exists():
            try:
                baseline_val_acc, _ = _get_best_val_accuracy_from_jsonl(
                    baseline_metrics_path,
                    val_key="val_accuracy",
                )
            except ValueError as e:
                print(f"[Warning] {e}")
                baseline_val_acc = 0.0
        else:
            print(f"[Warning] Missing baseline metrics file: {baseline_metrics_path}")
            baseline_val_acc = 0.0

        improvement = best_val_acc - baseline_val_acc

        lines.append(
            f"| [{run_idx}] "
            f"| [Config.toml]({_config_path_md(experiment_name, run_idx)}) "
            f"| {student_name} "
            f"| {teacher_name} "
            f"| `{dataset_name}` "
            f"| {kd_type} "
            f"| {pruning_method} "
            f"| [metrics.json]({_metrics_path_md(experiment_name, run_idx)}) "
            f"| [plots]({_plots_path_md(experiment_name, run_idx)}) "
            f"| {indices} "
            f"| {best_val_acc:.2f}% @ {best_epoch} "
            f"| {baseline_val_acc:.2f}% "
            f"| {improvement:.2f}% |"
        )

    lines.append("")
    return "\n".join(lines)


def main(output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    """
    Generate markdown tables for all configured experiments and
    write them to a single markdown file.
    """
    sections = [
        _create_ex_desc_table_only_teacher("Experiment_1", runs=12),
        _create_ex_desc_table_teacher_student("Experiment_2", runs=6),
        _create_ex_desc_table_teacher_student("Experiment_3", runs=24),
        _create_ex_desc_table_teacher_student("Experiment_4", runs=40),
        _create_ex_desc_table_teacher_student("Experiment_4a", runs=40),
        _create_ex_desc_table_only_teacher("Experiment_5", runs=8),
        _create_ex_desc_table_teacher_student("Experiment_6", runs=40),
        _create_ex_desc_table_teacher_student("Experiment_6a", runs=40),
        _create_ex_desc_table_compression("Experiment_7", runs=21),
        _create_ex_desc_table_compression("Experiment_7a", runs=21),
        _create_ex_desc_table_compression("Experiment_8", runs=12),
    ]

    markdown_content = "\n".join(sections)
    output_path.write_text(markdown_content, encoding="utf-8")

    print(f"Experiment tables written to: {output_path}")


if __name__ == "__main__":
    main()
