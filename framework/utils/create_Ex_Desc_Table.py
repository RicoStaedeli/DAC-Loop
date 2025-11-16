import json
from pathlib import Path

from framework.utils.utils import load_config_toml


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


def _create_ex_desc_table_only_teacher(experiment_name, runs) -> str:
    """
    Create a markdown table summarizing experiments in the given directory.
    Each experiment is expected to be in a subdirectory named 'Experiment_X'.
    The table includes Experiment Name, Description, and Config Path.
    """
    # Create markdown table content for runs 11–24
    table_md = """| Run   | Config                                           | Model | Dataset |   Metrics                                            | Plots                                | Bands                                                        | Best Accuracy | 
|------|--------------------------------------------------|---------------|---------------|---------------------------------------------------|--------------------------------------|--------------------------------------------------------------|------------------------------------------|\n"""
    for i in range(1, 1 + runs):
        logger_cfg_path = f"experiments/{experiment_name}/Run_{i}/config.toml"
        cfg = load_config_toml(logger_cfg_path, validate=True)
        name_student = cfg.model.name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands

        result_metrics_path = Path(f"experiments/{experiment_name}/Run_{i}/metrics.jsonl")

        best_val_acc, best_epoch = _get_best_val_accuracy_from_jsonl(result_metrics_path, val_key="val_accuracy")

        table_md += f'| [{i}]   |  [Config.toml]({experiment_name}/Run_{i}/config.toml)   |    {name_student}    | `{dataset_name}`  |   [metrics.json]({experiment_name}/Run_{i}/metrics.jsonl) | [plots]({experiment_name}/Run_{i}/plots)  | {indices}  |    {best_val_acc:.2f}% @ {best_epoch}           |\n'

    return table_md


def _create_ex_desc_table_teacher_student(experiment_name, runs) -> str:
    """
    Create a markdown table summarizing experiments in the given directory.
    Each experiment is expected to be in a subdirectory named 'Experiment_X'.
    The table includes Experiment Name, Description, and Config Path.
    """
    # Create markdown table content for runs 11–24
    table_md = """| Run   | Config                                           | Model | Model Teacher | Dataset |  KD-Type                                   | Metrics                                            | Plots                                | Bands                                                        | Best Accuracy | Baseline Accuracy | Improvement to Baseline |
|------|--------------------------------------------------|---------------|---------------|---------|-------------------------------------------|---------------------------------------------------|--------------------------------------|--------------------------------------------------------------|------------------------------------------|------|--------|\n"""
    for i in range(1, 1 + runs):
        logger_cfg_path = f"experiments/{experiment_name}/Run_{i}/config.toml"
        cfg = load_config_toml(logger_cfg_path, validate=True)
        name_student = cfg.model.name
        name_teacher = cfg.training.teacher_student.teacher_model_name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands
        teacher_student_cfg = cfg.training.teacher_student
        kd_type = teacher_student_cfg.kd_type

        result_metrics_path = Path(f"experiments/{experiment_name}/Run_{i}/metrics.jsonl")
        metrics_baseline_path = Path(teacher_student_cfg.baseline_student)


        # Check result metrics file
        if result_metrics_path.exists():
            best_val_acc, best_epoch = _get_best_val_accuracy_from_jsonl(
                result_metrics_path, val_key="val_accuracy"
            )
        else:
            print(f"[Warning] Missing result metrics file: {result_metrics_path}")
            best_val_acc, best_epoch = 0.0, -1  # or handle differently

        # Check baseline metrics file
        if metrics_baseline_path.exists():
            baseline_val_acc, _ = _get_best_val_accuracy_from_jsonl(
                metrics_baseline_path, val_key="val_accuracy"
            )
        else:
            print(f"[Warning] Missing baseline metrics file: {metrics_baseline_path}")
            baseline_val_acc = 0.0  # or whatever default makes sense

        # Compute improvement only if both are valid
        improvement = best_val_acc - baseline_val_acc

        table_md += f'| [{i}]   |  [Config.toml]({experiment_name}/Run_{i}/config.toml)   |    {name_student}    | {name_teacher}      | `{dataset_name}`  | {kd_type} |   [metrics.json]({experiment_name}/Run_{i}/metrics.jsonl) | [plots]({experiment_name}/Run_{i}/plots)  | {indices}  |    {best_val_acc:.2f}% @ {best_epoch}           |   {baseline_val_acc:.2f}%  |  {improvement:.2f}%  |\n'

    return table_md


def main():
    # print(_create_ex_desc_table_only_teacher(experiment_name = "Experiment_1",runs=12))
    # print(_create_ex_desc_table_only_teacher(experiment_name = "Experiment_5",runs=8))

    # print(_create_ex_desc_table_teacher_student(experiment_name = "Experiment_2",runs=6))
    # print(_create_ex_desc_table_teacher_student(experiment_name = "Experiment_3",runs=24))
    # print(_create_ex_desc_table_teacher_student(experiment_name = "Experiment_4",runs=40))
    # print(_create_ex_desc_table_teacher_student(experiment_name = "Experiment_4a",runs=40))
    # print(_create_ex_desc_table_teacher_student(experiment_name="Experiment_6", runs=40))
    print(_create_ex_desc_table_teacher_student(experiment_name="Experiment_6_a", runs=40))


if __name__ == "__main__":
    main()
