from pathlib import Path
from framework.utils.utils import load_config_toml



def _create_ex_desc_table(experiment_name, runs) -> str:
    """
    Create a markdown table summarizing experiments in the given directory.
    Each experiment is expected to be in a subdirectory named 'Experiment_X'.
    The table includes Experiment Name, Description, and Config Path.
    """
    # Create markdown table content for runs 11–24
    table_md = """| Run  | Date       | Author | Config                                           | Model Student | Model Teacher | KD-Type | Dataset                                   | Hardware   | Train Time | Metrics                                            | Plots                                | Bands                                                        | Notes/Issues | Next Steps |
    |--------------------|------------|--------|--------------------------------------------------|---------------|---------------|---------|-------------------------------------------|------------|------------|----------------------------------------------------|--------------------------------------|--------------------------------------------------------------|--------------|------------|
    """
    for i in range(1, 1+runs):
        logger_cfg_path = f"experiments/{experiment_name}/Run_{i}/config.toml"
        cfg = load_config_toml(logger_cfg_path, validate=True)
        name_student = cfg.model.name
        name_teacher = cfg.training.teacher_student.teacher_model_name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands
        kd_type = cfg.training.kd_type

        table_md += f'| [{i}] | 2025-10-25 | Rico   | [Config.toml]({experiment_name}/Run_{i}/config.toml)  |    {name_student}    | {name_teacher}      | Train/Val/Test split of {dataset_name} | {kd_type} | DGX-7 Rico |            | [metrics.json]({experiment_name}/Run_{i}/metrics.jsonl) | [plots]({experiment_name}/Run_{i}/plots)  | {indices}  |              |            |\n'

    return table_md



def main():
    print(_create_ex_desc_table(experiment_name = "Experiment_5",runs=5))

if __name__ == "__main__":
    main()