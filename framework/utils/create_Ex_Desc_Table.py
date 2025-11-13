from pathlib import Path
from framework.utils.utils import load_config_toml



def _create_ex_desc_table(experiment_name, runs) -> str:
    """
    Create a markdown table summarizing experiments in the given directory.
    Each experiment is expected to be in a subdirectory named 'Experiment_X'.
    The table includes Experiment Name, Description, and Config Path.
    """
    # Create markdown table content for runs 11–24
    table_md = """| Run  | Date       | Author | Config                                           | Model Student | Model Teacher | KD-Type | Dataset                                   |  Train Time | Metrics                                            | Plots                                | Bands                                                        | Params                                | Notes/Issues | Next Steps |
|------|------------|--------|--------------------------------------------------|---------------|---------------|---------|-------------------------------------------|------------|----------------------------------------------------|--------------------------------------|--------------------------------------------------------------|------------------------------------------|--------------|------------|\n"""
    for i in range(1, 1+runs):
        logger_cfg_path = f"experiments/{experiment_name}/Run_{i}/config.toml"
        cfg = load_config_toml(logger_cfg_path, validate=True)
        name_student = cfg.model.name
        name_teacher = cfg.training.teacher_student.teacher_model_name
        dataset_name = cfg.data.dataset_hf_repo_id
        indices = cfg.data.indices.calculated_indices + cfg.data.indices.selected_bands
        teacher_student_cfg = cfg.training.teacher_student
        kd_type = teacher_student_cfg.kd_type

        table_md += f'| [{i}]  | 2025-10-25 | Rico   |  [Config.toml]({experiment_name}/Run_{i}/config.toml)   |    {name_student}    | {name_teacher}      | `{dataset_name}`  | {kd_type} |            | [metrics.json]({experiment_name}/Run_{i}/metrics.jsonl) | [plots]({experiment_name}/Run_{i}/plots)  | {indices}  |     hidden_rep_loss_weight = {teacher_student_cfg.hidden_rep_loss_weight} <br>    ce_loss_weight = {teacher_student_cfg.ce_loss_weight} <br>  soft_target_loss_weight = {teacher_student_cfg.soft_target_loss_weight} <br>  alpha_kd = {teacher_student_cfg.alpha_kd} <br>  beta_feat = {teacher_student_cfg.beta_feat} <br>  T_param = {teacher_student_cfg.T_param}      |            |\n'

    return table_md



def main():
    print(_create_ex_desc_table(experiment_name = "Experiment_6",runs=40))

if __name__ == "__main__":
    main()