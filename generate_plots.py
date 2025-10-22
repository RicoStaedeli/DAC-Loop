from framework.utils.plots_2 import plot_teacher_student_metrics
from framework.utils.logger import setup_logging, log_meta_info
from framework.utils.utils import set_seed, load_config_toml
from pathlib import Path

CFG_PATH = f'experiments/Experiment_XY/Run_1/config.toml'

cfg = load_config_toml(CFG_PATH)
setup_logging(cfg.model_dump(), Path(), False)

plot_teacher_student_metrics(CFG_PATH)

# for i in range(1,9):
#     CFG_PATH = f'../experiments/Experiment_1/Run_{i}/config.toml'
#     plot_teacher_student_metrics(CFG_PATH)