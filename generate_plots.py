from framework.utils.plots_2 import plot_teacher_student_metrics
from framework.utils.logger import setup_logging
from framework.utils.utils import load_config_toml
from pathlib import Path


CFG_PATH = f'experiments/Experiment_3/Run_1/config.toml'
cfg = load_config_toml(CFG_PATH)
setup_logging(cfg.model_dump(), Path(), False)

for i in range(1,25):
    CFG_PATH = f'experiments/Experiment_3/Run_{i}/config.toml'
    plot_teacher_student_metrics(CFG_PATH)
