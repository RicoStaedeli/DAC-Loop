from framework.utils.plots_2 import plot_teacher_student_metrics

for i in range(1,7):
    CFG_PATH = f'../experiments/Experiment_2/Run_{i}/config.toml'
    plot_teacher_student_metrics(CFG_PATH)