import logging
from pathlib import Path

from framework.utils.logger import setup_logging
from framework.utils.plots.plot_repository import plot_teacher_student_metrics, make_overview_per_experiment, \
    plot_val_accuracy_over_epochs, plot_compression_experiment
from framework.utils.utils import load_config_toml

log = logging.getLogger(__name__)


def main():
    # Setup logging
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info(f"Config loaded from {logger_cfg_path}")

    ################################################################################################################
    ############################ Plots for Experiment             ##################################################
    ################################################################################################################

    # Generate plots for experiments
    # _make_plots_for_experiment(experiment_name="Experiment_1", num_runs=12, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_2", num_runs=6, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_3", num_runs=24, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_4", num_runs=40, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_4a", num_runs=40, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_5", num_runs=9, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_6", num_runs=40, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_6a", num_runs=40, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_6b", num_runs=40, only_overview=False, compression_experiment=False)
    # _make_plots_for_experiment(experiment_name="Experiment_7", num_runs=21, only_overview=False, compression_experiment=True)
    # _make_plots_for_experiment(experiment_name="Experiment_7a", num_runs=21, only_overview=False, compression_experiment=True)
    # _make_plots_for_experiment(experiment_name="Experiment_8", num_runs=12, only_overview=False, compression_experiment=True)
    _make_plots_for_experiment(experiment_name="Experiment_9", num_runs=4, only_overview=False, compression_experiment=True)

    ################################################################################################################
    ############################ Plots with different experiments ##################################################
    ################################################################################################################

    _plots_for_different_experiments()


def _plots_for_different_experiments():
    ########## Plot Validation Accuracy over Epochs for different models on EuroSAT ##########
    # 1 - EuroSAT Baseline training runs
    models = {
        "ResNet50": "experiments/results/Experiment_1/Run_10/metrics.jsonl",
        "ResNet18": "experiments/results/Experiment_1/Run_1/metrics.jsonl",
        "ResNet10": "experiments/results/Experiment_1/Run_4/metrics.jsonl",
        "TinyCNN": "experiments/results/Experiment_1/Run_7/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/1_baseline_eurosat_all_bands",
        title="Validation Accuracy over Epochs EuroSAT - All Bands",
        use_latex=True,
        figsize=(6, 4)
    )

    # 2 - EuroSAT RGB and Special Bands training runs
    models = {
        "ResNet50": "experiments/results/Experiment_1/Run_11/metrics.jsonl",
        "ResNet18": "experiments/results/Experiment_1/Run_2/metrics.jsonl",
        "ResNet10": "experiments/results/Experiment_1/Run_5/metrics.jsonl",
        "TinyCNN": "experiments/results/Experiment_1/Run_8/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/2_baseline_eurosat_rgb",
        title="Validation Accuracy over Epochs EuroSAT - RGB Bands",
        use_latex=True,
        figsize=(6, 4)
    )

    # 3 - EuroSAT Special Bands training runs
    models = {
        "ResNet50": "experiments/results/Experiment_1/Run_12/metrics.jsonl",
        "ResNet18": "experiments/results/Experiment_1/Run_3/metrics.jsonl",
        "ResNet10": "experiments/results/Experiment_1/Run_6/metrics.jsonl",
        "TinyCNN": "experiments/results/Experiment_1/Run_9/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/3_baseline_eurosat_special_bands",
        title="Validation Accuracy over Epochs EuroSAT - Special Bands",
        use_latex=True,
        figsize=(6, 4)
    )

    ########## Plot Validation Accuracy over Epochs for different models on RESISC45 ##########
    # 4 - RESISC45 Baseline training runs
    models = {
        "ResNet50": "experiments/results/Experiment_5/Run_7/metrics.jsonl",
        "ResNet18": "experiments/results/Experiment_5/Run_8/metrics.jsonl",
        "ResNet10": "experiments/results/Experiment_5/Run_2/metrics.jsonl",
        "ResNet10small": "experiments/results/Experiment_5/Run_9/metrics.jsonl",
        "TinyCNN": "experiments/results/Experiment_5/Run_3/metrics.jsonl",

    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/4_baseline_resisc45",
        title="Validation Accuracy over Epochs Resisc45",
        use_latex=True,
        figsize=(6, 4)
    )

    ########## Plot Validation Accuracy over Epochs for Student vs Baseline on EuroSAT ##########
    # 5 - EuroSAT TinyCNN Student vs Baseline training runs
    models = {
        "TinyCNN-Student": "experiments/results/Experiment_3/Run_1/metrics.jsonl",
        "TinyCNN-Student-Baseline": "experiments/results/Experiment_1/Run_7/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/5_eurosat_tinycnn_student_vs_baseline",
        title="TinyCNN - Validation Accuracy over Epochs on EuroSAT",
        use_latex=True,
        figsize=(6, 3)
    )

    # 6 - EuroSAT ResNet18 Student vs Baseline training runs
    models = {
        "ResNet18-Student": "experiments/results/Experiment_4/Run_3/metrics.jsonl",
        "ResNet18-Student-Baseline": "experiments/results/Experiment_1/Run_1/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/6_eurosat_resnet18_student_vs_baseline",
        title="ResNet18 - Validation Accuracy over Epochs on EuroSAT",
        use_latex=True,
        figsize=(6, 3)
    )

    # 7 - EuroSAT ResNet10 Student vs Baseline training runs
    models = {
        "ResNet10-Student": "experiments/results/Experiment_4a/Run_6/metrics.jsonl",
        "ResNet10-Student-Baseline": "experiments/results/Experiment_1/Run_4/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/7_eurosat_resnet10_student_vs_baseline",
        title="ResNet10 - Validation Accuracy over Epochs on EuroSAT",
        use_latex=True,
        figsize=(6, 3)
    )

    ########## Plot Validation Accuracy over Epochs for Student vs Baseline on RESISC45 ##########
    # 8 - RESISC45 ResNet18 Student vs Baseline training runs
    models = {
        "ResNet18-Student": "experiments/results/Experiment_6/Run_23/metrics.jsonl",
        "ResNet18-Student-Baseline": "experiments/results/Experiment_5/Run_8/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/8_resisc_resnet18_student_vs_baseline",
        title="ResNet18 - Validation Accuracy over Epochs on RESISC45",
        use_latex=True,
        figsize=(6, 3)
    )

    # 9 - RESISC45 ResNet10 Student vs Baseline training runs
    models = {
        "ResNet10-Student": "experiments/results/Experiment_6a/Run_40/metrics.jsonl",
        "ResNet10-Student-Baseline": "experiments/results/Experiment_5/Run_2/metrics.jsonl",
    }
    plot_val_accuracy_over_epochs(
        models,
        out_path_base="docs/plots/9_resisc_resnet10_student_vs_baseline",
        title="ResNet10 - Validation Accuracy over Epochs on RESISC45",
        use_latex=True,
        figsize=(6, 3)
    )


def _make_plots_for_experiment(experiment_name: str, num_runs: int, only_overview: bool = False, compression_experiment: bool = False):
    """
    Generate teacher-student plots for all runs in the specified experiment,
    and create an overview PNG combining loss and validation accuracy plots.
    """
    base_dir = Path(f"experiments/results/{experiment_name}")

    if not only_overview:
        for i in range(1, num_runs + 1):
            cfg_path = base_dir / f"Run_{i}" / "config.toml"
            if cfg_path.exists():
                log.info("Config file exists: %s", cfg_path)
                if not compression_experiment:
                    plot_teacher_student_metrics(cfg_path, project_root=".", export_png=True, export_pdf=False, png_dpi=600,
                                                 use_latex=True)
                else:
                    plot_compression_experiment(base_dir=base_dir / f"Run_{i}", cfg_path=cfg_path, export_png=True, export_pdf=False, png_dpi=600,
                                             use_latex=True)
            else:
                log.info("Config file does not exist:", cfg_path)

    make_overview_per_experiment(
        base_dir=str(base_dir),
        run_indices=range(1, num_runs + 1),
        out_path=str(base_dir / f"overview_plot_{experiment_name}.pdf"),
        compression=compression_experiment,
    )


if __name__ == "__main__":
    main()
