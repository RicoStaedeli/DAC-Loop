import argparse
import logging
from pathlib import Path

from framework.utils.logger import setup_logging
from framework.utils.evaluation.plot_repository import (
    plot_teacher_student_metrics,
    make_overview_per_experiment,
    plot_compression_experiment,
    plot_dac_compression_curves,
    plot_val_accuracy_over_epochs
)
from framework.utils.utils import load_config_toml

log = logging.getLogger(__name__)


def setup_logging_from_config():
    """Initialize logging from experiments/configurations/plot_config.toml"""
    logger_cfg_path = "experiments/configurations/plot_config.toml"
    cfg = load_config_toml(logger_cfg_path, validate=False)
    setup_logging(cfg.model_dump(), Path(), False)
    log.info("Logging config loaded from %s", logger_cfg_path)


def _make_plots_for_experiment(
    experiment_name: str,
    num_runs: int,
    only_overview: bool = False,
    compression_experiment: bool = False,
):
    """
    Generate teacher-student or compression plots for all runs in the given
    experiment, and create an overview PDF combining loss and val-accuracy plots.
    """
    base_dir = Path(f"experiments/results/{experiment_name}")

    if not base_dir.exists():
        log.error("Base experiment directory does not exist: %s", base_dir)
        return

    if not only_overview:
        for i in range(1, num_runs + 1):
            cfg_path = base_dir / f"Run_{i}" / "config.toml"
            if cfg_path.exists():
                log.info("Config file exists: %s", cfg_path)
                if not compression_experiment:
                    # Standard teacher–student plots
                    plot_teacher_student_metrics(
                        cfg_path,
                        project_root=".",
                        export_png=True,
                        export_pdf=False,
                        png_dpi=600,
                        use_latex=True,
                    )
                else:
                    # Compression experiment plots
                    plot_compression_experiment(
                        base_dir=base_dir,
                        cfg_path=cfg_path,
                        export_png=True,
                        export_pdf=False,
                        png_dpi=600,
                        use_latex=True,
                    )
            else:
                log.warning("Config file does NOT exist: %s", cfg_path)

    # Overview over all runs
    make_overview_per_experiment(
        base_dir=str(base_dir),
        run_indices=range(1, num_runs + 1),
        out_path=str(base_dir / f"overview_plot_{experiment_name}.pdf"),
        compression=compression_experiment,
    )
    log.info("Overview plot written for %s", experiment_name)


def _run_single(
    experiment_name: str,
    run_id: int,
    plot_type: str,
):
    """
    Generate plots for a single run.

    plot_type ∈ {"teacher_student", "compression"}.
    """
    base_dir = Path(f"experiments/results/{experiment_name}")
    cfg_path = base_dir / f"Run_{run_id}" / "config.toml"

    if not cfg_path.exists():
        log.error("Config file does not exist: %s", cfg_path)
        return

    if plot_type == "teacher_student":
        plot_teacher_student_metrics(
            cfg_path,
            project_root=".",
            export_png=True,
            export_pdf=False,
            png_dpi=600,
            use_latex=True,
        )
    elif plot_type == "compression":
        plot_compression_experiment(
            base_dir=base_dir,
            cfg_path=cfg_path,
            export_png=True,
            export_pdf=False,
            png_dpi=600,
            use_latex=True,
        )
    else:
        log.error("Unsupported plot type for single run: %s", plot_type)
        return

    log.info(
        "Finished plots for %s / Run_%d (type=%s)",
        experiment_name,
        run_id,
        plot_type,
    )


def _run_all_predefined():
    """
    Old hard-coded behavior (optional).

    Call this if you still want the original “generate everything” mode
    when no CLI arguments are given.
    """
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
        "ResNet18": "experiments/results/Experiment_5/Run_1/metrics.jsonl",
        "ResNet10": "experiments/results/Experiment_5/Run_2/metrics.jsonl",
        "ResNet10small": "experiments/results/Experiment_5/Run_1/metrics.jsonl",
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
        "ResNet18-Student-Baseline": "experiments/results/Experiment_5/Run_1/metrics.jsonl",
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

    ########## Final Plot DAC Effect on RESISC45 ##########
    # 10 - Final Plot
    plot_dac_compression_curves(
        json_path="experiments/results/Experiment_13/result.json",
        out_path_base="docs/plots/10_dac_compression_curves",
        use_latex=True,
        total_parameters=11191917,
        figsize=(5, 3.2)
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate training/validation and compression plots."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        help="Experiment name, e.g. 'Experiment_7'",
    )
    parser.add_argument(
        "--run-id",
        type=int,
        default=0,
        help="Run id (e.g. 3). If 0 or omitted, all runs are processed.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        help="Number of runs in the experiment (required if --run-id=0).",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["teacher_student", "compression", "overview"],
        default="teacher_student",
        help=(
            "Type of plots to generate: "
            "'teacher_student' (default), 'compression', or 'overview' (only overview PDF)."
        ),
    )
    parser.add_argument(
        "--use-default-batch",
        action="store_true",
        help="Ignore other flags and run the original hard-coded batch of experiments.",
    )
    return parser.parse_args()


def main():
    setup_logging_from_config()
    args = parse_args()

    # Preserve old behavior if desired
    if args.use_default_batch or (args.experiment is None and args.run_id == 0):
        log.info("Running default batch plot generation...")
        _run_all_predefined()
        return

    if not args.experiment:
        raise SystemExit("--experiment is required unless --use-default-batch is set.")

    # Single-run mode
    if args.run_id and args.run_id > 0:
        log.info(
            "Generating plots for single run: experiment=%s, run_id=%d, type=%s",
            args.experiment,
            args.run_id,
            args.type,
        )
        _run_single(args.experiment, args.run_id, args.type)
        return

    # Multi-run (whole experiment) mode
    if args.num_runs is None:
        raise SystemExit(
            "--num-runs is required when --run-id is 0 (i.e., for all runs)."
        )

    only_overview = args.type == "overview"
    compression_experiment = args.type == "compression"

    log.info(
        "Generating plots for all runs: experiment=%s, num_runs=%d, type=%s "
        "(only_overview=%s, compression=%s)",
        args.experiment,
        args.num_runs,
        args.type,
        only_overview,
        compression_experiment,
    )
    _make_plots_for_experiment(
        experiment_name=args.experiment,
        num_runs=args.num_runs,
        only_overview=only_overview,
        compression_experiment=compression_experiment,
    )


if __name__ == "__main__":
    main()