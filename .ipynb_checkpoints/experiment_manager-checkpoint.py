import logging
import os
import time
from pathlib import Path
import shutil
from dotenv import load_dotenv
import torch

# Framework Imports
from framework.utils.logger import setup_logging, log_meta_info
from framework.utils.utils import set_seed, load_config_toml
from framework.dataset.load_for_training import load_eurosat_hf_splits, build_loaders_from_hf
from framework.models.ResNet18 import ResNet18
from framework.training.trainer import train

log = logging.getLogger(__name__)
load_dotenv()

# TODO: set config paths here to run experiments
CFG_PATHS = [
    Path("configs/Experiment-1-Run-1.toml"),
    Path("configs/Experiment-1-Run-2.toml")
    # Path("configs/config_1.toml"),
    # Path("configs/config_2.toml"),
]

def main():
    for CFG_PATH in CFG_PATHS:
        t0 = time.time()
        # validate .toml
        cfg = load_config_toml(CFG_PATH)

        # Set all possible seeds for reproduction
        seed = cfg.training.seed
        if seed is not None:
            set_seed(seed)

        # get run_id & seed, check metadata
        experiment_dir = _create_log_directory(cfg)
        experiment_name = cfg.experiment.name

        # Logging setup
        setup_logging(cfg.model_dump(), experiment_dir)
        log_meta_info(cfg.model_dump(), experiment_dir, seed)

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}")

        # Start Training
        try:
            log.info(f"Starting experiment {experiment_name}")
            log.info(f"Load Dataset")
            train_ds, val_ds, test_ds = load_eurosat_hf_splits(cfg.data.dataset_hf_repo_id)
            log.info(f"Build loaders")
            train_loader, val_loader, norm_mean, norm_std = build_loaders_from_hf(train_ds, val_ds, cfg.training.batch_size,
                                                             cfg.data.image_size, cfg.data.indices.calculated_indices, cfg.data.augment, normalize=cfg.data.normalize,stats_samples=0, selected_bands=cfg.data.indices.selected_bands)

            extra_channels = len(cfg.data.indices.calculated_indices) if cfg.data.indices.calculated_indices else 0
            input_channels = len(cfg.data.indices.selected_bands) if cfg.data.indices.selected_bands else 0
            input_channels = input_channels + extra_channels
            log.info(f"Build model {cfg.model.name} with {input_channels} input channels and {cfg.model.args.classes} extra output classes")
            model = ResNet18(input_channels=input_channels, num_classes=cfg.model.args.classes)

            train(cfg, model, device, train_loader, val_loader,experiment_dir, CFG_PATH)

        except Exception as e:
            log.error(f"Experiment name: {experiment_name} --> error: {e}")
        finally:
            log.info(f"Experiment name: {experiment_name} --> end training")
            log.info("\n" + "=" * 88)
            log.info("RUN END    |  Experiment=%s  |  Run ID=%s  |  Duration=%.1fs",
                     experiment_name, cfg.experiment.run, time.time() - t0)
            log. info("=" * 88)


def _create_log_directory(cfg) -> Path:
    experiment_log_dir = Path(f"Experiments/{cfg.experiment.name}/Run_{cfg.experiment.run}")
    if os.path.exists(experiment_log_dir) and cfg.saving.overwrite_run == False:
        raise ValueError("There is already a directory named '%s'. Check if you updated the run ID to a new one. "
            "If you want to overwrite the current existing directory please set the parameter in the config file.",
            experiment_log_dir)

    elif os.path.exists(experiment_log_dir) and cfg.saving.overwrite_run == True:
        log.warning("There is already a directory named '%s'. But you choose to overwrite it. --> creating empty directory",experiment_log_dir)
        shutil.rmtree(experiment_log_dir)

    else:
        log.info(f"Creating directory '{experiment_log_dir}'")

    experiment_log_dir.mkdir(parents=True, exist_ok=True)
    return experiment_log_dir


if __name__ == "__main__":
    main()
