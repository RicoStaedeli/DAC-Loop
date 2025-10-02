import logging
import os
import time
import traceback
from pathlib import Path
import shutil

# Framework Imports
from framework.utils.logger import setup_logging, log_meta_info, log_config_summary
from framework.utils.utils import set_seed, log_batch_preview, load_config_toml

log = logging.getLogger(__name__)

# TODO: set config paths here to run experiments
CFG_PATHS = [
    Path("configs/config.toml")
    # Path("configs/config_1.toml"),
    # Path("configs/config_2.toml"),
]

def main():
    for CFG_PATH in CFG_PATHS:
        t0 = time.time()
        # validate .toml
        cfg = load_config_toml(CFG_PATH)
        # get run_id & seed, check metadata
        experiment_dir = _create_log_directory(cfg)
        setup_logging(cfg.model_dump(), experiment_dir)


        seed = cfg.training.seed

        if seed is not None:
            set_seed(seed)

        log_meta_info(cfg.model_dump(),experiment_dir, seed)
        exp_name = cfg.experiment.name

        try:
            log.info(f"Experiment name: {exp_name} --> start training")
        except Exception as e:
            log.error(f"Experiment name: {exp_name} --> error: {e}")
        finally:
            log.info(f"Experiment name: {exp_name} --> end training")
            log.info("\n" + "=" * 88)
            log.info("RUN END    |  Experiment=%s  |  Run ID=%s  |  Duration=%.1fs",
                     exp_name, cfg.experiment.run, time.time() - t0)
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
