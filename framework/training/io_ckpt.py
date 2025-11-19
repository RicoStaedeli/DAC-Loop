import logging
from pathlib import Path
from typing import Dict

import torch

log = logging.getLogger(__name__)


class CheckpointIO:
    @staticmethod
    def load(model, optimizer, path: Path, device):
        """
        Loads the model and optimizer state from a checkpoint file.
        Args:
            model: The model to load the state into.
            optimizer: The optimizer to load the state into.
            path (Path): Path to the checkpoint file.
            device: The device to map the loaded checkpoint.
        Returns:
            The loaded checkpoint dictionary.
        """
        ckpt = torch.load(str(path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        log.info("Loaded checkpoint from %s", path)
        return ckpt

    @staticmethod
    def save(state: Dict, path: Path):
        """
        Saves the model and optimizer state to a checkpoint file.
        Args:
            state (Dict): The state dictionary containing model and optimizer states.
            path (Path): Path to save the checkpoint file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))
