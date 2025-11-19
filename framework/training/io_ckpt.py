import logging
from pathlib import Path
from typing import Dict

import torch

log = logging.getLogger(__name__)


class CheckpointIO:
    @staticmethod
    def load(model, optimizer, path: Path, device):
        ckpt = torch.load(str(path), map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        log.info("Loaded checkpoint from %s", path)
        return ckpt

    @staticmethod
    def save(state: Dict, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, str(path))
