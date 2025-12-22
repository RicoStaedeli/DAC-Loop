# services/core.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _strip_module_prefix(sd: Mapping[str, torch.Tensor]) -> dict:
    """
    Strips the 'module.' prefix from state_dict keys if present.
    Args:
        sd (Mapping[str, torch.Tensor]): The state_dict to process.
    Returns:
        dict: The processed state_dict with 'module.' prefixes removed.
    """
    if sd and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return dict(sd)


def load_state_dict_from_any(ckpt_obj) -> dict:
    """
    Load state_dict from various checkpoint formats.
    Supports checkpoints with keys 'model_state', 'state_dict', or direct state_dict.
    Strips 'module.' prefix if present.
    Args:
        ckpt_obj: Loaded checkpoint object (dict).
    Returns:
        state_dict (dict): The model's state_dict.
    Raises:
        RuntimeError: If no valid state_dict is found in the checkpoint.
    """
    if isinstance(ckpt_obj, dict) and "model_state" in ckpt_obj:
        sd = ckpt_obj["model_state"]
    elif isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
        sd = ckpt_obj["state_dict"]
    elif isinstance(ckpt_obj, dict) and all(hasattr(v, "shape") for v in ckpt_obj.values()):
        sd = ckpt_obj
    else:
        raise RuntimeError("Checkpoint missing model state_dict.")
    return _strip_module_prefix(sd)


def maybe_load(model: nn.Module, path: Optional[str], device: torch.device, strict: bool = True) -> None:
    """
    Load model state_dict from a checkpoint file if path is provided.
    Args:
        model (nn.Module): The model to load the state_dict into.
        path (Optional[str]): Path to the checkpoint file. If None or empty, no loading is performed.
        device (torch.device): Device to map the loaded checkpoint.
        strict (bool): Whether to strictly enforce that the keys in state_dict match the model's keys.
    """
    if not path:
        return
    log.info("Loading state_dict from %s", path)
    ckpt = torch.load(path, map_location=device)
    sd = load_state_dict_from_any(ckpt)
    model.load_state_dict(sd, strict=strict)
    log.info("Loaded checkpoint: %s (strict=%s)", path, strict)


def save_state_dict(model: nn.Module, path: str | Path) -> None:
    """
    Save the model's state_dict to the specified path.
    Args:
        model (nn.Module): The model whose state_dict is to be saved.
        path (str | Path): The file path to save the state_dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    log.info("Saved model state_dict to %s", path)
