# services/core.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def _strip_module_prefix(sd: Mapping[str, torch.Tensor]) -> dict:
    if sd and next(iter(sd)).startswith("module."):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return dict(sd)


def load_state_dict_from_any(ckpt_obj) -> dict:
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
    if not path:
        return
    ckpt = torch.load(path, map_location=device)
    sd = load_state_dict_from_any(ckpt)
    model.load_state_dict(sd, strict=strict)
    log.info("Loaded checkpoint: %s (strict=%s)", path, strict)


def save_state_dict(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(path))
    log.info("Saved model state_dict to %s", path)
