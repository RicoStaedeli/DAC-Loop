import logging
import os
import random
import tomllib
from pathlib import Path

import numpy as np
import torch
from pydantic import ValidationError

from framework.config_parser.schemasBase import AppConfig

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def load_config_toml(path: str | Path, validate: bool = True) -> AppConfig:
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    if not validate:
        # build model without validation
        return AppConfig.model_construct(**data)

    try:
        return AppConfig.model_validate(data)
    except ValidationError as e:
        raise SystemExit(f"Config validation failed:\n{e}") from e


def log_batch_preview(loader, tok, name: str = "train") -> None:
    """
    Log a preview of the first batch from a DataLoader.
    - batch shape
    - first 10 token IDs
    - decoded preview text (first 200 chars)

    Only runs if DEBUG logging is enabled.
    """
    if not log.isEnabledFor(logging.DEBUG):
        return

    try:

        b0 = next(iter(loader))

        ids = b0.get("input_ids") if isinstance(b0, dict) else None

        if ids is None:
            log.debug("[%s] No 'input_ids' key; keys=%s",
                      name, list(b0.keys()) if isinstance(b0, dict) else type(b0))
            return

        # shape
        shape = tuple(getattr(ids, "shape", (len(ids),)))
        log.debug("[%s] First batch shape=%s", name, shape)

        # first example token IDs
        try:
            first_ids = ids[0].tolist() if hasattr(ids[0], "tolist") else list(ids[0])
            log.info("first id")
            log.info(first_ids)
        except Exception:
            first_ids = list(ids[0]) if hasattr(ids, "__getitem__") else list(ids)[:32]
        log.debug("[%s] First example ids=%s", name, first_ids)

        # decode preview
        try:
            decoded = tok.decode(first_ids, skip_special_tokens=False)
            log.debug("[%s] Decoded preview=%r", name, decoded)
        except Exception as e:
            log.debug("[%s] Decode failed: %s", name, e)

    except StopIteration:
        log.debug("[%s] Loader is empty.", name)
    except Exception as e:
        log.debug("[%s] Preview failed: %s", name, e)
