# pruning/services/pruning_utils.py
from typing import Iterable, Tuple

import torch.nn as nn


def prunable_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    """
    Generator that yields the names and modules of prunable layers in the model.
    Prunable layers are defined as instances of nn.Conv2d and nn.Linear.
    Args:
        model (nn.Module): The model to inspect.
    Yields:
        Iterable[Tuple[str, nn.Module]]: Tuples of (layer name, layer module
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module
