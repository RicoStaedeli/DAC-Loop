# pruning/services/pruning_utils.py
from typing import Iterable, Tuple
import torch.nn as nn

def prunable_layers(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            yield name, module