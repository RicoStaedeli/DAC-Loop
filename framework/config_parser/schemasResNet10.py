import warnings
from typing import Literal, Optional
from pydantic import BaseModel, PositiveInt, model_validator, NonNegativeInt


# Per-architecture args with defaults
class ResNet10Args(BaseModel):
    classes: int = 10


# Per-architecture model configs by 'model_version' name)
class ResNet10Config(BaseModel):
    name: Literal["ResNet10"]
    args: ResNet10Args = ResNet10Args()
