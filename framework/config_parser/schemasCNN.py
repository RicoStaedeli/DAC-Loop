from typing import Literal

from pydantic import BaseModel


# Per-architecture args with defaults
class CNNArgs(BaseModel):
    classes: int = 10


# Per-architecture model configurations by 'model_version' name)
class ResNet10Config(BaseModel):
    name: Literal["ResNet10"]
    args: CNNArgs = CNNArgs()

class ResNet10miniConfig(BaseModel):
    name: Literal["ResNet10mini"]
    args: CNNArgs = CNNArgs()



class ResNet18Config(BaseModel):
    name: Literal["ResNet18"]
    args: CNNArgs = CNNArgs()


class ResNet50Config(BaseModel):
    name: Literal["ResNet50"]
    args: CNNArgs = CNNArgs()


class TinyCNNConfig(BaseModel):
    name: Literal["TinyCNN"]
    args: CNNArgs = CNNArgs()
