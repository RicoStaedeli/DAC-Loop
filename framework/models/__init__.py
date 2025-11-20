# models/__init__.py

from .ResNet10 import ResNet10
from .ResNet10mini import ResNet10mini
from .ResNet18 import ResNet18
from .ResNet50 import ResNet50
from .TinyCNN import TinyCNN

MODEL_REGISTRY = {
    "ResNet10": ResNet10,
    "ResNet10mini": ResNet10mini,
    "ResNet18": ResNet18,
    "TinyCNN": TinyCNN,
    "ResNet50": ResNet50,
}
