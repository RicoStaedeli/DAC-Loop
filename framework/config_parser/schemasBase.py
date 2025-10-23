from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union, Annotated, List

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator, conint

from .schemasCNN import ResNet10Config, ResNet18Config, TinyCNNConfig

# TODO: add more model types here (instead of model registry)
ModelConfig = Annotated[
    Union[ResNet10Config, ResNet18Config, TinyCNNConfig],
    Field(discriminator="name")
]


class LogLevel(str, Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"


class LoggingConfig(BaseModel):
    default_level: LogLevel = LogLevel.INFO
    levels: Dict[str, LogLevel] = Field(default_factory=dict)


class ArtifactsConfig(BaseModel):
    hf_repo_id: str


class HFUploadConfig(BaseModel):
    enabled: bool = False

class ExperimentConfig(BaseModel):
    name: str
    run: PositiveInt

class SavingConfig(BaseModel):
    overwrite_run: bool = False

class TeacherStudentConfig(BaseModel):
    train_student: bool = False
    teacher_model_path: Optional[str] = None
    teacher_model_name: Optional[str] = None
    baseline_student: Optional[str] = None

class EarlyStoppingConfig(BaseModel):
    enabled: bool = False
    patience: int = 0
    min_delta: float = 0.0
    keep_last: int = 1

VALID_INDICES = [
    "NDVI", "NDWI", "NDMI", "EVI", "SAVI", "NDRE", "GNDVI",
    "BNDVI", "NBR", "MSI", "NDSI", "NDCI", "ARI", "BUI"
]

class IndicesConfig(BaseModel):
    calculated_indices: List[str]
    selected_bands: List[str]

    @field_validator("calculated_indices")
    @classmethod
    def validate_indices(cls, v):
        invalid = [idx for idx in v if idx not in VALID_INDICES]
        if invalid:
            raise ValueError(
                f"Invalid spectral index(es): {invalid}. "
                f"Valid options are: {', '.join(VALID_INDICES)}"
            )
        return v

class TrainingConfig(BaseModel):
    epochs: PositiveInt = 1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler: str = "cosine"  # "cosine", "step", "none"
    step_size: int = 10
    batch_size: PositiveInt = 1
    log_interval: PositiveInt = 10
    do_not_log_steps: bool = True
    seed: Optional[int] = 42
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    teacher_student: TeacherStudentConfig = TeacherStudentConfig()
    load_from_checkpoint: bool = False
    previous_checkpoint_path: Path = None
    kd_type: str = "CrossEntropyLoss"
    hidden_rep_loss_weight: float = 0.25
    ce_loss_weight: float = 0.75
    soft_target_loss_weight: float = 0.25
    T_param: PositiveInt = 1
    base_path_ckpts: str = ""

    @field_validator("learning_rate")
    @classmethod
    def lr_in_range(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError("learning_rate must be in (0, 1.0]")
        return v

    @field_validator("kd_type")
    @classmethod
    def valid_kd_type(cls, v: str) -> str:
        allowed = ["CrossEntropyLoss", "CosineEmbeddingLoss", "KLDivLoss"]
        if v not in allowed:
            raise ValueError(f"kd_type must be one of {allowed}")
        return v



class DataConfig(BaseModel):
    dataset_hf_repo_id: str
    num_used_entries_train:  conint(ge=0)
    num_used_entries_validation:  conint(ge=0)
    image_size: PositiveInt = 64
    augment: bool = False
    normalize: bool = True
    num_workers: int = 4
    pin_memory: bool = False
    indices: IndicesConfig = Field(default_factory=IndicesConfig)


# top level: validated config aggregated
class AppConfig(BaseModel):
    logging: LoggingConfig
    artifacts: ArtifactsConfig
    hf_upload: HFUploadConfig
    experiment: ExperimentConfig
    saving: SavingConfig
    training: TrainingConfig
    model: ModelConfig
    data: DataConfig
