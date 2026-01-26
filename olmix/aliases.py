from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Union

import yaml
from olmo_core.data.types import NumpyDatasetDType
from pydantic import BaseModel

PathType = Union[Path, PathLike[Any], str]


class Priority(str, Enum):
    """Beaker job priority levels."""

    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class TrainType(Enum):
    pretrain = "pretrain"
    anneal = "anneal"


class TopicConfig(BaseModel):
    name: str
    paths: list[str]
    max_repetition_factor: float = 1.0
    max_topic_ratio: float = 1.0
    weight: float | None = None


class SourceConfig(BaseModel):
    name: str
    paths: list[str] | None = None
    topics: list[TopicConfig] | None = None
    max_repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    budget: str
    workspace: str
    variants: int
    nodes: int
    gpus: int
    max_tokens: int
    sequence_length: int
    seed: int
    cluster: str
    tokenizer: str
    sources: list[SourceConfig]
    proxy_model_id: str
    priority: Priority = Priority.normal
    minimum_weight: float | None = None
    minimum_source_weight: float | None = None
    minimum_topic_weight: float | None = None
    checkpoint_path: str | None = None
    train_type: TrainType = TrainType.pretrain
    allow_repetition: bool = True
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32
    mix_temperature: float = 1.0
    source_mix_temperature: float | None = None
    topic_mix_temperature: float | None = None
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_source_strength: float | None = None
    max_source_strength: float | None = None
    min_topic_strength: float | None = None
    max_topic_strength: float | None = None
    nonzero_weight: list[str] | None = None
    fixed_source_weights: dict[str, float] | None = None
    device_batch_size: int = 4
    global_batch_size: int | None = None
    manual_prior: dict[str, float] | None = None
    manual_topic_prior: dict[str, float] | None = None
    sample_multiplier: int | None = None
    wandb_debug: bool = False
    existing_mix_file: str | None = None

    @classmethod
    def from_yaml(cls, path: PathType) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ExperimentInstance(BaseModel):
    """A single experiment instance with its mixture configuration."""

    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    """A group of experiment instances sharing a common configuration."""

    config: ExperimentConfig
    group_id: str
    instances: list[ExperimentInstance]


def config_from_path(config: PathType) -> ExperimentConfig:
    """Load an ExperimentConfig from a YAML file path."""
    return ExperimentConfig.from_yaml(config)
