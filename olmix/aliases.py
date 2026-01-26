from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Union

import yaml
from olmo_core.data.types import NumpyDatasetDType
from pydantic import BaseModel

PathType = Union[Path, PathLike[Any], str]

# Constants for Chinchilla scaling
TOKENS_PER_PARAM = 20  # Chinchilla optimal tokens per parameter

# Approximate non-embedding parameter counts for each model
# These are used to compute max_tokens from chinchilla_multiple
MODEL_NUM_PARAMS: dict[str, int] = {
    "olmo2_1m": 1_000_000,
    "olmo2_30m": 30_000_000,
    "olmo2_60m": 60_000_000,
    "olmo2_190m": 190_000_000,
    "olmo2_1b": 1_000_000_000,
    "olmo2_7b": 7_000_000_000,
}


def compute_max_tokens(chinchilla_multiple: float, num_params: int) -> int:
    """Compute max training tokens from Chinchilla multiple and model parameters.

    Formula: TOKENS_PER_PARAM * num_params * chinchilla_multiple
    """
    return int(TOKENS_PER_PARAM * num_params * chinchilla_multiple)


def get_model_num_params(proxy_model_id: str) -> int:
    """Get the approximate number of non-embedding parameters for a model ID.

    Args:
        proxy_model_id: Model identifier (e.g., 'olmo2_30m')

    Returns:
        Approximate number of non-embedding parameters

    Raises:
        ValueError: If the model ID is not recognized
    """
    if proxy_model_id not in MODEL_NUM_PARAMS:
        raise ValueError(f"Unknown model: {proxy_model_id}. Available: {list(MODEL_NUM_PARAMS.keys())}")
    return MODEL_NUM_PARAMS[proxy_model_id]


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


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class InstanceFilterConfig(BaseModel):
    """Config for filtering repetitive sequences at the instance level."""

    repetition_min_period: int = 1
    repetition_max_period: int = 13
    repetition_max_count: int = 32


class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    budget: str
    workspace: str
    variants: int
    nodes: int
    gpus: int
    chinchilla_multiple: float = 1.0  # Trains for N * Chinchilla optimal tokens (20 * params * N)
    seed: int
    cluster: str
    tokenizer: str
    sources: list[SourceConfig]
    proxy_model_id: str
    priority: Priority = Priority.normal
    instance_filter: InstanceFilterConfig | None = None  # Optional quality filter for repetitive sequences
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

    def get_max_tokens(self) -> int:
        """Compute the maximum training tokens from chinchilla_multiple and model size.

        Returns:
            The total number of tokens to train for.
        """
        num_params = get_model_num_params(self.proxy_model_id)
        return compute_max_tokens(self.chinchilla_multiple, num_params)


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
