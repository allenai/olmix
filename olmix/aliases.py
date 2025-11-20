from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Optional, Union

from olmo_core.data.types import NumpyDatasetDType
from pydantic import BaseModel
import pydantic
from typing import List, Optional, Union

PathType = Union[Path, PathLike[Any], str]


class TrainType(Enum):
    pretrain = "pretrain"
    anneal = "anneal"


class TopicConfig(BaseModel):
    name: str
    paths: List[str]
    max_repetition_factor: float = 1.0
    max_topic_ratio: float = 1.0
    weight: Optional[float] = None


class SourceConfig(BaseModel):
    name: str
    paths: Optional[List[str]] = None
    topics: Optional[List[TopicConfig]] = None
    max_repetition_factor: float = 1.0
    max_source_ratio: float = 1.0


class SourceInstance(BaseModel):
    name: str
    paths: list[str]
    ratio: float
    repetition_factor: float = 1.0


class ExperimentConfig(BaseModel):
    name: str
    description: str
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
    tokenizer: str
    proxy_model_id: str
    minimum_weight: Optional[float] = None
    minimum_source_weight: Optional[float] = None
    minimum_topic_weight: Optional[float] = None
    checkpoint_path: Optional[str] = None
    train_type: TrainType = TrainType.pretrain
    allow_repetition: bool = True
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32
    mix_temperature: float = 1.0
    source_mix_temperature: Optional[float] = None
    topic_mix_temperature: Optional[float] = None
    preemptible: bool = True
    shared_filesystem: bool = False
    weka: bool = False
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_source_strength: Optional[float] = None
    max_source_strength: Optional[float] = None
    min_topic_strength: Optional[float] = None
    max_topic_strength: Optional[float] = None
    nonzero_weight: Optional[list[str]] = None
    fixed_source_weights: Optional[dict[str, float]] = None
    device_batch_size: int = 4
    global_batch_size: Optional[int] = None
    manual_prior: Optional[dict[str, float]] = None
    sample_multiplier: Optional[int] = None
    wandb_debug: bool = False
    existing_mix_file: Optional[str] = None 
    # TODO(undfined): Add field validation for weka/cluster/train_type here
