from enum import Enum, StrEnum
from os import PathLike
from pathlib import Path
from typing import Any, Union

import yaml
from olmo_core.data.types import NumpyDatasetDType
from pydantic import BaseModel, model_validator

from olmix.fit.config import InLoopEvalConfig, PriorsConfig

PathType = Union[Path, PathLike[Any], str]

# Constants for Chinchilla scaling
TOKENS_PER_PARAM = 20  # Chinchilla optimal tokens per parameter

# Approximate non-embedding parameter counts for each model
# These are used to compute max_tokens from chinchilla_multiple
MODEL_NUM_PARAMS: dict[str, int] = {
    # OLMo2 models
    "olmo2_1m": 1_000_000,
    "olmo2_30m": 30_000_000,
    "olmo2_60m": 60_000_000,
    "olmo2_190m": 190_000_000,
    "olmo2_1b": 1_000_000_000,
    "olmo2_7b": 7_000_000_000,
    # OLMo3 models (from OLMo-core)
    "olmo3_1m": 1_000_000,
    "olmo3_14m": 14_000_000,
    "olmo3_30m": 30_000_000,
    "olmo3_60m": 60_000_000,
    "olmo3_100m": 100_000_000,
    "olmo3_190m": 190_000_000,
    "olmo3_370m": 370_000_000,
    "olmo3_600m": 600_000_000,
    "olmo3_760m": 760_000_000,
    "olmo3_1b": 1_000_000_000,
    "olmo3_3b": 3_000_000_000,
    "olmo3_7b": 7_000_000_000,
    "olmo3_13b": 13_000_000_000,
    "olmo3_32b": 32_000_000_000,
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


class Priority(StrEnum):
    """Beaker job priority levels."""

    low = "low"
    normal = "normal"
    high = "high"
    urgent = "urgent"


class TrainType(Enum):
    pretrain = "pretrain"
    anneal = "anneal"


class QualityConfig(BaseModel):
    """Configuration for a quality level within a topic or source.

    Name can be any string like "vigintile_0001", "high", "medium", "low", etc.
    """

    name: str  # e.g., "vigintile_0001", "high", "low"
    paths: list[str]
    weight: float | None = None  # Optional weight for quality bucket (used in upsampling experiments)


class TopicConfig(BaseModel):
    name: str
    paths: list[str] | None = None
    quality: list[QualityConfig] | None = None
    weight: float | None = None

    def model_post_init(self, __context) -> None:
        """Validate that exactly one of paths or quality is provided."""
        if self.paths is not None and self.quality is not None:
            raise ValueError("TopicConfig cannot have both 'paths' and 'quality' - use exactly one")
        if self.paths is None and self.quality is None:
            raise ValueError("TopicConfig must have either 'paths' or 'quality'")


class SourceConfig(BaseModel):
    name: str
    paths: list[str] | None = None
    topics: list[TopicConfig] | None = None
    quality: list[QualityConfig] | None = None
    weight: float | None = None

    def model_post_init(self, __context) -> None:
        """Validate that exactly one of paths, topics, or quality is provided."""
        options = [self.paths is not None, self.topics is not None, self.quality is not None]
        if sum(options) != 1:
            raise ValueError("SourceConfig must have exactly one of 'paths', 'topics', or 'quality'")


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


class InfraConfig(BaseModel):
    """Beaker launch infrastructure settings."""

    budget: str
    workspace: str
    cluster: str
    priority: Priority = Priority.normal
    preemptible: bool = True
    nodes: int = 1
    gpus: int
    weka: bool = False
    shared_filesystem: bool = False
    wandb_debug: bool = False


class TrainingConfig(BaseModel):
    """OLMo Core model + training + eval settings."""

    proxy_model_id: str
    tokenizer: str
    chinchilla_multiple: float
    seed: int
    device_batch_size: int = 4
    global_batch_size: int | None = None
    checkpoint_path: str | None = None
    train_type: TrainType = TrainType.pretrain
    eval_interval: int = 1000  # Steps between evaluations
    no_eval: bool = False  # Disable downstream evaluations
    instance_filter: InstanceFilterConfig | None = None  # Optional quality filter for repetitive sequences

    def get_max_tokens(self) -> int:
        """Compute the maximum training tokens from chinchilla_multiple and model size.

        Returns:
            The total number of tokens to train for.
        """
        num_params = get_model_num_params(self.proxy_model_id)
        return compute_max_tokens(self.chinchilla_multiple, num_params)


class DataConfig(BaseModel):
    """Data landscape: what data exists and how it's stored."""

    sources: list[SourceConfig]
    dtype: NumpyDatasetDType = NumpyDatasetDType.uint32


class SwarmConfig(BaseModel):
    """Outer-loop mixture sampling parameters."""

    seed: int = 42  # Dirichlet sampling seed for synthesize_mixture.py
    variants: int = 1
    mix_temperature: float = 1.0
    source_mix_temperature: float | None = None
    topic_mix_temperature: float | None = None
    min_strength: float = 0.1
    max_strength: float = 5.0
    min_source_strength: float | None = None
    max_source_strength: float | None = None
    min_topic_strength: float | None = None
    max_topic_strength: float | None = None
    minimum_weight: float | None = None
    minimum_source_weight: float | None = None
    minimum_topic_weight: float | None = None
    nonzero_weight: list[str] | None = None
    fixed_source_weights: dict[str, float] | None = None
    manual_prior: dict[str, float] | None = None
    manual_topic_prior: dict[str, float] | None = None
    allow_repetition: bool = True
    sample_multiplier: int | None = None
    existing_mix_file: str | None = None


class GenerationConfig(BaseModel):
    """Input to ``olmix generate``. Contains everything needed to sample mixes."""

    name: str = ""
    data: DataConfig
    priors: PriorsConfig
    swarm: SwarmConfig = SwarmConfig()
    max_tokens: int

    @classmethod
    def from_yaml(cls, path: PathType) -> "GenerationConfig":
        """Load a GenerationConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class MixEntry(BaseModel):
    """A single domain entry in a mixture variant."""

    weight: float
    repetition_factor: float


def _flatten_mix_node(
    prefix: str,
    node: dict[str, Any],
    result: dict[str, Any],
    inherited_weight: float = 1.0,
    inherited_rep: float = 1.0,
) -> None:
    """Recursively flatten a nested mix node into leaf entries.

    At each level, ``weight`` and ``repetition_factor`` are properties;
    all other keys are children.  The final leaf weight is the product of
    weights along the path from root to leaf.  ``repetition_factor`` is
    inherited from the nearest ancestor that sets it.
    """
    weight = inherited_weight * node.get("weight", 1.0)
    rep = node.get("repetition_factor", inherited_rep)

    children = {k: v for k, v in node.items() if k not in ("weight", "repetition_factor")}

    if not children:
        # Leaf node
        result[prefix] = {"weight": weight, "repetition_factor": rep}
    else:
        for child_name, child_value in children.items():
            child_key = f"{prefix}:{child_name}"
            if isinstance(child_value, dict):
                _flatten_mix_node(child_key, child_value, result, weight, rep)
            else:
                # Scalar value is a weight shorthand
                result[child_key] = {"weight": weight * float(child_value), "repetition_factor": rep}


def flatten_mix(mix: dict[str, Any]) -> dict[str, Any]:
    """Flatten a potentially nested mix dict into leaf-level entries.

    Supports two formats:

    **Flat** (generated configs) — keys are colon-separated domain paths,
    values are ``{weight, repetition_factor}``::

        {"dclm:code": {"weight": 0.5, "repetition_factor": 1.0}}

    **Nested** (hand-written configs) — keys are source/topic/quality names,
    nesting mirrors the data hierarchy::

        {"dclm": {"weight": 0.98, "code": {"weight": 0.5}}}

    Returns a flat dict suitable for ``dict[str, MixEntry]``.
    """
    flat: dict[str, Any] = {}
    for key, value in mix.items():
        if not isinstance(value, dict):
            flat[key] = value
            continue
        non_props = {k for k in value if k not in ("weight", "repetition_factor")}
        if not non_props:
            # Already a leaf entry
            flat[key] = value
        else:
            # Nested — flatten recursively
            _flatten_mix_node(key, value, flat)
    return flat


class LaunchConfig(BaseModel):
    """Base config for ``olmix launch``. Everything needed to run training."""

    name: str
    description: str = ""
    infra: InfraConfig
    training: TrainingConfig
    data: DataConfig
    eval: InLoopEvalConfig
    mix: dict[str, MixEntry] | None = None
    group_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _resolve_nested_mix(cls, data: Any) -> Any:
        """Auto-flatten nested mix dicts into leaf-level MixEntry dicts."""
        if isinstance(data, dict) and isinstance(data.get("mix"), dict):
            data = dict(data)
            data["mix"] = flatten_mix(data["mix"])
        return data

    @classmethod
    def from_yaml(cls, path: PathType) -> "LaunchConfig":
        """Load a LaunchConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)


class ExperimentInstance(BaseModel):
    """A single experiment instance with its mixture configuration."""

    name: str
    sources: list[SourceInstance]


class ExperimentGroup(BaseModel):
    """A group of experiment instances sharing a common configuration."""

    config: LaunchConfig
    group_id: str
    instances: list[ExperimentInstance]
