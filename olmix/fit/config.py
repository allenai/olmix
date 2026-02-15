"""Pydantic configuration models for olmix fit.

Defines the YAML-driven configuration for `olmix fit --config <yaml>`.
"""

from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Union

import yaml
from pydantic import BaseModel, Discriminator, Tag, model_validator

PathType = Union[Path, PathLike[Any], str]


class SwarmDataConfig(BaseModel):
    """Paths to the swarm CSV files (ratios and metrics)."""

    ratios: str
    metrics: str


class PriorsConfig(BaseModel):
    """Token distribution across domains.

    Only ``token_counts`` is stored; ``relative_sizes`` and ``total_tokens``
    are computed properties derived from it.
    """

    token_counts: dict[str, int]

    @model_validator(mode="before")
    @classmethod
    def _strip_derived_fields(cls, data: Any) -> Any:
        """Drop ``relative_sizes`` and ``total_tokens`` from input for backward compat."""
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k not in ("relative_sizes", "total_tokens")}
        return data

    @property
    def relative_sizes(self) -> dict[str, float]:
        """Normalized token fractions (sum to 1)."""
        total = sum(self.token_counts.values())
        if total == 0:
            return {k: 0.0 for k in self.token_counts}
        return {k: v / total for k, v in self.token_counts.items()}

    @property
    def total_tokens(self) -> int:
        """Total token count across all domains."""
        return sum(self.token_counts.values())


class RegressionConfig(BaseModel):
    """Regression model settings."""

    type: str = "log_linear"
    seed: int = 0
    n_test: int = 0
    train_split: list[float] = [1.0]
    aggregate_task_families: bool = False


class ProposerConfig(BaseModel):
    """Mixture proposer settings."""

    type: str = "exact"
    temperature: float | None = None
    kl_reg: float | None = None
    use_natural_kl: bool = False
    fit_only: bool = False
    make_worst_mix: bool = False


class ConstraintsConfig(BaseModel):
    """Token budget constraints."""

    enabled: bool = False
    target_tokens: int | None = None
    repetition_factor: float = 5.0


class FilteringConfig(BaseModel):
    """Domain/metric filtering."""

    keep_sources: list[str] = []
    support_domains: list[str] = []
    drop_metrics: list[str] = []
    fixed_weight: dict[str, float] = {}
    obj_weights: dict[str, float] = {}


class InLoopEvalConfig(BaseModel):
    """Eval config for in-loop (WandB) metrics.

    Tasks are nested by family: {family: {task_id: wandb_metric_name}}.
    Used by ``olmix launch`` (task_ids) and ``olmix fit`` (metric_names).
    """

    type: Literal["inloop"] = "inloop"
    tasks: dict[str, dict[str, str]]

    @property
    def metric_names(self) -> list[str]:
        """Flattened list of WandB metric names (CSV column names)."""
        return [name for family in self.tasks.values() for name in family.values()]

    @property
    def task_ids(self) -> list[str]:
        """Flattened list of olmo-core task IDs."""
        return [tid for family in self.tasks.values() for tid in family.keys()]

    @property
    def task_families(self) -> dict[str, list[str]]:
        """Family → list of metric names."""
        return {family: list(mapping.values()) for family, mapping in self.tasks.items()}


class OfflineEvalConfig(BaseModel):
    """Eval config for offline (cookbook-eval) metrics.

    Tasks are nested by family: {family: [metric_name, ...]}.
    Used by ``olmix fit`` only.
    """

    type: Literal["offline"] = "offline"
    tasks: dict[str, list[str]]

    @property
    def metric_names(self) -> list[str]:
        """Flattened list of metric names (CSV column names)."""
        return [name for names in self.tasks.values() for name in names]

    @property
    def task_families(self) -> dict[str, list[str]]:
        """Family → list of metric names."""
        return dict(self.tasks)


def _eval_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        return v.get("type", "offline")
    return getattr(v, "type", "offline")


EvalConfig = Annotated[
    Union[
        Annotated[InLoopEvalConfig, Tag("inloop")],
        Annotated[OfflineEvalConfig, Tag("offline")],
    ],
    Discriminator(_eval_discriminator),
]


class FitConfig(BaseModel):
    """Top-level fit configuration, composed of all sub-configs."""

    swarm: SwarmDataConfig
    priors: PriorsConfig
    eval: EvalConfig
    regression: RegressionConfig = RegressionConfig()
    proposer: ProposerConfig = ProposerConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    filtering: FilteringConfig = FilteringConfig()

    @classmethod
    def _evaluate_fraction(cls, value: str) -> float:
        """Safely evaluate fraction strings like '7/52.0' to floats."""
        # Try to parse as a simple division expression
        if "/" in value:
            try:
                parts = value.split("/")
                if len(parts) == 2:
                    numerator = float(parts[0].strip())
                    denominator = float(parts[1].strip())
                    return numerator / denominator
            except (ValueError, ZeroDivisionError):
                pass
        # Fallback to direct float conversion
        return float(value)

    @classmethod
    def from_yaml(cls, path: PathType) -> "FitConfig":
        """Load a FitConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        # Preprocess fraction strings in filtering.obj_weights and filtering.fixed_weight
        if "filtering" in data and isinstance(data["filtering"], dict):
            for field_name in ["obj_weights", "fixed_weight"]:
                if field_name in data["filtering"] and isinstance(data["filtering"][field_name], dict):
                    result = {}
                    for key, value in data["filtering"][field_name].items():
                        if isinstance(value, str):
                            result[key] = cls._evaluate_fraction(value)
                        else:
                            result[key] = float(value)
                    data["filtering"][field_name] = result

        return cls(**data)
