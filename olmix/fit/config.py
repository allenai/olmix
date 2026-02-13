"""Pydantic configuration models for olmix fit.

Defines the YAML-driven configuration for `olmix fit --config <yaml>`.
"""

from os import PathLike
from pathlib import Path
from typing import Any, Union

import yaml
from pydantic import BaseModel

PathType = Union[Path, PathLike[Any], str]


class SwarmDataConfig(BaseModel):
    """Paths to the swarm CSV files (ratios and metrics)."""

    ratios: str
    metrics: str


class PriorsConfig(BaseModel):
    """Token distribution across domains (inline priors)."""

    relative_sizes: dict[str, float]
    total_tokens: int
    token_counts: dict[str, int]

    def to_tuple(self) -> tuple[dict[str, float], int, dict[str, int]]:
        """Return the (relative_sizes, total_tokens, token_counts) tuple expected by run_fit."""
        return (dict(self.relative_sizes), self.total_tokens, dict(self.token_counts))


class RegressionConfig(BaseModel):
    """Regression model settings."""

    type: str = "log_linear"
    alpha: float = 1.0
    seed: int = 0
    n_test: int = 0
    train_split: list[float] = [1.0]
    simulation_samples: int = 100_000
    opt_avg_metric: bool = False
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


class FitConfig(BaseModel):
    """Top-level fit configuration, composed of all sub-configs."""

    swarm: SwarmDataConfig
    priors: PriorsConfig
    regression: RegressionConfig = RegressionConfig()
    proposer: ProposerConfig = ProposerConfig()
    constraints: ConstraintsConfig = ConstraintsConfig()
    filtering: FilteringConfig = FilteringConfig()

    @classmethod
    def from_yaml(cls, path: PathType) -> "FitConfig":
        """Load a FitConfig from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
