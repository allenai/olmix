"""Tests for FitConfig and related Pydantic models."""

import pytest
import yaml

from olmix.fit.config import (
    ConstraintsConfig,
    FilteringConfig,
    FitConfig,
    PriorsConfig,
    ProposerConfig,
    RegressionConfig,
    SwarmDataConfig,
)


@pytest.fixture
def sample_config_dict():
    """Minimal valid FitConfig dict."""
    return {
        "swarm": {
            "ratios": "ratios.csv",
            "metrics": "metrics.csv",
        },
        "priors": {
            "relative_sizes": {"domain_a": 0.6, "domain_b": 0.4},
            "total_tokens": 1_000_000,
            "token_counts": {"domain_a": 600_000, "domain_b": 400_000},
        },
    }


class TestFitConfig:
    def test_minimal_config(self, sample_config_dict):
        cfg = FitConfig(**sample_config_dict)
        assert cfg.swarm.ratios == "ratios.csv"
        assert cfg.swarm.metrics == "metrics.csv"
        assert cfg.priors.total_tokens == 1_000_000

    def test_defaults(self, sample_config_dict):
        cfg = FitConfig(**sample_config_dict)
        assert cfg.regression.type == "log_linear"
        assert cfg.regression.seed == 0
        assert cfg.regression.n_test == 0
        assert cfg.regression.train_split == [1.0]
        assert cfg.regression.aggregate_task_families is False

        assert cfg.proposer.type == "exact"
        assert cfg.proposer.temperature is None
        assert cfg.proposer.kl_reg is None
        assert cfg.proposer.use_natural_kl is False
        assert cfg.proposer.fit_only is False
        assert cfg.proposer.make_worst_mix is False

        assert cfg.constraints.enabled is False
        assert cfg.constraints.target_tokens is None
        assert cfg.constraints.repetition_factor == 5.0

        assert cfg.filtering.keep_sources == []
        assert cfg.filtering.support_domains == []
        assert cfg.filtering.drop_metrics == []
        assert cfg.filtering.fixed_weight == {}
        assert cfg.filtering.obj_weights == {}

    def test_from_yaml(self, sample_config_dict, tmp_path):
        config_file = tmp_path / "fit.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        cfg = FitConfig.from_yaml(config_file)
        assert cfg.swarm.ratios == "ratios.csv"
        assert cfg.priors.total_tokens == 1_000_000

    def test_from_yaml_with_overrides(self, sample_config_dict, tmp_path):
        sample_config_dict["regression"] = {"type": "lightgbm"}
        sample_config_dict["proposer"] = {"type": "simulation", "fit_only": True}

        config_file = tmp_path / "fit.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        cfg = FitConfig.from_yaml(config_file)
        assert cfg.regression.type == "lightgbm"
        assert cfg.proposer.type == "simulation"
        assert cfg.proposer.fit_only is True

    def test_round_trip(self, sample_config_dict, tmp_path):
        """model_dump -> re-parse should be identical."""
        cfg = FitConfig(**sample_config_dict)
        dumped = cfg.model_dump()
        cfg2 = FitConfig(**dumped)
        assert cfg == cfg2

    def test_missing_required_fields(self):
        with pytest.raises(ValueError):
            FitConfig()

        with pytest.raises(ValueError):
            FitConfig(swarm={"ratios": "r.csv", "metrics": "m.csv"})

    def test_full_config_yaml(self, tmp_path):
        """Test loading a fully-specified YAML (all sections)."""
        full = {
            "swarm": {"ratios": "r.csv", "metrics": "m.csv"},
            "priors": {
                "relative_sizes": {"a": 0.5, "b": 0.5},
                "total_tokens": 2_000_000,
                "token_counts": {"a": 1_000_000, "b": 1_000_000},
            },
            "regression": {
                "type": "log_linear",
                "seed": 42,
                "n_test": 10,
                "train_split": [0.8],
                "aggregate_task_families": True,
            },
            "proposer": {
                "type": "exact",
                "temperature": 0.5,
                "kl_reg": 0.1,
                "use_natural_kl": True,
                "fit_only": False,
                "make_worst_mix": False,
            },
            "constraints": {
                "enabled": True,
                "target_tokens": 1_000_000_000,
                "repetition_factor": 3.0,
            },
            "filtering": {
                "keep_sources": ["a"],
                "support_domains": ["a", "b"],
                "drop_metrics": ["bad_metric"],
                "fixed_weight": {"a": 0.7},
                "obj_weights": {"task1": 0.5, "task2": 0.5},
            },
        }
        config_file = tmp_path / "full.yaml"
        with open(config_file, "w") as f:
            yaml.dump(full, f)

        cfg = FitConfig.from_yaml(config_file)
        assert cfg.regression.seed == 42
        assert cfg.constraints.enabled is True
        assert cfg.constraints.target_tokens == 1_000_000_000
        assert cfg.filtering.keep_sources == ["a"]
        assert cfg.filtering.fixed_weight == {"a": 0.7}


class TestPriorsConfig:
    def test_to_tuple(self):
        priors = PriorsConfig(
            relative_sizes={"a": 0.6, "b": 0.4},
            total_tokens=1_000_000,
            token_counts={"a": 600_000, "b": 400_000},
        )
        rel, total, counts = priors.to_tuple()
        assert rel == {"a": 0.6, "b": 0.4}
        assert total == 1_000_000
        assert counts == {"a": 600_000, "b": 400_000}

    def test_to_tuple_returns_copies(self):
        """Mutating the returned dicts should not affect the original."""
        priors = PriorsConfig(
            relative_sizes={"a": 0.6, "b": 0.4},
            total_tokens=1_000_000,
            token_counts={"a": 600_000, "b": 400_000},
        )
        rel, _, counts = priors.to_tuple()
        rel["c"] = 0.0
        counts["c"] = 0
        assert "c" not in priors.relative_sizes
        assert "c" not in priors.token_counts


class TestSwarmDataConfig:
    def test_basic(self):
        cfg = SwarmDataConfig(ratios="r.csv", metrics="m.csv")
        assert cfg.ratios == "r.csv"
        assert cfg.metrics == "m.csv"

    def test_missing_field(self):
        with pytest.raises(ValueError):
            SwarmDataConfig(ratios="r.csv")


class TestRegressionConfig:
    def test_defaults(self):
        cfg = RegressionConfig()
        assert cfg.type == "log_linear"
        assert cfg.train_split == [1.0]


class TestProposerConfig:
    def test_defaults(self):
        cfg = ProposerConfig()
        assert cfg.type == "exact"
        assert cfg.fit_only is False


class TestConstraintsConfig:
    def test_defaults(self):
        cfg = ConstraintsConfig()
        assert cfg.enabled is False
        assert cfg.repetition_factor == 5.0


class TestFilteringConfig:
    def test_defaults(self):
        cfg = FilteringConfig()
        assert cfg.fixed_weight == {}
        assert cfg.obj_weights == {}

    def test_native_dict(self):
        cfg = FilteringConfig(fixed_weight={"domain_a": 0.3, "domain_b": 0.7})
        assert cfg.fixed_weight["domain_a"] == 0.3


class TestDCLMBaselineConfig:
    def test_loads_example_config(self):
        """Verify the shipped example config loads and validates."""
        cfg = FitConfig.from_yaml("configs/fits/dclm_baseline.yaml")
        assert cfg.swarm.ratios == "dclm_ratios.csv"
        assert cfg.swarm.metrics == "dclm_metrics.csv"
        assert len(cfg.priors.relative_sizes) == 24
        assert cfg.regression.type == "log_linear"
        assert cfg.proposer.type == "exact"
        assert cfg.proposer.kl_reg == 0.1
