"""Tests for FitConfig and related Pydantic models."""

import pytest
import yaml

from olmix.fit.config import (
    ConstraintsConfig,
    FilteringConfig,
    FitConfig,
    InLoopEvalConfig,
    OfflineEvalConfig,
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
            "token_counts": {"domain_a": 600_000, "domain_b": 400_000},
        },
        "eval": {
            "type": "offline",
            "tasks": {
                "math": ["metric_a", "metric_b"],
                "code": ["metric_c"],
            },
        },
    }


class TestFitConfig:
    def test_minimal_config(self, sample_config_dict):
        cfg = FitConfig(**sample_config_dict)
        assert cfg.swarm.ratios == "ratios.csv"
        assert cfg.swarm.metrics == "metrics.csv"

    def test_defaults(self, sample_config_dict):
        cfg = FitConfig(**sample_config_dict)
        assert cfg.regression.type == "log_linear"
        assert cfg.regression.seed == 0
        assert cfg.regression.n_test == 0
        assert cfg.regression.train_split == 1.0
        assert cfg.regression.aggregate_task_families is False

        assert cfg.proposer.type == "exact"
        assert cfg.proposer.temperature is None
        assert cfg.proposer.kl_reg is None
        assert cfg.proposer.fit_only is False
        assert cfg.proposer.make_worst_mix is False

        assert cfg.constraints.enabled is False
        assert cfg.constraints.target_tokens is None
        assert cfg.constraints.repetition_factor == 4.0

        assert cfg.filtering.drop_metrics == []
        assert cfg.filtering.obj_weights == {}

    def test_from_yaml(self, sample_config_dict, tmp_path):
        config_file = tmp_path / "fit.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        cfg = FitConfig.from_yaml(config_file)
        assert cfg.swarm.ratios == "ratios.csv"

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
            FitConfig(
                swarm={"ratios": "r.csv", "metrics": "m.csv"},
                eval={"type": "offline", "tasks": {"qa": ["m1"]}},
            )

    def test_full_config_yaml(self, tmp_path):
        """Test loading a fully-specified YAML (all sections)."""
        full = {
            "swarm": {"ratios": "r.csv", "metrics": "m.csv"},
            "priors": {
                "relative_sizes": {"a": 0.5, "b": 0.5},
                "token_counts": {"a": 1_000_000, "b": 1_000_000},
            },
            "eval": {
                "type": "offline",
                "tasks": {
                    "math": ["metric_a"],
                    "code": ["metric_b", "metric_c"],
                },
            },
            "regression": {
                "type": "log_linear",
                "seed": 42,
                "n_test": 10,
                "train_split": 0.8,
                "aggregate_task_families": True,
            },
            "proposer": {
                "type": "exact",
                "temperature": 0.5,
                "kl_reg": 0.1,
                "fit_only": False,
                "make_worst_mix": False,
            },
            "constraints": {
                "enabled": True,
                "target_tokens": 1_000_000_000,
                "repetition_factor": 3.0,
            },
            "filtering": {
                "drop_metrics": ["bad_metric"],
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


class TestPriorsConfig:
    def test_to_tuple(self):
        priors = PriorsConfig(
            relative_sizes={"a": 0.6, "b": 0.4},
            token_counts={"a": 600_000, "b": 400_000},
        )
        rel, counts = priors.to_tuple()
        assert rel == {"a": 0.6, "b": 0.4}
        assert sum(counts.values()) == 1_000_000
        assert counts == {"a": 600_000, "b": 400_000}

    def test_to_tuple_returns_copies(self):
        """Mutating the returned dicts should not affect the original."""
        priors = PriorsConfig(
            relative_sizes={"a": 0.6, "b": 0.4},
            token_counts={"a": 600_000, "b": 400_000},
        )
        rel, counts = priors.to_tuple()
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
        assert cfg.train_split == 1.0


class TestProposerConfig:
    def test_defaults(self):
        cfg = ProposerConfig()
        assert cfg.type == "exact"
        assert cfg.fit_only is False


class TestConstraintsConfig:
    def test_defaults(self):
        cfg = ConstraintsConfig()
        assert cfg.enabled is False
        assert cfg.repetition_factor == 4.0


class TestFilteringConfig:
    def test_defaults(self):
        cfg = FilteringConfig()
        assert cfg.obj_weights == {}


class TestInLoopEvalConfig:
    def test_basic(self):
        cfg = InLoopEvalConfig(
            tasks={
                "math": {"gsm8k_gold_bpb_5shot": "eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)"},
                "code": {"codex_humaneval_gold_bpb_3shot": "eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2)"},
            }
        )
        assert cfg.type == "inloop"
        assert len(cfg.task_ids) == 2
        assert len(cfg.metric_names) == 2
        assert "gsm8k_gold_bpb_5shot" in cfg.task_ids
        assert "eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)" in cfg.metric_names

    def test_task_families(self):
        cfg = InLoopEvalConfig(
            tasks={
                "math": {"task_a": "metric_a", "task_b": "metric_b"},
                "code": {"task_c": "metric_c"},
            }
        )
        families = cfg.task_families
        assert set(families.keys()) == {"math", "code"}
        assert families["math"] == ["metric_a", "metric_b"]
        assert families["code"] == ["metric_c"]


class TestOfflineEvalConfig:
    def test_basic(self):
        cfg = OfflineEvalConfig(
            tasks={
                "math": ["minerva_math_algebra::olmes"],
                "code": ["codex_humaneval:3shot::none", "mbpp:3shot::none"],
            }
        )
        assert cfg.type == "offline"
        assert len(cfg.metric_names) == 3
        assert "minerva_math_algebra::olmes" in cfg.metric_names

    def test_task_families(self):
        cfg = OfflineEvalConfig(
            tasks={
                "math": ["m1", "m2"],
                "code": ["m3"],
            }
        )
        families = cfg.task_families
        assert families["math"] == ["m1", "m2"]
        assert families["code"] == ["m3"]


class TestEvalDiscriminator:
    def test_offline_default(self):
        """When type is omitted, should default to offline."""
        cfg = FitConfig(
            swarm={"ratios": "r.csv", "metrics": "m.csv"},
            priors={
                "relative_sizes": {"a": 0.5, "b": 0.5},
                "token_counts": {"a": 500_000, "b": 500_000},
            },
            eval={"tasks": {"qa": ["metric1"]}},
        )
        assert cfg.eval.type == "offline"

    def test_inloop_explicit(self):
        cfg = FitConfig(
            swarm={"ratios": "r.csv", "metrics": "m.csv"},
            priors={
                "relative_sizes": {"a": 0.5, "b": 0.5},
                "token_counts": {"a": 500_000, "b": 500_000},
            },
            eval={"type": "inloop", "tasks": {"qa": {"task_a": "metric_a"}}},
        )
        assert cfg.eval.type == "inloop"


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

    def test_eval_config(self):
        """Verify the eval section of the shipped config."""
        cfg = FitConfig.from_yaml("configs/fits/dclm_baseline.yaml")
        assert cfg.eval.type == "offline"
        assert len(cfg.eval.metric_names) == 110
        families = cfg.eval.task_families
        assert set(families.keys()) == {"math", "code", "qa"}
        assert len(families["math"]) == 7
        assert len(families["code"]) == 19
        assert len(families["qa"]) == 84
