"""Tests for config types and related configuration classes."""

import pytest

from olmix.aliases import (
    TOKENS_PER_PARAM,
    DataConfig,
    ExperimentGroup,
    ExperimentInstance,
    GenerationConfig,
    InfraConfig,
    InstanceFilterConfig,
    LaunchConfig,
    MixEntry,
    Priority,
    SourceConfig,
    SourceInstance,
    TopicConfig,
    TrainingConfig,
    TrainType,
    compute_max_tokens,
    flatten_mix,
    get_model_num_params,
)
from olmix.fit.config import InLoopEvalConfig

# Minimal InLoopEvalConfig for tests
_MINIMAL_EVAL = InLoopEvalConfig(
    tasks={"qa": {"arc_challenge_test_rc_5shot": "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"}},
)


class TestSourceConfig:
    """Test SourceConfig model."""

    def test_basic_source_config(self):
        """Test creating a basic source config."""
        source = SourceConfig(
            name="wikipedia",
            paths=["s3://bucket/wiki/**/*.npy"],
        )

        assert source.name == "wikipedia"
        assert len(source.paths) == 1
        assert source.topics is None

    def test_source_with_weight(self):
        """Test source config with weight field."""
        source = SourceConfig(
            name="wikipedia",
            paths=["s3://bucket/wiki/**/*.npy"],
            weight=2.5,
        )

        assert source.weight == 2.5

    def test_source_weight_default_none(self):
        """Test source config weight defaults to None."""
        source = SourceConfig(
            name="wikipedia",
            paths=["s3://bucket/wiki/**/*.npy"],
        )

        assert source.weight is None

    def test_source_with_topics(self):
        """Test source config with topics."""
        source = SourceConfig(
            name="dclm",
            topics=[
                TopicConfig(name="math", paths=["s3://bucket/dclm/math/*.npy"]),
                TopicConfig(name="code", paths=["s3://bucket/dclm/code/*.npy"]),
            ],
        )

        assert source.name == "dclm"
        assert source.paths is None
        assert len(source.topics) == 2
        assert source.topics[0].name == "math"
        assert source.topics[1].name == "code"


class TestSourceInstance:
    """Test SourceInstance model."""

    def test_source_instance(self):
        """Test creating a source instance."""
        instance = SourceInstance(
            name="wikipedia",
            paths=["s3://bucket/wiki/part1.npy", "s3://bucket/wiki/part2.npy"],
            ratio=0.5,
            repetition_factor=1.5,
        )

        assert instance.name == "wikipedia"
        assert len(instance.paths) == 2
        assert instance.ratio == 0.5
        assert instance.repetition_factor == 1.5


class TestLaunchConfig:
    """Test LaunchConfig model."""

    @pytest.fixture
    def sample_config_dict(self):
        """Return a sample config dictionary."""
        return {
            "name": "test-swarm",
            "description": "Test experiment",
            "infra": {
                "budget": "ai2/oe-data",
                "workspace": "ai2/dolma2",
                "cluster": "ai2/saturn-cirrascale",
                "nodes": 1,
                "gpus": 8,
            },
            "training": {
                "proxy_model_id": "olmo2_30m",
                "tokenizer": "gpt_neox",
                "chinchilla_multiple": 1.0,
                "seed": 42,
            },
            "data": {
                "sources": [
                    {"name": "wikipedia", "paths": ["s3://bucket/wiki/**/*.npy"]},
                    {"name": "dclm", "paths": ["s3://bucket/dclm/**/*.npy"]},
                ],
            },
            "eval": {
                "type": "inloop",
                "tasks": {
                    "qa": {"arc_challenge_test_rc_5shot": "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"},
                },
            },
        }

    def test_valid_config_parses(self, sample_config_dict):
        """Test that a valid config parses successfully."""
        config = LaunchConfig(**sample_config_dict)

        assert config.name == "test-swarm"
        assert len(config.data.sources) == 2
        assert config.training.chinchilla_multiple == 1.0

    def test_config_defaults(self, sample_config_dict):
        """Test default values are set correctly."""
        config = LaunchConfig(**sample_config_dict)

        assert config.infra.priority == Priority.normal
        assert config.training.train_type == TrainType.pretrain
        assert config.infra.preemptible is True
        assert config.infra.weka is False

    def test_config_from_yaml(self, sample_config_dict, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = LaunchConfig.from_yaml(config_file)

        assert config.name == "test-swarm"
        assert len(config.data.sources) == 2


class TestGenerationConfig:
    """Test GenerationConfig model."""

    @pytest.fixture
    def sample_config_dict(self):
        """Return a sample generation config dictionary."""
        return {
            "name": "test-gen",
            "data": {
                "sources": [
                    {"name": "wikipedia", "paths": ["s3://bucket/wiki/**/*.npy"]},
                    {"name": "dclm", "paths": ["s3://bucket/dclm/**/*.npy"]},
                ],
            },
            "priors": {
                "token_counts": {"wikipedia": 1_000_000, "dclm": 500_000},
                "relative_sizes": {"wikipedia": 0.66, "dclm": 0.34},
            },
            "swarm": {
                "seed": 42,
                "variants": 4,
            },
            "max_tokens": 600_000_000,
        }

    def test_valid_config_parses(self, sample_config_dict):
        """Test that a valid generation config parses successfully."""
        config = GenerationConfig(**sample_config_dict)

        assert config.name == "test-gen"
        assert config.max_tokens == 600_000_000
        assert config.swarm.variants == 4
        assert len(config.data.sources) == 2

    def test_config_from_yaml(self, sample_config_dict, tmp_path):
        """Test loading generation config from YAML file."""
        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = GenerationConfig.from_yaml(config_file)

        assert config.name == "test-gen"
        assert config.max_tokens == 600_000_000


class TestLaunchConfigMix:
    """Test LaunchConfig mix and group_id fields."""

    def test_launch_config_without_mix(self):
        """Test that LaunchConfig works without mix/group_id (backward compat)."""
        config = LaunchConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test", gpus=1),
            training=TrainingConfig(proxy_model_id="olmo2_30m", tokenizer="dolma2", chinchilla_multiple=1.0, seed=42),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
            eval=_MINIMAL_EVAL,
        )
        assert config.mix is None
        assert config.group_id is None

    def test_launch_config_with_mix(self):
        """Test LaunchConfig with mix and group_id fields."""
        config = LaunchConfig(
            name="test-abc123-0000",
            infra=InfraConfig(budget="test", workspace="test", cluster="test", gpus=1),
            training=TrainingConfig(proxy_model_id="olmo2_30m", tokenizer="dolma2", chinchilla_multiple=1.0, seed=42),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
            eval=_MINIMAL_EVAL,
            mix={
                "wiki": MixEntry(weight=0.6, repetition_factor=1.0),
                "code": MixEntry(weight=0.4, repetition_factor=1.5),
            },
            group_id="abc12345",
        )
        assert config.mix is not None
        assert len(config.mix) == 2
        assert config.mix["wiki"].weight == 0.6
        assert config.mix["code"].repetition_factor == 1.5
        assert config.group_id == "abc12345"

    def test_launch_config_with_mix_from_yaml(self, tmp_path):
        """Test loading LaunchConfig with mix from YAML."""
        import yaml

        config_data = {
            "name": "test-abc123-0000",
            "group_id": "abc12345",
            "infra": {"budget": "test", "workspace": "test", "cluster": "test", "gpus": 1},
            "training": {"proxy_model_id": "olmo2_30m", "tokenizer": "dolma2", "chinchilla_multiple": 1.0, "seed": 42},
            "data": {"sources": [{"name": "wiki", "paths": ["test.npy"]}]},
            "eval": {"type": "inloop", "tasks": {"qa": {"arc": "eval/arc (BPB v2)"}}},
            "mix": {
                "wiki": {"weight": 0.6, "repetition_factor": 1.0},
                "code": {"weight": 0.4, "repetition_factor": 1.5},
            },
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = LaunchConfig.from_yaml(config_file)
        assert config.mix is not None
        assert config.mix["wiki"].weight == 0.6
        assert config.group_id == "abc12345"


class TestNestedMix:
    """Test nested mix format auto-flattening."""

    def test_flat_mix_passthrough(self):
        """Test that flat mix entries pass through unchanged."""
        flat = flatten_mix(
            {
                "dclm:code": {"weight": 0.5, "repetition_factor": 1.0},
                "wiki": {"weight": 0.5, "repetition_factor": 2.0},
            }
        )
        assert flat == {
            "dclm:code": {"weight": 0.5, "repetition_factor": 1.0},
            "wiki": {"weight": 0.5, "repetition_factor": 2.0},
        }

    def test_nested_two_levels(self):
        """Test source:topic nesting flattens correctly."""
        flat = flatten_mix(
            {
                "dclm": {
                    "weight": 1.0,
                    "code": {"weight": 0.6},
                    "science": {"weight": 0.4},
                },
            }
        )
        assert flat["dclm:code"]["weight"] == pytest.approx(0.6)
        assert flat["dclm:science"]["weight"] == pytest.approx(0.4)

    def test_nested_three_levels(self):
        """Test source:topic:quality nesting flattens with product weights."""
        flat = flatten_mix(
            {
                "all_dressed": {
                    "weight": 0.98,
                    "science": {
                        "weight": 0.20,
                        "high": {"weight": 0.70},
                        "low": {"weight": 0.30},
                    },
                },
                "arxiv": {"weight": 0.02, "repetition_factor": 1.5},
            }
        )
        assert flat["all_dressed:science:high"]["weight"] == pytest.approx(0.98 * 0.20 * 0.70)
        assert flat["all_dressed:science:low"]["weight"] == pytest.approx(0.98 * 0.20 * 0.30)
        assert flat["arxiv"]["weight"] == pytest.approx(0.02)
        assert flat["arxiv"]["repetition_factor"] == 1.5

    def test_repetition_factor_inherited(self):
        """Test that repetition_factor is inherited from parent."""
        flat = flatten_mix(
            {
                "wiki": {
                    "weight": 1.0,
                    "repetition_factor": 2.0,
                    "topic_a": {"weight": 0.6},
                    "topic_b": {"weight": 0.4, "repetition_factor": 3.0},
                },
            }
        )
        assert flat["wiki:topic_a"]["repetition_factor"] == 2.0  # inherited
        assert flat["wiki:topic_b"]["repetition_factor"] == 3.0  # overridden

    def test_default_weight_is_one(self):
        """Test that omitting weight defaults to 1.0."""
        flat = flatten_mix(
            {
                "dclm": {
                    "code": {"weight": 0.5},
                    "science": {"weight": 0.3},
                },
            }
        )
        # dclm has no weight key → defaults to 1.0
        assert flat["dclm:code"]["weight"] == pytest.approx(0.5)
        assert flat["dclm:science"]["weight"] == pytest.approx(0.3)

    def test_nested_mix_in_launch_config(self):
        """Test that LaunchConfig accepts nested mix and flattens it."""
        config = LaunchConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test", gpus=1),
            training=TrainingConfig(proxy_model_id="olmo2_30m", tokenizer="dolma2", chinchilla_multiple=1.0, seed=42),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
            eval=_MINIMAL_EVAL,
            mix={
                "dclm": {
                    "weight": 0.8,
                    "code": {"weight": 0.6},
                    "science": {"weight": 0.4},
                },
                "wiki": {"weight": 0.2, "repetition_factor": 2.0},
            },
        )
        assert config.mix is not None
        assert "dclm:code" in config.mix
        assert "dclm:science" in config.mix
        assert "wiki" in config.mix
        assert config.mix["dclm:code"].weight == pytest.approx(0.48)
        assert config.mix["dclm:science"].weight == pytest.approx(0.32)
        assert config.mix["wiki"].weight == pytest.approx(0.2)
        assert config.mix["wiki"].repetition_factor == 2.0

    def test_nested_mix_from_yaml(self, tmp_path):
        """Test loading nested mix from YAML file."""
        import yaml

        config_data = {
            "name": "test",
            "infra": {"budget": "test", "workspace": "test", "cluster": "test", "gpus": 1},
            "training": {"proxy_model_id": "olmo2_30m", "tokenizer": "dolma2", "chinchilla_multiple": 1.0, "seed": 42},
            "data": {"sources": [{"name": "wiki", "paths": ["test.npy"]}]},
            "eval": {"type": "inloop", "tasks": {"qa": {"arc": "eval/arc (BPB v2)"}}},
            "mix": {
                "all_dressed": {
                    "weight": 0.98,
                    "science": {
                        "weight": 0.5,
                        "high": {"weight": 0.7},
                        "low": {"weight": 0.3},
                    },
                    "code": {
                        "weight": 0.5,
                        "high": {"weight": 0.7},
                        "low": {"weight": 0.3},
                    },
                },
                "arxiv": {"weight": 0.02, "repetition_factor": 1.5},
            },
        }
        config_file = tmp_path / "nested.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        config = LaunchConfig.from_yaml(config_file)
        assert config.mix is not None
        assert config.mix["all_dressed:science:high"].weight == pytest.approx(0.98 * 0.5 * 0.7)
        assert config.mix["all_dressed:code:low"].weight == pytest.approx(0.98 * 0.5 * 0.3)
        assert config.mix["arxiv"].weight == pytest.approx(0.02)
        assert config.mix["arxiv"].repetition_factor == 1.5


class TestHandWrittenConfigsWithMix:
    """Test that migrated hand-written configs parse with mix field."""

    def test_data_proportions_config(self):
        """Test a data_proportions config loads with normalized mix."""
        config = LaunchConfig.from_yaml("configs/experiments/data_proportions/mix_heavy_code.yaml")
        assert config.mix is not None
        assert "dclm:software_development" in config.mix
        assert config.mix["dclm:software_development"].weight == 0.5
        assert config.mix["wikipedia"].repetition_factor == 2.0
        total = sum(e.weight for e in config.mix.values())
        assert abs(total - 1.0) < 1e-4

    def test_quality_upsampling_config(self):
        """Test a quality_upsampling config loads with normalized mix."""
        config = LaunchConfig.from_yaml("configs/experiments/quality_upsampling/heavy_code/aggressive.yaml")
        assert config.mix is not None
        assert "all_dressed:science_math_and_technology:high" in config.mix
        # Weights are normalized to sum to 1.0
        total = sum(e.weight for e in config.mix.values())
        assert abs(total - 1.0) < 1e-4

    def test_training_duration_config(self):
        """Test a training_duration config loads with equal weights."""
        config = LaunchConfig.from_yaml("configs/experiments/training_duration/duration_0.5x.yaml")
        assert config.mix is not None
        assert config.mix["dclm:science_math_and_technology"].weight == pytest.approx(0.2, abs=1e-4)
        assert config.mix["arxiv"].repetition_factor == 1.5


class TestExperimentInstance:
    """Test ExperimentInstance model."""

    def test_experiment_instance(self):
        """Test creating an experiment instance."""
        instance = ExperimentInstance(
            name="test-swarm-abc123-0001",
            sources=[
                SourceInstance(
                    name="wiki",
                    paths=["s3://bucket/wiki/part1.npy"],
                    ratio=0.6,
                    repetition_factor=1.0,
                ),
                SourceInstance(
                    name="code",
                    paths=["s3://bucket/code/part1.npy"],
                    ratio=0.4,
                    repetition_factor=1.0,
                ),
            ],
        )

        assert instance.name == "test-swarm-abc123-0001"
        assert len(instance.sources) == 2
        assert instance.sources[0].ratio + instance.sources[1].ratio == 1.0


class TestExperimentGroup:
    """Test ExperimentGroup model."""

    @pytest.fixture
    def sample_config(self):
        """Return a sample LaunchConfig."""
        return LaunchConfig(
            name="test-swarm",
            description="Test",
            infra=InfraConfig(
                budget="ai2/oe-data",
                workspace="ai2/dolma2",
                cluster="ai2/saturn-cirrascale",
                nodes=1,
                gpus=8,
            ),
            training=TrainingConfig(
                proxy_model_id="olmo2_30m",
                tokenizer="gpt_neox",
                chinchilla_multiple=1.0,
                seed=42,
            ),
            data=DataConfig(
                sources=[
                    SourceConfig(name="wiki", paths=["s3://bucket/wiki/*.npy"]),
                ],
            ),
            eval=_MINIMAL_EVAL,
        )

    def test_experiment_group(self, sample_config):
        """Test creating an experiment group."""
        instances = [
            ExperimentInstance(
                name="test-swarm-abc-0001",
                sources=[SourceInstance(name="wiki", paths=["s3://bucket/wiki/*.npy"], ratio=1.0)],
            ),
            ExperimentInstance(
                name="test-swarm-abc-0002",
                sources=[SourceInstance(name="wiki", paths=["s3://bucket/wiki/*.npy"], ratio=1.0)],
            ),
        ]

        group = ExperimentGroup(
            config=sample_config,
            group_id="abc123",
            instances=instances,
        )

        assert group.group_id == "abc123"
        assert len(group.instances) == 2
        assert group.config.name == "test-swarm"


class TestChinchillaScaling:
    """Test Chinchilla scaling functions."""

    def test_get_model_num_params(self):
        """Test getting model parameter counts."""
        assert get_model_num_params("olmo2_30m") == 30_000_000
        assert get_model_num_params("olmo2_1b") == 1_000_000_000

    def test_get_model_num_params_unknown(self):
        """Test that unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_num_params("unknown_model")

    def test_compute_max_tokens(self):
        """Test computing max tokens from Chinchilla multiple."""
        # 1xC with 30M params = 20 * 30M * 1 = 600M tokens
        assert compute_max_tokens(1.0, 30_000_000) == 600_000_000

        # 5xC with 30M params = 20 * 30M * 5 = 3B tokens
        assert compute_max_tokens(5.0, 30_000_000) == 3_000_000_000

        # 0.5xC with 1B params = 20 * 1B * 0.5 = 10B tokens
        assert compute_max_tokens(0.5, 1_000_000_000) == 10_000_000_000

    def test_tokens_per_param_constant(self):
        """Test that TOKENS_PER_PARAM is 20."""
        assert TOKENS_PER_PARAM == 20


class TestTrainingConfigChinchilla:
    """Test TrainingConfig Chinchilla methods."""

    def test_get_max_tokens(self):
        """Test get_max_tokens method."""
        training = TrainingConfig(
            proxy_model_id="olmo2_30m",
            tokenizer="dolma2",
            chinchilla_multiple=1.0,
            seed=42,
        )
        # 1xC with 30M params = 600M tokens
        assert training.get_max_tokens() == 600_000_000

    def test_get_max_tokens_5x(self):
        """Test get_max_tokens with 5x Chinchilla multiple."""
        training = TrainingConfig(
            proxy_model_id="olmo2_30m",
            tokenizer="dolma2",
            chinchilla_multiple=5.0,
            seed=42,
        )
        # 5xC with 30M params = 3B tokens
        assert training.get_max_tokens() == 3_000_000_000


class TestInstanceFilterConfig:
    """Test InstanceFilterConfig model."""

    def test_default_values(self):
        """Test default values for InstanceFilterConfig."""
        filter_config = InstanceFilterConfig()

        assert filter_config.repetition_min_period == 1
        assert filter_config.repetition_max_period == 13
        assert filter_config.repetition_max_count == 32

    def test_custom_values(self):
        """Test custom values for InstanceFilterConfig."""
        filter_config = InstanceFilterConfig(
            repetition_min_period=2,
            repetition_max_period=20,
            repetition_max_count=64,
        )

        assert filter_config.repetition_min_period == 2
        assert filter_config.repetition_max_period == 20
        assert filter_config.repetition_max_count == 64

    def test_launch_config_with_filter(self):
        """Test LaunchConfig with instance filter."""
        config = LaunchConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test", gpus=1),
            eval=_MINIMAL_EVAL,
            training=TrainingConfig(
                proxy_model_id="olmo2_30m",
                tokenizer="dolma2",
                chinchilla_multiple=1.0,
                seed=42,
                instance_filter=InstanceFilterConfig(
                    repetition_min_period=1,
                    repetition_max_period=13,
                    repetition_max_count=32,
                ),
            ),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
        )

        assert config.training.instance_filter is not None
        assert config.training.instance_filter.repetition_max_period == 13
