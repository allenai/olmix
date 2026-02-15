"""Tests for ExperimentConfig and related configuration classes."""

import pytest

from olmix.aliases import (
    TOKENS_PER_PARAM,
    DataConfig,
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    InfraConfig,
    InstanceFilterConfig,
    Priority,
    SourceConfig,
    SourceInstance,
    SwarmConfig,
    TopicConfig,
    TrainingConfig,
    TrainType,
    compute_max_tokens,
    config_from_path,
    get_model_num_params,
)
from olmix.fit.config import InLoopEvalConfig, PriorsConfig

# Minimal InLoopEvalConfig for tests
_MINIMAL_EVAL = InLoopEvalConfig(
    tasks={"qa": {"arc_challenge_test_rc_5shot": "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"}},
)

# Minimal PriorsConfig for tests
_MINIMAL_PRIORS = PriorsConfig(token_counts={"wiki": 1_000_000, "code": 500_000})


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
        assert source.max_repetition_factor == 1.0
        assert source.topics is None

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


class TestExperimentConfig:
    """Test ExperimentConfig model."""

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
            "priors": {
                "token_counts": {"wikipedia": 1_000_000, "dclm": 500_000},
            },
            "swarm": {
                "seed": 42,
                "variants": 64,
            },
        }

    def test_valid_config_parses(self, sample_config_dict):
        """Test that a valid config parses successfully."""
        config = ExperimentConfig(**sample_config_dict)

        assert config.name == "test-swarm"
        assert config.swarm.variants == 64
        assert len(config.data.sources) == 2
        assert config.training.chinchilla_multiple == 1.0

    def test_config_defaults(self, sample_config_dict):
        """Test default values are set correctly."""
        config = ExperimentConfig(**sample_config_dict)

        assert config.infra.priority == Priority.normal
        assert config.training.train_type == TrainType.pretrain
        assert config.infra.preemptible is True
        assert config.infra.weka is False
        assert config.swarm.mix_temperature == 1.0

    def test_config_from_yaml(self, sample_config_dict, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = ExperimentConfig.from_yaml(config_file)

        assert config.name == "test-swarm"
        assert len(config.data.sources) == 2

    def test_config_from_path_helper(self, sample_config_dict, tmp_path):
        """Test config_from_path helper function."""
        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = config_from_path(config_file)

        assert config.name == "test-swarm"


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
        """Return a sample ExperimentConfig."""
        return ExperimentConfig(
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
            priors=_MINIMAL_PRIORS,
            swarm=SwarmConfig(
                seed=42,
                variants=2,
            ),
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


class TestExperimentConfigChinchilla:
    """Test ExperimentConfig Chinchilla methods."""

    def test_get_max_tokens(self):
        """Test get_max_tokens method."""
        config = ExperimentConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test"),
            training=TrainingConfig(
                proxy_model_id="olmo2_30m",
                tokenizer="dolma2",
                chinchilla_multiple=1.0,
                seed=42,
            ),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
            eval=_MINIMAL_EVAL,
            priors=_MINIMAL_PRIORS,
            swarm=SwarmConfig(seed=42, variants=1),
        )

        # 1xC with 30M params = 600M tokens
        assert config.training.get_max_tokens() == 600_000_000

    def test_get_max_tokens_5x(self):
        """Test get_max_tokens with 5x Chinchilla multiple."""
        config = ExperimentConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test"),
            training=TrainingConfig(
                proxy_model_id="olmo2_30m",
                tokenizer="dolma2",
                chinchilla_multiple=5.0,
                seed=42,
            ),
            data=DataConfig(sources=[SourceConfig(name="wiki", paths=["test.npy"])]),
            eval=_MINIMAL_EVAL,
            priors=_MINIMAL_PRIORS,
            swarm=SwarmConfig(seed=42, variants=1),
        )

        # 5xC with 30M params = 3B tokens
        assert config.training.get_max_tokens() == 3_000_000_000


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

    def test_experiment_config_with_filter(self):
        """Test ExperimentConfig with instance filter."""
        config = ExperimentConfig(
            name="test",
            infra=InfraConfig(budget="test", workspace="test", cluster="test"),
            eval=_MINIMAL_EVAL,
            priors=_MINIMAL_PRIORS,
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
            swarm=SwarmConfig(seed=42, variants=1),
        )

        assert config.training.instance_filter is not None
        assert config.training.instance_filter.repetition_max_period == 13
