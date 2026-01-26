"""Tests for ExperimentConfig and related configuration classes."""


import pytest

from olmix.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    Priority,
    SourceConfig,
    SourceInstance,
    TopicConfig,
    TrainType,
    config_from_path,
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
        assert source.max_repetition_factor == 1.0
        assert source.max_source_ratio == 1.0
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
            "budget": "ai2/oe-data",
            "workspace": "ai2/dolma2",
            "nodes": 1,
            "gpus": 8,
            "variants": 64,
            "max_tokens": 1000000000,
            "sequence_length": 2048,
            "seed": 42,
            "cluster": "ai2/saturn-cirrascale",
            "tokenizer": "gpt_neox",
            "proxy_model_id": "olmo_30m",
            "sources": [
                {"name": "wikipedia", "paths": ["s3://bucket/wiki/**/*.npy"]},
                {"name": "dclm", "paths": ["s3://bucket/dclm/**/*.npy"]},
            ],
        }

    def test_valid_config_parses(self, sample_config_dict):
        """Test that a valid config parses successfully."""
        config = ExperimentConfig(**sample_config_dict)

        assert config.name == "test-swarm"
        assert config.variants == 64
        assert len(config.sources) == 2
        assert config.max_tokens == 1000000000

    def test_config_defaults(self, sample_config_dict):
        """Test default values are set correctly."""
        config = ExperimentConfig(**sample_config_dict)

        assert config.priority == Priority.normal
        assert config.train_type == TrainType.pretrain
        assert config.preemptible is True
        assert config.weka is False
        assert config.mix_temperature == 1.0

    def test_config_from_yaml(self, sample_config_dict, tmp_path):
        """Test loading config from YAML file."""
        import yaml

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(sample_config_dict, f)

        config = ExperimentConfig.from_yaml(config_file)

        assert config.name == "test-swarm"
        assert len(config.sources) == 2

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
            budget="ai2/oe-data",
            workspace="ai2/dolma2",
            nodes=1,
            gpus=8,
            variants=2,
            max_tokens=1000000,
            sequence_length=2048,
            seed=42,
            cluster="ai2/saturn-cirrascale",
            tokenizer="gpt_neox",
            proxy_model_id="olmo_30m",
            sources=[
                SourceConfig(name="wiki", paths=["s3://bucket/wiki/*.npy"]),
            ],
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
