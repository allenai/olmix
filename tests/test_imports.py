"""Test that all modules import correctly."""

import pytest


class TestPackageImports:
    """Test that the olmix package imports without errors."""

    def test_package_imports(self):
        """Test basic package import."""
        import olmix

        assert olmix.__version__ is not None

    def test_aliases_imports(self):
        """Test aliases module imports."""
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

        # Verify classes exist
        assert ExperimentConfig is not None
        assert ExperimentGroup is not None
        assert ExperimentInstance is not None
        assert Priority is not None
        assert SourceConfig is not None
        assert SourceInstance is not None
        assert TopicConfig is not None
        assert TrainType is not None
        assert config_from_path is not None

    def test_fit_module_imports(self):
        """Test fit module imports."""
        from olmix.fit.constants import GroupedWandbMetrics, ObjectiveWeights, WandbMetrics
        from olmix.fit.law import ScalingLaw
        from olmix.fit.utils import (
            LightGBMRegressor,
            LinearRegressor,
            Regressor,
        )

        assert GroupedWandbMetrics is not None
        assert ObjectiveWeights is not None
        assert WandbMetrics is not None
        assert ScalingLaw is not None
        assert LightGBMRegressor is not None
        assert LinearRegressor is not None
        assert Regressor is not None

    def test_launch_module_imports(self):
        """Test launch module imports."""
        from olmix.launch.launch_utils import (
            config_from_path,
            mk_mixes,
            mk_source_instances,
            prettify_mixes,
        )

        assert config_from_path is not None
        assert mk_mixes is not None
        assert mk_source_instances is not None
        assert prettify_mixes is not None

    def test_launch_synthesize_imports(self):
        """Test synthesize_mixture module imports."""
        from olmix.launch.synthesize_mixture import (
            calculate_priors,
            mk_mixtures,
        )

        assert calculate_priors is not None
        assert mk_mixtures is not None

    @pytest.mark.skipif(
        True,  # Skip if beaker not installed
        reason="Beaker optional dependency not installed",
    )
    def test_beaker_imports(self):
        """Test beaker module imports (requires beaker-py)."""
        try:
            from olmix.launch.beaker import (
                get_beaker_username,
                mk_experiment_group,
                mk_experiments,
                mk_instance_cmd,
                mk_launch_configs,
            )

            assert get_beaker_username is not None
            assert mk_experiment_group is not None
            assert mk_experiments is not None
            assert mk_instance_cmd is not None
            assert mk_launch_configs is not None
        except ImportError:
            pytest.skip("beaker-py not installed")

    def test_model_module_imports(self):
        """Test model module imports."""
        from olmix.model.aliases import ModelTrainConfig
        from olmix.model.evaluators import (
            CodeTasks,
            DownstreamEvaluators,
            DownstreamEvaluatorsSmall,
        )

        assert ModelTrainConfig is not None
        assert CodeTasks is not None
        assert DownstreamEvaluators is not None
        assert DownstreamEvaluatorsSmall is not None

    def test_utils_module_imports(self):
        """Test utils module imports."""
        from olmix.utils.cloud import expand_cloud_globs

        assert expand_cloud_globs is not None

    def test_direct_olmo_core_usage(self):
        """Test that olmo-core can be used directly."""
        from olmo_core.data import TokenizerConfig
        from olmo_core.nn.transformer import TransformerConfig

        tokenizer = TokenizerConfig.dolma2()
        model = TransformerConfig.olmo2_30M(vocab_size=tokenizer.padded_vocab_size())

        assert model.d_model == 256
        assert tokenizer.vocab_size == 100278


class TestEnumValues:
    """Test enum values are accessible."""

    def test_priority_enum(self):
        """Test Priority enum values."""
        from olmix.aliases import Priority

        assert Priority.low.value == "low"
        assert Priority.normal.value == "normal"
        assert Priority.high.value == "high"
        assert Priority.urgent.value == "urgent"

    def test_train_type_enum(self):
        """Test TrainType enum values."""
        from olmix.aliases import TrainType

        assert TrainType.pretrain.value == "pretrain"
        assert TrainType.anneal.value == "anneal"
