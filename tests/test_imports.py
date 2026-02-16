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
            ExperimentGroup,
            ExperimentInstance,
            GenerationConfig,
            LaunchConfig,
            MixEntry,
            Priority,
            SourceConfig,
            SourceInstance,
            TopicConfig,
            TrainType,
        )

        assert ExperimentGroup is not None
        assert ExperimentInstance is not None
        assert GenerationConfig is not None
        assert LaunchConfig is not None
        assert MixEntry is not None
        assert Priority is not None
        assert SourceConfig is not None
        assert SourceInstance is not None
        assert TopicConfig is not None
        assert TrainType is not None

    def test_fit_module_imports(self):
        """Test fit module imports."""
        from olmix.fit.config import InLoopEvalConfig, OfflineEvalConfig
        from olmix.fit.law import ScalingLaw
        from olmix.fit.utils import (
            LightGBMRegressor,
            LogLinearRegressor,
            Regressor,
        )

        assert InLoopEvalConfig is not None
        assert OfflineEvalConfig is not None
        assert ScalingLaw is not None
        assert LightGBMRegressor is not None
        assert LogLinearRegressor is not None
        assert Regressor is not None

    def test_generate_module_imports(self):
        """Test generate module imports."""
        from olmix.generate import (
            calculate_priors,
            mk_mixes,
            mk_mixtures,
            prettify_mixes,
        )

        assert calculate_priors is not None
        assert mk_mixes is not None
        assert mk_mixtures is not None
        assert prettify_mixes is not None

    def test_launch_module_imports(self):
        """Test launch module imports."""
        from olmix.launch.utils import mk_source_instances

        assert mk_source_instances is not None

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
                mk_instance_cmd,
                mk_launch_configs,
            )

            assert get_beaker_username is not None
            assert mk_experiment_group is not None
            assert mk_instance_cmd is not None
            assert mk_launch_configs is not None
        except ImportError:
            pytest.skip("beaker-py not installed")

    def test_model_module_imports(self):
        """Test model module imports."""
        from olmix.model.aliases import ModelTrainConfig

        assert ModelTrainConfig is not None

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
