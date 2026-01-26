"""Tests for TransformerConfigBuilder Chinchilla calculations."""

import pytest

from olmix.model.transformer import (
    SEQUENCE_LENGTH,
    TOKENS_PER_PARAM,
    TransformerConfigBuilder,
)


class TestChinchillaConstants:
    """Test Chinchilla constants."""

    def test_sequence_length(self):
        """Test that sequence length is fixed at 8192."""
        assert SEQUENCE_LENGTH == 8192

    def test_tokens_per_param(self):
        """Test that tokens per param is 20."""
        assert TOKENS_PER_PARAM == 20


class TestTransformerConfigBuilderCalculations:
    """Test TransformerConfigBuilder scaling calculations."""

    @pytest.fixture
    def builder_30m(self):
        """Create a builder for 30M model."""
        from olmix.aliases import SourceInstance

        return TransformerConfigBuilder(
            run_name="test-run",
            sources=[
                SourceInstance(name="test", paths=["test.npy"], ratio=1.0),
            ],
            chinchilla_multiple=1.0,
            group_id="test-group",
            cluster="test-cluster",
            beaker_user="test-user",
            tokenizer="dolma2",
            dtype="uint32",
            model_identifier="olmo2_30m",
            weka=False,
            device_batch_size=4,
        )

    def test_batch_size_power_of_2(self, builder_30m):
        """Test that batch size is rounded to power of 2."""
        batch_size = builder_30m.get_batch_size(30_000_000)
        # Should be a power of 2
        assert batch_size > 0
        assert (batch_size & (batch_size - 1)) == 0

    def test_batch_size_scaling(self, builder_30m):
        """Test batch size scales with model size."""
        batch_30m = builder_30m.get_batch_size(30_000_000)
        batch_1b = builder_30m.get_batch_size(1_000_000_000)

        # Larger model should have larger batch size
        assert batch_1b > batch_30m

    def test_lr_scaling(self, builder_30m):
        """Test learning rate scales inversely with model size."""
        lr_30m = builder_30m.get_lr(30_000_000)
        lr_1b = builder_30m.get_lr(1_000_000_000)

        # Larger model should have smaller LR
        assert lr_1b < lr_30m
        # Both should be positive
        assert lr_30m > 0
        assert lr_1b > 0

    def test_lr_halving(self, builder_30m):
        """Test that LR is halved from base Chinchilla formula."""
        num_params = 30_000_000
        lr = builder_30m.get_lr(num_params)

        # Base formula: 0.0047 * (N / 108M)^(-1/3)
        base_lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)

        # Should be halved
        assert abs(lr - base_lr / 2) < 1e-10

    def test_warmup_tokens(self, builder_30m):
        """Test warmup tokens is capped appropriately."""
        num_params = 30_000_000
        warmup = builder_30m.get_warmup_tokens(num_params)

        # For 1xC, warmup = min(num_params, 5% of duration)
        # Duration = 20 * 30M * 1.0 = 600M
        # 5% of 600M = 30M = num_params, so warmup == num_params
        assert warmup == num_params

    def test_warmup_tokens_capped_for_small_chinchilla(self):
        """Test warmup is capped for small chinchilla_multiple."""
        from olmix.aliases import SourceInstance

        builder = TransformerConfigBuilder(
            run_name="test-run",
            sources=[
                SourceInstance(name="test", paths=["test.npy"], ratio=1.0),
            ],
            chinchilla_multiple=0.01,  # Very small
            group_id="test-group",
            cluster="test-cluster",
            beaker_user="test-user",
            tokenizer="dolma2",
            dtype="uint32",
            model_identifier="olmo2_30m",
            weka=False,
            device_batch_size=4,
        )

        num_params = 30_000_000
        warmup = builder.get_warmup_tokens(num_params)
        duration = builder.get_duration(num_params)

        # Warmup should be capped at 5% of duration
        assert warmup <= duration * 0.05
        assert warmup < num_params  # Should be less than default

    def test_duration_1x_chinchilla(self, builder_30m):
        """Test duration at 1xC."""
        num_params = 30_000_000
        duration = builder_30m.get_duration(num_params)

        # 1xC = 20 * params * 1.0 = 600M
        expected = TOKENS_PER_PARAM * num_params * 1.0
        assert duration == expected

    def test_duration_5x_chinchilla(self):
        """Test duration at 5xC."""
        from olmix.aliases import SourceInstance

        builder = TransformerConfigBuilder(
            run_name="test-run",
            sources=[
                SourceInstance(name="test", paths=["test.npy"], ratio=1.0),
            ],
            chinchilla_multiple=5.0,
            group_id="test-group",
            cluster="test-cluster",
            beaker_user="test-user",
            tokenizer="dolma2",
            dtype="uint32",
            model_identifier="olmo2_30m",
            weka=False,
            device_batch_size=4,
        )

        num_params = 30_000_000
        duration = builder.get_duration(num_params)

        # 5xC = 20 * 30M * 5 = 3B
        expected = TOKENS_PER_PARAM * num_params * 5.0
        assert duration == expected

    def test_next_power_of_2(self, builder_30m):
        """Test next_power_of_2 helper."""
        assert builder_30m.next_power_of_2(0) == 1
        assert builder_30m.next_power_of_2(1) == 1
        assert builder_30m.next_power_of_2(2) == 2
        assert builder_30m.next_power_of_2(3) == 4
        assert builder_30m.next_power_of_2(5) == 8
        assert builder_30m.next_power_of_2(100) == 128
        assert builder_30m.next_power_of_2(1000) == 1024

    def test_sequence_length_fixed(self, builder_30m):
        """Test that sequence length is fixed at 8192."""
        assert builder_30m.sequence_length == SEQUENCE_LENGTH


class TestWSWSScheduler:
    """Test WSDS scheduler configuration."""

    @pytest.fixture
    def builder(self):
        """Create a builder."""
        from olmix.aliases import SourceInstance

        return TransformerConfigBuilder(
            run_name="test-run",
            sources=[
                SourceInstance(name="test", paths=["test.npy"], ratio=1.0),
            ],
            chinchilla_multiple=2.0,
            group_id="test-group",
            cluster="test-cluster",
            beaker_user="test-user",
            tokenizer="dolma2",
            dtype="uint32",
            model_identifier="olmo2_30m",
            weka=False,
            device_batch_size=4,
        )

    def test_scheduler_has_warmup(self, builder):
        """Test that scheduler has warmup configured."""
        from olmo_core.optim import WSDS

        num_params = 30_000_000
        batch_size = builder.get_batch_size(num_params)
        scheduler = builder.get_scheduler(num_params, batch_size)

        assert isinstance(scheduler, WSDS)
        assert scheduler.warmup == num_params

    def test_scheduler_has_period_lengths(self, builder):
        """Test that scheduler has period lengths configured."""
        from olmo_core.optim import WSDS

        num_params = 30_000_000
        batch_size = builder.get_batch_size(num_params)
        scheduler = builder.get_scheduler(num_params, batch_size)

        assert isinstance(scheduler, WSDS)
        # For 2xC, should have periods at 0.5xC, 1xC, 2xC
        assert len(scheduler.period_lengths) >= 1

    def test_scheduler_decay_fraction(self, builder):
        """Test that scheduler has decay fraction set to 0.1."""
        from olmo_core.optim import WSDS

        num_params = 30_000_000
        batch_size = builder.get_batch_size(num_params)
        scheduler = builder.get_scheduler(num_params, batch_size)

        assert isinstance(scheduler, WSDS)
        assert scheduler.decay_fraction == 0.1

    def test_scheduler_small_chinchilla_multiple(self):
        """Test scheduler works with small chinchilla_multiple."""
        from olmo_core.optim import WSDS

        from olmix.aliases import SourceInstance

        builder = TransformerConfigBuilder(
            run_name="test-run",
            sources=[
                SourceInstance(name="test", paths=["test.npy"], ratio=1.0),
            ],
            chinchilla_multiple=0.01,  # Very small
            group_id="test-group",
            cluster="test-cluster",
            beaker_user="test-user",
            tokenizer="dolma2",
            dtype="uint32",
            model_identifier="olmo2_30m",
            weka=False,
            device_batch_size=4,
        )

        num_params = 30_000_000
        batch_size = builder.get_batch_size(num_params)
        scheduler = builder.get_scheduler(num_params, batch_size)

        assert isinstance(scheduler, WSDS)
        # Should have single period for small chinchilla_multiple
        assert len(scheduler.period_lengths) == 1
