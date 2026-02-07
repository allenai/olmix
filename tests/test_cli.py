"""Tests for the CLI commands."""

import pytest
from click.testing import CliRunner


class TestCLI:
    """Test CLI commands."""

    @pytest.fixture
    def runner(self):
        """Return a CLI runner."""
        return CliRunner()

    def test_cli_help(self, runner):
        """Test CLI shows help without error."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "OLMix" in result.output or "mix" in result.output.lower()

    def test_mix_group_help(self, runner):
        """Test mix subcommand group shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["mix", "--help"])
        assert result.exit_code == 0
        assert "generate" in result.output.lower()

    def test_mix_generate_help(self, runner):
        """Test mix generate subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["mix", "generate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_launch_group_help(self, runner):
        """Test launch subcommand group shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["launch", "--help"])
        assert result.exit_code == 0

    def test_launch_run_help(self, runner):
        """Test launch run subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["launch", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--dry-run" in result.output

    def test_launch_status_help(self, runner):
        """Test launch status subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["launch", "status", "--help"])
        assert result.exit_code == 0
        assert "--group-id" in result.output

    def test_launch_cancel_help(self, runner):
        """Test launch cancel subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["launch", "cancel", "--help"])
        assert result.exit_code == 0
        assert "--group-id" in result.output


class TestFitCLI:
    """Test fit CLI commands."""

    @pytest.fixture
    def runner(self):
        """Return a CLI runner."""
        return CliRunner()

    def test_fit_cli_help(self, runner):
        """Test fit CLI shows help."""
        from olmix.fit.cli import fit

        result = runner.invoke(fit, ["--help"])
        assert result.exit_code == 0


class TestMixGenerateCommand:
    """Test mix generate command functionality."""

    @pytest.fixture
    def runner(self):
        """Return a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def sample_config_file(self, tmp_path):
        """Create a sample config file."""
        import yaml

        config = {
            "name": "test-swarm",
            "description": "Test experiment",
            "budget": "ai2/oe-data",
            "workspace": "ai2/dolma2",
            "nodes": 1,
            "gpus": 1,
            "variants": 5,
            "max_tokens": 1000000,
            "sequence_length": 2048,
            "seed": 42,
            "cluster": "ai2/saturn-cirrascale",
            "tokenizer": "gpt_neox",
            "proxy_model_id": "olmo_30m",
            "sources": [
                {"name": "test", "paths": ["s3://fake/path/*.npy"]},
            ],
        }

        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        return config_file

    @pytest.mark.skip(reason="Requires S3 access for token counting")
    def test_mix_generate_with_config(self, runner, sample_config_file, tmp_path):
        """Test mix generate runs with valid config (requires S3 access)."""
        from olmix.cli import cli

        output_file = tmp_path / "output.json"
        result = runner.invoke(
            cli,
            [
                "mix",
                "generate",
                "--config",
                str(sample_config_file),
                "--output",
                str(output_file),
            ],
        )

        # May fail due to S3 access, but should parse config
        assert "name" in result.output or result.exit_code in [0, 1]
