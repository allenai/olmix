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
        assert "OLMix" in result.output or "generate" in result.output.lower()

    def test_generate_help(self, runner):
        """Test generate subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output
        assert "--base" in result.output
        assert "--output" in result.output

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
        assert "--variants" in result.output
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

    def test_launch_preview_help(self, runner):
        """Test launch preview subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["launch", "preview", "--help"])
        assert result.exit_code == 0
        assert "--variants" in result.output

    def test_priors_group_help(self, runner):
        """Test priors subcommand group shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["priors", "--help"])
        assert result.exit_code == 0
        assert "compute" in result.output.lower()

    def test_priors_compute_help(self, runner):
        """Test priors compute subcommand shows help."""
        from olmix.cli import cli

        result = runner.invoke(cli, ["priors", "compute", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output


class TestLoadLaunchConfigs:
    """Test _load_launch_configs accepts file or directory."""

    def test_load_single_file(self, tmp_path):
        """Test loading a single launch config file."""
        import yaml

        from olmix.cli import _load_launch_configs

        config_data = {
            "name": "test",
            "infra": {"budget": "test", "workspace": "test", "cluster": "test", "gpus": 1},
            "training": {"proxy_model_id": "olmo2_30m", "tokenizer": "dolma2", "chinchilla_multiple": 1.0, "seed": 42},
            "data": {"sources": [{"name": "wiki", "paths": ["test.npy"]}]},
            "eval": {"type": "inloop", "tasks": {"qa": {"arc": "eval/arc (BPB v2)"}}},
            "mix": {"wiki": {"weight": 1.0, "repetition_factor": 1.0}},
        }
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        configs = _load_launch_configs(str(config_file))
        assert len(configs) == 1
        assert configs[0].name == "test"

    def test_load_directory(self, tmp_path):
        """Test loading configs from a directory."""
        import yaml

        from olmix.cli import _load_launch_configs

        for i in range(3):
            config_data = {
                "name": f"test-{i}",
                "infra": {"budget": "test", "workspace": "test", "cluster": "test", "gpus": 1},
                "training": {
                    "proxy_model_id": "olmo2_30m",
                    "tokenizer": "dolma2",
                    "chinchilla_multiple": 1.0,
                    "seed": 42,
                },
                "data": {"sources": [{"name": "wiki", "paths": ["test.npy"]}]},
                "eval": {"type": "inloop", "tasks": {"qa": {"arc": "eval/arc (BPB v2)"}}},
                "mix": {"wiki": {"weight": 1.0, "repetition_factor": 1.0}},
            }
            with open(tmp_path / f"config_{i}.yaml", "w") as f:
                yaml.dump(config_data, f)

        configs = _load_launch_configs(str(tmp_path))
        assert len(configs) == 3


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
