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
