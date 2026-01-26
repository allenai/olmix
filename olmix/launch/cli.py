import logging
from pathlib import Path

import click
from olmo_core.utils import prepare_cli_environment

from olmix.launch.launch_utils import (
    mk_mixes,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path for the generated mixes (defaults to generated_mix.json if not specified)",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Do not cache sources for this experiment group.",
)
def generate_mixes(config: Path, output: Path | None = None, no_cache: bool = False):
    """Generate a set of mixtures based on a provided config"""
    mk_mixes(config, output, use_cache=not no_cache)


if __name__ == "__main__":
    cli()
