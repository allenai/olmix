"""Unified CLI for olmix experiments."""

import concurrent.futures
import json
import logging
from pathlib import Path

import click
import yaml
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """OLMix - Data mixture optimization for OLMo training."""
    prepare_cli_environment()


# ============================================================================
# Mix subcommands
# ============================================================================


@cli.group()
def mix():
    """Commands for generating and managing mixture configurations."""
    pass


@mix.command("generate")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path for the generated mixes.",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Do not cache sources for this experiment group.",
)
def generate_mixes(config: Path, output: Path | None = None, no_cache: bool = False):
    """Generate a set of mixtures based on a provided config."""
    from olmix.launch.launch_utils import mk_mixes

    mk_mixes(config, output, use_cache=not no_cache)


# ============================================================================
# Launch subcommands
# ============================================================================


@cli.group()
def launch():
    """Commands for launching experiments to Beaker."""
    pass


@launch.command("run")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
@click.option(
    "-m",
    "--mixture-file",
    help="(Optional) Path to a mixture configuration file.",
    type=click.Path(exists=True),
    required=False,
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Do not cache sources for this experiment group.",
)
def launch_run(config: Path, mixture_file: Path | None, dry_run: bool, no_cache: bool):
    """Launch an experiment group to Beaker."""
    from beaker import Beaker

    from olmix.aliases import ExperimentConfig
    from olmix.launch.beaker import mk_experiment_group, mk_launch_configs
    from olmix.launch.launch_utils import mk_mixes

    with open(config) as f:
        data = yaml.safe_load(f)

    experiment_config = ExperimentConfig(**data)
    group_uuid = generate_uuid()[:8]

    beaker_user = (Beaker.from_env().account.whoami().name).upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info("Generating experiment group from the following config...")
    logger.info(experiment_config)

    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    launch_configs = None

    if mixture_file:
        with open(mixture_file) as f:
            predefined_mixes = json.load(f)

        logger.info(predefined_mixes)
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_configs = mk_launch_configs(
                    group=mk_experiment_group(
                        config=experiment_config,
                        mixes=predefined_mixes["mixes"],
                        group_uuid=group_uuid,
                    ),
                    beaker_user=beaker_user,
                )
                spinner.ok("Done")
    else:
        mixes = mk_mixes(config, use_cache=(no_cache is False), group_uuid=group_uuid)
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_configs = mk_launch_configs(
                    group=mk_experiment_group(experiment_config, mixes=mixes, group_uuid=group_uuid),
                    beaker_user=beaker_user,
                )
                spinner.ok("Done")
        else:
            logger.info("Launch cancelled!")
            return

    if not launch_configs:
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Printing experiment configurations...")
                for lc in launch_configs:
                    logger.info(lc.build_experiment_spec())
                return

            results = []
            torchrun = experiment_config.gpus > 1
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(lc.launch, torchrun=torchrun) for lc in launch_configs]

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Launching experiments",
                ):
                    results.append(future.result())

            spinner.ok("Done")
            logger.info(results)
            logger.info(f"Experiment group '{group_uuid}' launched successfully!")
        except KeyboardInterrupt:
            logger.warning(
                "\nAborting experiment group launch! You may need to manually stop the launched experiments."
            )


@launch.command("status")
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
def launch_status(config: Path, group_id: str):
    """Get the status of a launched experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import config_from_path

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    exp_config = config_from_path(config)
    cluster = beaker.cluster.get(exp_config.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{exp_config.name}-{group_id}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


@launch.command("cancel")
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to cancel.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
def launch_cancel(config: Path, group_id: str):
    """Cancel all running jobs for an experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import config_from_path

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    exp_config = config_from_path(config)
    cluster = beaker.cluster.get(exp_config.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{exp_config.name}-{group_id}")
    ]

    if len(jobs) == 0:
        logger.info(f"No jobs found for group {group_id}")
        return

    jobs.sort(key=lambda x: x["display_name"])
    logger.info("Jobs to cancel:")
    logger.info(jobs)
    if click.confirm("Cancel these jobs?", default=False):
        for job in jobs:
            logger.info(f"Stopping job {job['display_name']}...")
            client.stop(job["id"])


@launch.command("preview")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the experiment configuration file.",
)
def launch_preview(config: Path):
    """Preview sampled mixtures and training commands without launching."""
    from olmix.aliases import ExperimentConfig
    from olmix.launch.beaker import mk_experiment_group, mk_instance_cmd
    from olmix.launch.launch_utils import mk_mixes

    with open(config) as f:
        data = yaml.safe_load(f)

    mixes = mk_mixes(config)
    experiment_group = mk_experiment_group(ExperimentConfig(**data), mixes, generate_uuid()[:8])

    for experiment in experiment_group.instances:
        logger.info(mk_instance_cmd(experiment, experiment_group.config, experiment_group.group_id, "preview"))


# ============================================================================
# Fit subcommand (from existing fit CLI)
# ============================================================================


@cli.command("fit")
@click.pass_context
def fit(ctx):
    """Run regression fitting and mixture optimization.

    This command delegates to the fit module CLI.
    """
    from olmix.fit.cli import cli as fit_cli

    ctx.invoke(fit_cli)


# Add the fit subcommands directly
def register_fit_commands():
    """Register fit commands from the fit module."""
    try:
        from olmix.fit.cli import cli as fit_cli

        # Get all commands from fit_cli and add them under 'fit' group
        for name, cmd in fit_cli.commands.items():
            cli.add_command(cmd, name=f"fit-{name}" if name != "fit" else "fit")
    except ImportError:
        pass


# Try to register fit commands
try:
    from olmix.fit import cli as fit_module

    cli.add_command(fit_module.cli, name="fit")
except (ImportError, AttributeError):
    # fit module may have different structure, just add as subgroup
    pass


if __name__ == "__main__":
    cli()
