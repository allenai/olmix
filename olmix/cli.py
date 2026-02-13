"""Unified CLI for olmix experiments."""

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import click
import yaml
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

logger = logging.getLogger(__name__)


def _get_output_path_from_config(config_path: Path, group_uuid: str, timestamp: str | None = None) -> Path:
    """Derive output path from config path, mirroring the config hierarchy.

    Example:
        configs/experiments/quality_thresholds/heavy_code/top10pct.yaml
        -> output/mixes/quality_thresholds/heavy_code/top10pct/20260204_143025-<uuid>.json
    """
    config_path = Path(config_path).resolve()

    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    # Find the experiments/ directory in the path
    parts = config_path.parts
    try:
        experiments_idx = parts.index("experiments")
    except ValueError:
        # Fallback: use just the filename stem if not in experiments/
        return Path(f"output/mixes/{config_path.stem}/{timestamp}-{group_uuid}.json")

    # Get the relative path after experiments/
    # Config name becomes a directory, timestamp-uuid becomes the filename
    relative_parts = parts[experiments_idx + 1 :]
    relative_path = Path(*relative_parts)
    output_name = f"{timestamp}-{group_uuid}.json"
    # Include the config stem as a subdirectory
    output_dir = relative_path.parent / relative_path.stem

    return Path("output/mixes") / output_dir / output_name


def _get_git_info() -> dict[str, str]:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:8]
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return {"git_commit": commit, "git_branch": branch}
    except Exception:
        return {"git_commit": "unknown", "git_branch": "unknown"}


def _save_launch_metadata(
    config_path: Path,
    group_uuid: str,
    beaker_user: str,
    mixes: list[dict],
    results: list,
    priors: dict | None = None,
) -> None:
    """Save launch metadata to the mix file for reproducibility tracking."""
    output_path = _get_output_path_from_config(config_path, group_uuid)

    # Load the config contents for reproducibility
    with open(config_path) as f:
        config_contents = yaml.safe_load(f)

    # Build experiment info from launch results
    experiments = []
    for i, result in enumerate(results):
        exp = result.experiment
        experiments.append(
            {
                "variant": i,
                "beaker_id": exp.id,
                "beaker_name": exp.name,
                "beaker_url": f"https://beaker.org/ex/{exp.id}",
            }
        )

    # Build full metadata
    metadata = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "config_path": str(config_path),
            "beaker_user": beaker_user,
            "group_id": group_uuid,
            "wandb_url": f"https://wandb.ai/ai2-llm/olmix?group={group_uuid}",
            **_get_git_info(),
        },
        "config": config_contents,
        "experiments": experiments,
        "mixes": mixes,
    }
    if priors is not None:
        metadata["priors"] = priors

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Launch metadata saved to {output_path}")


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

    mk_mixes(config, output, use_cache=not no_cache)  # priors returned but not needed here


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

    beaker_user = Beaker.from_env().user_name.upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info("Generating experiment group from the following config...")
    logger.info(experiment_config)

    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    launch_configs = None
    mixes = None
    priors = None

    if mixture_file:
        with open(mixture_file) as f:
            predefined_mixes = json.load(f)

        mixes = predefined_mixes["mixes"]
        logger.info(predefined_mixes)
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_configs = mk_launch_configs(
                    group=mk_experiment_group(
                        config=experiment_config,
                        mixes=mixes,
                        group_uuid=group_uuid,
                    ),
                    beaker_user=beaker_user,
                )
                spinner.ok("Done")
    else:
        mixes, priors = mk_mixes(config, use_cache=(no_cache is False), group_uuid=group_uuid, save=False)
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
                logger.info("Dry run mode enabled. Running dry-run for each experiment...")
                torchrun = experiment_config.infra.gpus > 1
                for lc in launch_configs:
                    logger.info(f"Dry run for {lc.name}:")
                    lc.dry_run(torchrun=torchrun)

                # Save launch metadata (without beaker experiment IDs)
                _save_launch_metadata(
                    config_path=config,
                    group_uuid=group_uuid,
                    beaker_user=beaker_user,
                    mixes=mixes,
                    results=[],
                    priors=priors,
                )
                return

            results = []
            torchrun = experiment_config.infra.gpus > 1
            for lc in tqdm(launch_configs, desc="Launching experiments"):
                results.append(lc.launch(torchrun=torchrun))

            spinner.ok("Done")

            # Save launch metadata for reproducibility tracking
            _save_launch_metadata(
                config_path=config,
                group_uuid=group_uuid,
                beaker_user=beaker_user,
                mixes=mixes,
                results=results,
                priors=priors,
            )

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
    cluster = beaker.cluster.get(exp_config.infra.cluster)
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
    cluster = beaker.cluster.get(exp_config.infra.cluster)
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

    mixes, _priors = mk_mixes(config)
    experiment_group = mk_experiment_group(ExperimentConfig(**data), mixes, generate_uuid()[:8])

    for experiment in experiment_group.instances:
        logger.info(mk_instance_cmd(experiment, experiment_group.config, experiment_group.group_id, "preview"))


# ============================================================================
# Fit subcommand (from existing fit CLI)
# ============================================================================

try:
    from olmix.fit.cli import fit as fit_command

    cli.add_command(fit_command, name="fit")
except ImportError:
    # fit module not available (missing optional dependencies)
    pass


if __name__ == "__main__":
    cli()
