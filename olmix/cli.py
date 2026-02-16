"""Unified CLI for olmix experiments."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from olmix.aliases import LaunchConfig

import click
import yaml
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

logger = logging.getLogger(__name__)


def _get_git_info() -> dict[str, str]:
    """Get current git commit and branch info."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()[:8]
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
        return {"git_commit": commit, "git_branch": branch}
    except Exception:
        return {"git_commit": "unknown", "git_branch": "unknown"}


def _save_launch_metadata(
    configs: list[LaunchConfig],
    group_uuid: str,
    beaker_user: str,
    results: list,
) -> None:
    """Save launch metadata to the mix file for reproducibility tracking."""
    base = configs[0]
    # Use experiment name as a proxy path for output
    output_path = Path(f"output/mixes/{base.name}/{group_uuid}.json")

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
            "beaker_user": beaker_user,
            "group_id": group_uuid,
            "wandb_url": f"https://wandb.ai/ai2-llm/olmix?group={group_uuid}",
            **_get_git_info(),
        },
        "config": base.model_dump(mode="json"),
        "experiments": experiments,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Launch metadata saved to {output_path}")


def _load_launch_configs(variants_path: str) -> list[LaunchConfig]:
    """Load LaunchConfig YAML files from a file or directory."""
    from olmix.aliases import LaunchConfig

    p = Path(variants_path)
    if p.is_file():
        return [LaunchConfig.from_yaml(p)]

    if not p.is_dir():
        raise click.BadParameter(f"Variants path is not a file or directory: {variants_path}")

    variant_files = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
    if not variant_files:
        raise click.BadParameter(f"No YAML files found in variants directory: {variants_path}")

    configs = []
    for vf in variant_files:
        configs.append(LaunchConfig.from_yaml(vf))

    return configs


@click.group()
def cli():
    """OLMix - Data mixture optimization for OLMo training."""
    prepare_cli_environment()


# ============================================================================
# Generate command (top-level)
# ============================================================================


@cli.command("generate")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the generation configuration file.",
)
@click.option(
    "--base",
    type=click.Path(exists=True),
    required=True,
    help="Path to the base launch configuration file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory for self-contained launch config YAML files.",
)
def generate(config: str, base: str, output: str):
    """Generate self-contained launch configs from a generation config and base launch config."""
    from olmix.aliases import GenerationConfig, LaunchConfig
    from olmix.generate.utils import mk_mixes

    gen_config = GenerationConfig.from_yaml(config)
    base_config = LaunchConfig.from_yaml(base)
    group_uuid = generate_uuid()[:8]

    mixes = mk_mixes(gen_config)

    # Write each mix as a self-contained LaunchConfig YAML file
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    name = gen_config.name or Path(config).stem

    for idx, mix in enumerate(mixes):
        variant_name = f"{name}-{group_uuid}-{idx:04}"
        variant_config = base_config.model_copy(update={"name": variant_name, "mix": mix, "group_id": group_uuid})
        variant_file = output_path / f"{variant_name}.yaml"
        with open(variant_file, "w") as f:
            yaml.dump(
                variant_config.model_dump(mode="json", exclude_none=True, exclude_defaults=True),
                f,
                default_flow_style=False,
                sort_keys=False,
            )

    click.echo(f"Generated {len(mixes)} variant(s) in {output_path}/")
    for vf in sorted(output_path.glob("*.yaml")):
        click.echo(f"  {vf.name}")


# ============================================================================
# Launch subcommands
# ============================================================================


@cli.group()
def launch():
    """Commands for launching experiments to Beaker."""
    pass


@launch.command("run")
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    required=True,
    help="Path to a launch config file or directory of launch config files.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
def launch_run(variants: str, dry_run: bool):
    """Launch an experiment group to Beaker."""
    from beaker import Beaker

    from olmix.launch.beaker import launch_noninteractive, mk_experiment_group, mk_launch_configs

    configs = _load_launch_configs(variants)
    group_uuid = configs[0].group_id or generate_uuid()[:8]

    beaker_user = Beaker.from_env().user_name.upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info("Generating experiment group from the following config...")
    logger.info(configs[0])
    logger.info(f"With {len(configs)} variant(s) from {variants}")

    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Building experiment group...", color="yellow") as spinner:
        beaker_launch_configs = mk_launch_configs(
            group=mk_experiment_group(
                configs=configs,
                group_uuid=group_uuid,
            ),
            beaker_user=beaker_user,
        )
        spinner.ok("Done")

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Running dry-run for each experiment...")
                torchrun = configs[0].infra.gpus > 1
                for blc in beaker_launch_configs:
                    logger.info(f"Dry run for {blc.name}:")
                    blc.dry_run(torchrun=torchrun)

                # Save launch metadata (without beaker experiment IDs)
                _save_launch_metadata(
                    configs=configs,
                    group_uuid=group_uuid,
                    beaker_user=beaker_user,
                    results=[],
                )
                return

            results = []
            torchrun = configs[0].infra.gpus > 1
            for blc in tqdm(beaker_launch_configs, desc="Launching experiments"):
                results.append(launch_noninteractive(blc, torchrun=torchrun))

            spinner.ok("Done")

            # Save launch metadata for reproducibility tracking
            _save_launch_metadata(
                configs=configs,
                group_uuid=group_uuid,
                beaker_user=beaker_user,
                results=results,
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
    required=False,
    default=None,
    help="The group ID of the experiment group. Auto-extracted from config if not provided.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to any generated launch config file (for cluster info and group ID).",
)
def launch_status(config: str, group_id: str | None):
    """Get the status of a launched experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import LaunchConfig

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    launch_cfg = LaunchConfig.from_yaml(config)

    gid = group_id or launch_cfg.group_id
    if gid is None:
        raise click.BadParameter("No --group-id provided and config has no group_id field.")

    cluster = beaker.cluster.get(launch_cfg.infra.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{launch_cfg.name}-{gid}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


@launch.command("cancel")
@click.option(
    "-g",
    "--group-id",
    required=False,
    default=None,
    help="The group ID of the experiment group to cancel. Auto-extracted from config if not provided.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to any generated launch config file (for cluster info and group ID).",
)
def launch_cancel(config: str, group_id: str | None):
    """Cancel all running jobs for an experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import LaunchConfig

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    launch_cfg = LaunchConfig.from_yaml(config)

    gid = group_id or launch_cfg.group_id
    if gid is None:
        raise click.BadParameter("No --group-id provided and config has no group_id field.")

    cluster = beaker.cluster.get(launch_cfg.infra.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{launch_cfg.name}-{gid}")
    ]

    if len(jobs) == 0:
        logger.info(f"No jobs found for group {gid}")
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
    "-v",
    "--variants",
    type=click.Path(exists=True),
    required=True,
    help="Path to a launch config file or directory of launch config files.",
)
def launch_preview(variants: str):
    """Preview training commands without launching."""
    from olmix.launch.beaker import mk_experiment_group, mk_instance_cmd

    configs = _load_launch_configs(variants)
    group_uuid = configs[0].group_id or generate_uuid()[:8]

    experiment_group = mk_experiment_group(configs, group_uuid)

    for experiment in experiment_group.instances:
        logger.info(mk_instance_cmd(experiment, experiment_group.config, experiment_group.group_id, "preview"))


# ============================================================================
# Priors subcommands
# ============================================================================


@cli.group()
def priors():
    """Commands for computing and managing data priors."""
    pass


@priors.command("compute")
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to a configuration file with data.sources.",
)
@click.option(
    "--no-cache",
    "-n",
    is_flag=True,
    default=False,
    help="Do not use cached token counts.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output file path. Defaults to stdout.",
)
def priors_compute(config: str, no_cache: bool, output: str | None):
    """Compute token counts for a config by scanning data sources."""
    from olmix.aliases import DataConfig
    from olmix.generate.synthesize_mixture import calculate_priors

    with open(config) as f:
        data = yaml.safe_load(f)

    # Extract data section — works with any config type that has a data field
    data_section = data.get("data", data)
    data_config = DataConfig(**data_section)

    _, _, token_counts = calculate_priors(data_config.sources, data_config.dtype, use_cache=not no_cache)

    result = yaml.dump({"priors": {"token_counts": token_counts}}, default_flow_style=False, sort_keys=True)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(result)
        logger.info(f"Priors written to {output}")
    else:
        click.echo(result)


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
