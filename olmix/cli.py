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
    variants_dir: str,
    results: list,
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
            "variants_dir": variants_dir,
            "beaker_user": beaker_user,
            "group_id": group_uuid,
            "wandb_url": f"https://wandb.ai/ai2-llm/olmix?group={group_uuid}",
            **_get_git_info(),
        },
        "config": config_contents,
        "experiments": experiments,
    }

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Launch metadata saved to {output_path}")


def _load_variants(variants_dir: str):
    """Load all VariantConfig YAML files from a directory."""
    from olmix.aliases import VariantConfig

    variants_path = Path(variants_dir)
    if not variants_path.is_dir():
        raise click.BadParameter(f"Variants path is not a directory: {variants_dir}")

    variant_files = sorted(variants_path.glob("*.yaml")) + sorted(variants_path.glob("*.yml"))
    if not variant_files:
        raise click.BadParameter(f"No YAML files found in variants directory: {variants_dir}")

    variants = []
    for vf in variant_files:
        variants.append(VariantConfig.from_yaml(vf))

    return variants


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
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Output directory for variant YAML files.",
)
def generate(config: str, output: str):
    """Generate a set of mixture variants from a generation config."""
    from olmix.aliases import GenerationConfig, VariantConfig
    from olmix.generate.utils import mk_mixes

    gen_config = GenerationConfig.from_yaml(config)
    group_uuid = generate_uuid()[:8]

    mixes = mk_mixes(gen_config)

    # Write each mix as a VariantConfig YAML file
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    name = gen_config.name or Path(config).stem

    for idx, mix in enumerate(mixes):
        variant_name = f"{name}-{group_uuid}-{idx:04}"
        variant = VariantConfig(name=variant_name, mix=mix)
        variant_file = output_path / f"{variant_name}.yaml"
        with open(variant_file, "w") as f:
            yaml.dump(variant.model_dump(), f, default_flow_style=False, sort_keys=False)

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
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the launch configuration file.",
)
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing variant YAML files (output of olmix generate).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Print the experiment group configurations without launching.",
)
def launch_run(config: str, variants: str, dry_run: bool):
    """Launch an experiment group to Beaker."""
    from beaker import Beaker

    from olmix.aliases import LaunchConfig
    from olmix.launch.beaker import mk_experiment_group, mk_launch_configs

    launch_config = LaunchConfig.from_yaml(config)
    variant_configs = _load_variants(variants)
    group_uuid = generate_uuid()[:8]

    beaker_user = Beaker.from_env().user_name.upper()
    logger.info(f"Launching experiment group '{group_uuid}' as user '{beaker_user}'")

    logger.info("Generating experiment group from the following config...")
    logger.info(launch_config)
    logger.info(f"With {len(variant_configs)} variant(s) from {variants}")

    if not click.confirm("Proceed with this configuration?", default=False):
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Building experiment group...", color="yellow") as spinner:
        launch_configs = mk_launch_configs(
            group=mk_experiment_group(
                config=launch_config,
                variants=variant_configs,
                group_uuid=group_uuid,
            ),
            beaker_user=beaker_user,
        )
        spinner.ok("Done")

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Running dry-run for each experiment...")
                torchrun = launch_config.infra.gpus > 1
                for lc in launch_configs:
                    logger.info(f"Dry run for {lc.name}:")
                    lc.dry_run(torchrun=torchrun)

                # Save launch metadata (without beaker experiment IDs)
                _save_launch_metadata(
                    config_path=Path(config),
                    group_uuid=group_uuid,
                    beaker_user=beaker_user,
                    variants_dir=variants,
                    results=[],
                )
                return

            results = []
            torchrun = launch_config.infra.gpus > 1
            for lc in tqdm(launch_configs, desc="Launching experiments"):
                results.append(lc.launch(torchrun=torchrun))

            spinner.ok("Done")

            # Save launch metadata for reproducibility tracking
            _save_launch_metadata(
                config_path=Path(config),
                group_uuid=group_uuid,
                beaker_user=beaker_user,
                variants_dir=variants,
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
    required=True,
    help="The group ID of the experiment group.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Path to the launch configuration file.",
)
def launch_status(config: str, group_id: str):
    """Get the status of a launched experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import LaunchConfig

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    launch_cfg = LaunchConfig.from_yaml(config)
    cluster = beaker.cluster.get(launch_cfg.infra.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{launch_cfg.name}-{group_id}")
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
    help="Path to the launch configuration file.",
)
def launch_cancel(config: str, group_id: str):
    """Cancel all running jobs for an experiment group."""
    from beaker import Beaker
    from beaker.services.job import JobClient

    from olmix.aliases import LaunchConfig

    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    launch_cfg = LaunchConfig.from_yaml(config)
    cluster = beaker.cluster.get(launch_cfg.infra.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{launch_cfg.name}-{group_id}")
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
    help="Path to the launch configuration file.",
)
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing variant YAML files (output of olmix generate).",
)
def launch_preview(config: str, variants: str):
    """Preview training commands without launching."""
    from olmix.aliases import LaunchConfig
    from olmix.launch.beaker import mk_experiment_group, mk_instance_cmd

    launch_config = LaunchConfig.from_yaml(config)
    variant_configs = _load_variants(variants)
    group_uuid = generate_uuid()[:8]

    experiment_group = mk_experiment_group(launch_config, variant_configs, group_uuid)

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
