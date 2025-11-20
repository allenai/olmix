import concurrent.futures
import json
import logging
from pathlib import Path
from typing import Optional

import click
import yaml
from beaker import Beaker
from beaker.services.job import JobClient
from olmo_core.utils import generate_uuid, prepare_cli_environment
from tqdm import tqdm
from yaspin import yaspin

from regmixer.aliases import ExperimentConfig, LaunchGroup
from regmixer.model.transformer import TransformerConfigBuilder
from regmixer.utils import (
    config_from_path,
    mk_experiment_group,
    mk_instance_cmd,
    mk_launch_configs,
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
    "-m",
    "--mixture-file",
    help="(Optional) Relative path to a mixture configuration file.",
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
def launch(config: Path, mixture_file: Optional[Path], dry_run: bool, no_cache: bool):
    """Launch an experiment."""

    with open(config, "r") as f:
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

    launch_group = None

    if mixture_file:
        with open(mixture_file, "r") as f:
            predefined_mixes = json.load(f)

        logger.info(predefined_mixes)
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_group = LaunchGroup(
                    instances=mk_launch_configs(
                        group=mk_experiment_group(
                            config=experiment_config,
                            mixes=predefined_mixes["mixes"],
                            group_uuid=group_uuid,
                        ),
                        beaker_user=beaker_user,
                    )
                )
                spinner.ok("✔")
    else:
        mixes = mk_mixes(config, group_uuid, use_cache=(no_cache == False))
        if click.confirm(f"Launch experiment {group_uuid} with this set of mixtures?", default=False):
            with yaspin(text="Building experiment group...", color="yellow") as spinner:
                launch_group = LaunchGroup(
                    instances=mk_launch_configs(
                        group=mk_experiment_group(
                            experiment_config, mixes=mixes, group_uuid=group_uuid
                        ),
                        beaker_user=beaker_user,
                    )
                )
                spinner.ok("✔")
        else:
            logger.info("Launch cancelled!")
            return

    if not launch_group:
        logger.info("Launch cancelled!")
        return

    with yaspin(text="Launching experiment group...", color="yellow") as spinner:
        try:
            if dry_run:
                logger.info("Dry run mode enabled. Printing experiment configurations...")
                for experiment in launch_group.instances:
                    logger.info(experiment.build_experiment_spec())
                return

            results = []            
            torchrun = True if experiment_config.gpus > 1 else False 
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(experiment.launch, torchrun=torchrun) for experiment in launch_group.instances
                ]

                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Launching experiments",
                ):
                    results.append(future.result())

            spinner.ok("✔")
            logger.info(results)
            logger.info(f"Experiment group '{group_uuid}' launched successfully!")
        except KeyboardInterrupt:
            logger.warning(
                "\nAborting experiment group launch! You may need to manually stop the launched experiments."
            )


def status_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = client.list(cluster=cluster)

    statuses = [
        {"status": job.status, "display_name": job.display_name}
        for job in jobs
        if job.display_name.startswith(f"{config.name}-{group_id}")
    ]
    statuses.sort(key=lambda x: x["display_name"])
    logger.info(statuses)


def stop_for_group(path: Path, group_id: str):
    beaker = Beaker.from_env()
    client = JobClient(beaker=beaker)
    config = config_from_path(path)
    cluster = beaker.cluster.get(config.cluster)
    jobs = [
        {"id": job.id, "display_name": job.display_name, "status": job.status}
        for job in client.list(cluster=cluster)
        if job.display_name.startswith(f"{config.name}-{group_id}")
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
def generate_mixes(config: Path, output: Optional[Path] = None):
    """Generate a set of mixtures based on a provided config"""
    mk_mixes(config, output)


@cli.command()
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def validate(config: Path):
    """Validate an experiment configuration."""
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    mixes = mk_mixes(config)
    experiment_group = mk_experiment_group(ExperimentConfig(**data), mixes, generate_uuid()[:8])
    beaker_user = "validate-no-op"

    for experiment in experiment_group.instances:
        logger.info(
            mk_instance_cmd(
                experiment, experiment_group.config, experiment_group.group_id, beaker_user
            )
        )
        transformer = TransformerConfigBuilder(
            cluster=experiment_group.config.cluster,
            beaker_user="validate-no-op",
            group_id="validate-no-op",
            run_name="validate-no-op",
            max_tokens=experiment_group.config.max_tokens,
            sources=experiment.sources,
            sequence_length=experiment_group.config.sequence_length,
            seed=experiment_group.config.seed,
            tokenizer=experiment_group.config.tokenizer,
            dtype=experiment_group.config.dtype,
            model_identifier=experiment_group.config.proxy_model_id,
            weka=experiment_group.config.weka,
            device_batch_size=experiment_group.config.device_batch_size,
        ).build()
        dataset = transformer.dataset.build()
        dataset.prepare()


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def status(config: Path, group_id: str):
    """Get the status of a launched experiment group."""

    status_for_group(config, group_id)


@cli.command()
@click.option(
    "-g",
    "--group-id",
    required=True,
    help="The group ID of the experiment group to stop.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    required=True,
    help="Relative path to the experiment configuration file.",
)
def cancel(config: Path, group_id: str):
    """Cancel all running jobs for an experiment group."""

    stop_for_group(config, group_id)


if __name__ == "__main__":
    cli()
