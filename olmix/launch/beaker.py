"""Beaker launch configuration utilities for olmix experiments."""

import logging

from beaker import Beaker
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerEnvVar, BeakerLaunchConfig, BeakerWekaBucket

from olmix.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
)
from olmix.launch.launch_utils import mk_source_instances

logger = logging.getLogger(__name__)


def mk_experiments(
    config: ExperimentConfig, mixes: list[dict[str, tuple[float, float]]], group_uuid: str
) -> list[ExperimentInstance]:
    """
    Generate experiment instances from a config and mixture samples.

    Args:
        config: Experiment configuration
        mixes: List of mixture weight dictionaries
        group_uuid: Unique identifier for this experiment group

    Returns:
        List of ExperimentInstance objects
    """
    return [
        ExperimentInstance(
            name=f"{config.name}-{group_uuid}-{idx:04}",
            sources=mk_source_instances(config.data.sources, mix),
        )
        for idx, mix in enumerate(mixes)
    ]


def mk_experiment_group(
    config: ExperimentConfig, mixes: list[dict[str, tuple[float, float]]], group_uuid: str
) -> ExperimentGroup:
    """
    Build an experiment group from an experiment config.

    Args:
        config: Experiment configuration
        mixes: List of mixture weight dictionaries
        group_uuid: Unique identifier for this experiment group

    Returns:
        ExperimentGroup containing all experiment instances
    """
    return ExperimentGroup(
        config=config,
        group_id=group_uuid,
        instances=mk_experiments(config, mixes, group_uuid),
    )


def mk_instance_cmd(
    instance: ExperimentInstance, config: ExperimentConfig, group_id: str, beaker_user: str
) -> list[str]:
    """
    Build a command for launching an experiment instance.

    Args:
        instance: Experiment instance to launch
        config: Experiment configuration
        group_id: Unique identifier for the experiment group
        beaker_user: Beaker username for the job

    Returns:
        List of command-line arguments for the training script
    """
    sources = []

    for source in instance.sources:
        paths = [f'"{path}"' for path in source.paths]
        source_str = f'-s ("{source.name}",[{",".join(paths)}],{source.ratio},{source.repetition_factor})'
        sources.append(source_str)

    cmd_list = [
        "olmix/launch/train.py",
        "train",
        f"-n {instance.name}",
        f"-g {group_id}",
        f"-x {config.training.chinchilla_multiple}",
        f"-S {config.training.seed}",
        f"-c {config.infra.cluster}",
        f"-u {beaker_user}",
        f"-d {config.data.dtype.value}",
        f"-T {config.training.tokenizer}",
        f"-m {config.training.proxy_model_id}",
        f"-w {config.infra.weka}",
        f"-y {config.training.train_type.value}",
        f"-b {config.training.device_batch_size}",
    ]

    if config.training.global_batch_size:
        cmd_list.append(f"-B {config.training.global_batch_size}")

    if config.training.checkpoint_path:
        cmd_list.append(f"-C {config.training.checkpoint_path}")

    # In-loop evaluation settings
    if config.training.no_eval:
        cmd_list.append("--no-eval")
    else:
        cmd_list.append(f"-E {config.training.eval_interval}")
        for task in config.eval.task_ids:
            cmd_list.append(f"-e {task}")

    cmd_list.extend(sources)

    return cmd_list


def mk_launch_configs(group: ExperimentGroup, beaker_user: str) -> list[BeakerLaunchConfig]:
    """
    Build Beaker launch configs from an experiment group.

    Args:
        group: Experiment group containing all instances
        beaker_user: Beaker username for the job

    Returns:
        List of BeakerLaunchConfig objects ready for submission
    """
    weka_buckets: list[BeakerWekaBucket] = []
    if group.config.infra.weka:
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    # Build environment variables
    env_vars: list[BeakerEnvVar] = []

    if group.config.infra.wandb_debug:
        env_vars.append(BeakerEnvVar(name="WANDB_DEBUG", value="true"))

    if group.config.infra.gpus == 1:
        # Single-process environment variables
        env_vars.extend(
            [
                BeakerEnvVar(name="WORLD_SIZE", value="1"),
                BeakerEnvVar(name="RANK", value="0"),
                BeakerEnvVar(name="LOCAL_RANK", value="0"),
                BeakerEnvVar(name="MASTER_ADDR", value="127.0.0.1"),
                BeakerEnvVar(name="MASTER_PORT", value="29500"),
            ]
        )

    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config, group.group_id, beaker_user),
            clusters=[group.config.infra.cluster],
            num_nodes=group.config.infra.nodes,
            num_gpus=group.config.infra.gpus,
            shared_filesystem=group.config.infra.weka,
            allow_dirty=True,
            weka_buckets=weka_buckets,
            budget=group.config.infra.budget or "ai2/oe-data",
            workspace=group.config.infra.workspace,
            preemptible=group.config.infra.preemptible,
            beaker_image="petew/olmo-core-tch270cu128-v2.1",
            priority=group.config.infra.priority.value,
            env_vars=env_vars,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
                BeakerEnvSecret(name="GOOGLE_CLOUD_PROJECT", secret="GOOGLE_CLOUD_PROJECT"),
            ],
            aws_config_secret=f"{beaker_user}_AWS_CONFIG",
            aws_credentials_secret=f"{beaker_user}_AWS_CREDENTIALS",
        )
        for experiment in group.instances
    ]


def get_beaker_username() -> str:
    """Get the current Beaker username."""
    beaker = Beaker.from_env()  # type: ignore[attr-defined]
    return beaker.account.whoami().name
