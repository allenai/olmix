"""Beaker launch configuration utilities for olmix experiments."""

import logging

from beaker import Beaker
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig, BeakerWekaBucket

from olmix.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)

logger = logging.getLogger(__name__)


def mk_source_instances(sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]) -> list[SourceInstance]:
    """
    Create source instances from source configs and mixture weights.

    Args:
        sources: List of source configurations
        mix_map: Dictionary mapping source names to (weight, repetition_factor) tuples

    Returns:
        List of SourceInstance objects with non-zero weights
    """
    instances = []

    for source in sources:
        if source.topics:
            for topic in source.topics:
                full_name = f"{source.name}:{topic.name}"
                if full_name not in mix_map or mix_map[full_name][0] == 0:
                    continue
                instances.append(
                    SourceInstance(
                        name=full_name,
                        paths=topic.paths,
                        ratio=mix_map[full_name][0],
                        repetition_factor=mix_map[full_name][1],
                    )
                )
        else:
            if source.name not in mix_map or mix_map[source.name][0] == 0:
                continue
            instances.append(
                SourceInstance(
                    name=source.name,
                    paths=source.paths,
                    ratio=mix_map[source.name][0],
                    repetition_factor=mix_map[source.name][1],
                )
            )

    return instances


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
            sources=mk_source_instances(config.sources, mix),
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
        "src/olmix/launch/train.py",
        "train",
        f"-n {instance.name}",
        f"-g {group_id}",
        f"-x {config.chinchilla_multiple}",
        f"-S {config.seed}",
        f"-c {config.cluster}",
        f"-u {beaker_user}",
        f"-d {config.dtype.value}",
        f"-T {config.tokenizer}",
        f"-m {config.proxy_model_id}",
        f"-w {config.weka}",
        f"-y {config.train_type.value}",
        f"-b {config.device_batch_size}",
    ]

    if config.global_batch_size:
        cmd_list.append(f"-B {config.global_batch_size}")

    if config.checkpoint_path:
        cmd_list.append(f"-C {config.checkpoint_path}")

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
    if group.config.weka:
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    setup_steps = [
        'git clone "$REPO_URL"',
        "conda shell.bash activate base",
        "cd olmix",
        'git checkout "$GIT_REF"',
        "git submodule update --init --recursive",
        "pip install -e '.[all]'",
        "pip install torch==2.7.0 torchaudio torchvision --index-url https://download.pytorch.org/whl/test/cu128",
        "pip freeze",
        # Move AWS credentials from env to relevant files
        "mkdir -p ~/.aws",
        "printenv AWS_CONFIG > ~/.aws/config",
        "printenv AWS_CREDENTIALS > ~/.aws/credentials",
    ]

    if group.config.wandb_debug:
        setup_steps.append("export WANDB_DEBUG=true")

    if group.config.gpus == 1:
        setup_steps += [
            # Single-process values
            "export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0",
            "export MASTER_ADDR=127.0.0.1",
            # Pick a free port for the training PG (env:// consumers)
            "export MASTER_PORT=$(python - <<'PY'\n"
            "import socket, contextlib\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0)); print(s.getsockname()[1])\n"
            "PY)",
            # Also set the *distributed default* port
            "export TORCH_DISTRIBUTED_DEFAULT_PORT=$(python - <<'PY'\n"
            "import socket, contextlib\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0)); print(s.getsockname()[1])\n"
            "PY)",
            # Keep the job on GPU 0 even if the box has more
            "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        ]

    return [
        BeakerLaunchConfig(
            name=f"{experiment.name}",
            description=group.config.description,
            task_name=experiment.name,
            cmd=mk_instance_cmd(experiment, group.config, group.group_id, beaker_user),
            clusters=[group.config.cluster],
            num_nodes=group.config.nodes,
            num_gpus=group.config.gpus,
            shared_filesystem=group.config.weka,
            allow_dirty=True,
            weka_buckets=weka_buckets,
            budget=group.config.budget or "ai2/oe-data",
            workspace=group.config.workspace,
            preemptible=group.config.preemptible,
            beaker_image="petew/olmo-core-tch270cu128-v2.1",
            priority=group.config.priority.value,
            env_secrets=[
                BeakerEnvSecret(name="BEAKER_TOKEN", secret=f"{beaker_user}_BEAKER_TOKEN"),
                BeakerEnvSecret(name="WANDB_API_KEY", secret=f"{beaker_user}_WANDB_API_KEY"),
                BeakerEnvSecret(name="AWS_CONFIG", secret=f"{beaker_user}_AWS_CONFIG"),
                BeakerEnvSecret(name="AWS_CREDENTIALS", secret=f"{beaker_user}_AWS_CREDENTIALS"),
                BeakerEnvSecret(name="R2_ENDPOINT_URL", secret="R2_ENDPOINT_URL"),
                BeakerEnvSecret(name="WEKA_ENDPOINT_URL", secret="WEKA_ENDPOINT_URL"),
                BeakerEnvSecret(name="GOOGLE_CLOUD_PROJECT", secret="GOOGLE_CLOUD_PROJECT"),
            ],
            setup_steps=setup_steps,
        )
        for experiment in group.instances
    ]


def get_beaker_username() -> str:
    """Get the current Beaker username."""
    beaker = Beaker.from_env()
    return beaker.account.whoami().name
