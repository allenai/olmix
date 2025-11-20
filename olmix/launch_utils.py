import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
import yaml
from beaker import Beaker
from olmo_core.launch.beaker import BeakerEnvSecret, BeakerLaunchConfig, BeakerWekaBucket
from olmo_core.utils import generate_uuid

from regmixer.aliases import (
    ExperimentConfig,
    ExperimentGroup,
    ExperimentInstance,
    SourceConfig,
    SourceInstance,
)
from regmixer.synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


"""def mk_source_instances(
    sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]
) -> list[SourceInstance]:
    # Note: We filter out any sources that have a weight of 0 so we don' try to build
    # empty token indices downstream in olmo-core
    filtered_sources = [source for source in sources if mix_map[source.name][0] > 0]
    return [
        SourceInstance(
            name=source.name,
            paths=source.paths,
            ratio=mix_map[source.name][0],
            repetition_factor=mix_map[source.name][1],
        )
        for source in filtered_sources
    ]"""


def mk_source_instances(
    sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]
) -> list[SourceInstance]:
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
    """Generate source instances from a config."""
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
    """Build an experiment group from an experiment config."""

    return ExperimentGroup(
        config=config,
        group_id=group_uuid,
        instances=mk_experiments(config, mixes, group_uuid),
    )


def mk_instance_cmd(
    instance: ExperimentInstance, config: ExperimentConfig, group_id: str, beaker_user: str
) -> List[str]:
    """Build a command for launching an experiment instance."""

    sources = []

    for source in instance.sources:
        paths = [f'"{path}"' for path in source.paths]
        source_str = (
            f'-s ("{source.name}",[{",".join(paths)}],{source.ratio},{source.repetition_factor})'
        )
        sources.append(source_str)

    cmd_list = [
        "src/regmixer/train.py",
        "train",
        f"-n {instance.name}",
        f"-g {group_id}",
        f"-l {config.sequence_length}",
        f"-t {config.max_tokens}",
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
    """Build a beaker launch config from an experiment group."""

    weka_buckets: List[BeakerWekaBucket] = []
    if group.config.weka:
        weka_buckets.append(BeakerWekaBucket("oe-training-default", "/weka/oe-training-default"))

    setup_steps = [
        'git clone "$REPO_URL"',
        "conda shell.bash activate base",
        "cd regmixer",
        'git checkout "$GIT_REF"',
        "git submodule update --init --recursive",
        "pip install -e '.[all]'",
        # Temporary until they release a fix for 2.7.0
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
        """setup_steps += [
            "export LOCAL_RANK=0",
            "export RANK=0",
            "export WORLD_SIZE=1",
            # bind the rendez-vous server on this host
            "export MASTER_ADDR=127.0.0.1",
            # pick a free port at launch time
            "export MASTER_PORT=$(python - <<'PY'\n"
            "import socket, contextlib, sys\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0))              # ask the kernel for any free port\n"
            "    sys.stdout.write(str(s.getsockname()[1]))\n"
            "PY)",
            # 2) tell the single process how to rendez-vous
            "export MASTER_ADDR=127.0.0.1",
            # 3) standard single-process values
            "export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0",
            # 4) keep the job on GPU 0 even if the box has more
            "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        ]"""

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

            # Also set the *distributed default* port that some launchers default to (incl. torchrun RDZV when not specified)
            "export TORCH_DISTRIBUTED_DEFAULT_PORT=$(python - <<'PY'\n"
            "import socket, contextlib\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0)); print(s.getsockname()[1])\n"
            "PY)",

            # Keep the job on GPU 0 even if the box has more (unrelated to ports, but fine)
            "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
        ]

        """setup_steps += [
            # Standard single-process values
            "export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0",
            "export MASTER_ADDR=127.0.0.1",
            # Pick a free port for MASTER_PORT (used by env:// init)
            "export MASTER_PORT=$(python - <<'PY'\n"
            "import socket, contextlib\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0))\n"
            "    print(s.getsockname()[1])\n"
            "PY)",
            # Pick a separate free port for the rendezvous endpoint (elastic agent)
            "export RDZV_PORT=$(python - <<'PY'\n"
            "import socket, contextlib\n"
            "with contextlib.closing(socket.socket()) as s:\n"
            "    s.bind(('', 0))\n"
            "    print(s.getsockname()[1])\n"
            "PY)",
            # Keep the job on GPU 0 even if the box has more
            "export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}",
            # Provide args to append to torchrun
            'export TORCHRUN_ARGS="--standalone --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT}"',
        ]"""

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
            priority=group.config.priority,
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


def prettify_mixes(mixes: list[dict[str, Tuple[float, float]]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(
    config_file: Path, group_uuid: str, output: Optional[Path] = None, use_cache: bool = True
) -> list[dict[str, Tuple[float, float]]]:
    with open(config_file, "r") as f:
        data = yaml.safe_load(f)

    config = ExperimentConfig(**data)
    mixes = mk_mixtures(config, group_uuid, use_cache=use_cache)
    mix_string = prettify_mixes(mixes)
    breakpoint()

    if not output:
        output = Path(f"/tmp/regmixer/{config.name}_{group_uuid}.json")

    if output:
        os.makedirs(os.path.dirname(output), exist_ok=True)

        with open(output, "w") as f:
            f.write(mix_string)

        logger.info(f"Mixes saved to {output}:")

    from copy import deepcopy

    display_mixes = deepcopy(mixes)

    nested_mixes = []
    for mix in display_mixes:
        mix = {k: v for k, v in mix.items() if v[0] > 0}

        # Organize into source → topic → weight
        source_totals = defaultdict(float)
        source_topics = defaultdict(dict)

        for domain, (weight, _) in mix.items():
            if ":" in domain:
                source, topic = domain.split(":", 1)
                source_totals[source] += weight
                source_topics[source][topic] = weight
            else:
                source_totals[domain] += weight

        # Combine into final nested structure
        nested = {}
        for source in source_totals:
            if source in source_topics:
                nested[source] = {"total": source_totals[source], "topics": source_topics[source]}
            else:
                nested[source] = source_totals[source]

        nested_mixes.append(nested)
    logger.info(nested_mixes)

    return mixes
