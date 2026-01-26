import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import yaml

from olmix.aliases import (
    ExperimentConfig,
    SourceConfig,
    SourceInstance,
)
from olmix.launch.synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)


def config_from_path(config: Path) -> ExperimentConfig:
    with open(config) as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)


def mk_source_instances(sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]) -> list[SourceInstance]:
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


def prettify_mixes(mixes: list[dict[str, tuple[float, float]]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(
    config_file: Path, output: Path | None = None, use_cache: bool = True, group_uuid: str | None = None
) -> list[dict[str, tuple[float, float]]]:
    import uuid

    with open(config_file) as f:
        data = yaml.safe_load(f)

    config = ExperimentConfig(**data)
    if group_uuid is None:
        group_uuid = str(uuid.uuid4())[:8]
    mixes = mk_mixtures(config, group_uuid, use_cache=use_cache)
    mix_string = prettify_mixes(mixes)

    if not output:
        output = Path(f"/tmp/olmix/{config.name}_{group_uuid}.json")

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
