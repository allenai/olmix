import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Tuple
from collections import defaultdict
import yaml

from aliases import (
    ExperimentConfig,
    SourceConfig,
    SourceInstance,
)
from synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)

def config_from_path(config: Path) -> ExperimentConfig:
    with open(config, "r") as f:
        data = yaml.safe_load(f)

    return ExperimentConfig(**data)

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
