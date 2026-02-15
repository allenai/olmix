import json
import logging
import os
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from olmix.aliases import (
    SourceConfig,
    SourceInstance,
    config_from_path,  # Re-export from aliases for backwards compatibility
)
from olmix.launch.synthesize_mixture import mk_mixtures

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


def mk_source_instances(sources: list[SourceConfig], mix_map: dict[str, tuple[float, float]]) -> list[SourceInstance]:
    """
    Create source instances from source configs and mixture weights.

    Args:
        sources: List of source configurations
        mix_map: Dictionary mapping source names to (weight, repetition_factor) tuples.
                 Keys can be: "source", "source:topic", or "source:topic:quality"

    Returns:
        List of SourceInstance objects with non-zero weights
    """
    instances = []

    for source in sources:
        if source.topics:
            for topic in source.topics:
                if topic.quality:
                    # Handle quality buckets: source:topic:quality
                    for quality in topic.quality:
                        full_name = f"{source.name}:{topic.name}:{quality.name}"
                        if full_name not in mix_map or mix_map[full_name][0] == 0:
                            continue
                        instances.append(
                            SourceInstance(
                                name=full_name,
                                paths=quality.paths,
                                ratio=mix_map[full_name][0],
                                repetition_factor=mix_map[full_name][1],
                            )
                        )
                else:
                    # Handle topics without quality: source:topic
                    full_name = f"{source.name}:{topic.name}"
                    if full_name not in mix_map or mix_map[full_name][0] == 0:
                        continue
                    assert topic.paths is not None, f"Topic {full_name} has no paths defined"
                    instances.append(
                        SourceInstance(
                            name=full_name,
                            paths=topic.paths,
                            ratio=mix_map[full_name][0],
                            repetition_factor=mix_map[full_name][1],
                        )
                    )
        elif source.quality:
            # Handle source-level quality buckets: source:quality
            for quality in source.quality:
                full_name = f"{source.name}:{quality.name}"
                if full_name not in mix_map or mix_map[full_name][0] == 0:
                    continue
                instances.append(
                    SourceInstance(
                        name=full_name,
                        paths=quality.paths,
                        ratio=mix_map[full_name][0],
                        repetition_factor=mix_map[full_name][1],
                    )
                )
        else:
            # Handle simple sources: source
            if source.name not in mix_map or mix_map[source.name][0] == 0:
                continue
            assert source.paths is not None, f"Source {source.name} has no paths defined"
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
    config_file: Path,
    output: Path | None = None,
    group_uuid: str | None = None,
    save: bool = True,
) -> list[dict[str, tuple[float, float]]]:
    import uuid

    config = config_from_path(config_file)
    if group_uuid is None:
        group_uuid = str(uuid.uuid4())[:8]
    mixes = mk_mixtures(config)
    mix_string = prettify_mixes(mixes)

    if save:
        if not output:
            output = _get_output_path_from_config(config_file, group_uuid)

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
