import logging

from olmix.aliases import (
    MixEntry,
    SourceConfig,
    SourceInstance,
)

logger = logging.getLogger(__name__)


def mk_source_instances(sources: list[SourceConfig], mix_map: dict[str, MixEntry]) -> list[SourceInstance]:
    """
    Create source instances from source configs and mixture weights.

    Args:
        sources: List of source configurations
        mix_map: Dictionary mapping source names to MixEntry objects.
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
                        if full_name not in mix_map or mix_map[full_name].weight == 0:
                            continue
                        instances.append(
                            SourceInstance(
                                name=full_name,
                                paths=quality.paths,
                                ratio=mix_map[full_name].weight,
                                repetition_factor=mix_map[full_name].repetition_factor,
                            )
                        )
                else:
                    # Handle topics without quality: source:topic
                    full_name = f"{source.name}:{topic.name}"
                    if full_name not in mix_map or mix_map[full_name].weight == 0:
                        continue
                    assert topic.paths is not None, f"Topic {full_name} has no paths defined"
                    instances.append(
                        SourceInstance(
                            name=full_name,
                            paths=topic.paths,
                            ratio=mix_map[full_name].weight,
                            repetition_factor=mix_map[full_name].repetition_factor,
                        )
                    )
        elif source.quality:
            # Handle source-level quality buckets: source:quality
            for quality in source.quality:
                full_name = f"{source.name}:{quality.name}"
                if full_name not in mix_map or mix_map[full_name].weight == 0:
                    continue
                instances.append(
                    SourceInstance(
                        name=full_name,
                        paths=quality.paths,
                        ratio=mix_map[full_name].weight,
                        repetition_factor=mix_map[full_name].repetition_factor,
                    )
                )
        else:
            # Handle simple sources: source
            if source.name not in mix_map or mix_map[source.name].weight == 0:
                continue
            assert source.paths is not None, f"Source {source.name} has no paths defined"
            instances.append(
                SourceInstance(
                    name=source.name,
                    paths=source.paths,
                    ratio=mix_map[source.name].weight,
                    repetition_factor=mix_map[source.name].repetition_factor,
                )
            )

    return instances
