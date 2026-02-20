import json
import logging
from collections import defaultdict
from copy import deepcopy

from olmix.aliases import GenerationConfig, MixEntry
from olmix.generate.synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)


def prettify_mixes(mixes: list[dict[str, MixEntry]]):
    result = {"mixes": [{k: v.model_dump() for k, v in mix.items()} for mix in mixes]}
    return json.dumps(result, indent=2)


def mk_mixes(config: GenerationConfig, group_uuid: str | None = None) -> list[dict[str, MixEntry]]:
    """Generate mixture configurations from a GenerationConfig.

    Returns the raw mixes and logs a nested summary.
    """
    mixes = mk_mixtures(config, group_uuid=group_uuid)

    display_mixes = deepcopy(mixes)

    nested_mixes = []
    for mix in display_mixes:
        mix = {k: v for k, v in mix.items() if v.weight > 0}

        # Organize into source -> topic -> weight
        source_totals: dict[str, float] = defaultdict(float)
        source_topics: dict[str, dict[str, float]] = defaultdict(dict)

        for domain, entry in mix.items():
            if ":" in domain:
                source, topic = domain.split(":", 1)
                source_totals[source] += entry.weight
                source_topics[source][topic] = entry.weight
            else:
                source_totals[domain] += entry.weight

        # Combine into final nested structure
        nested: dict[str, float | dict] = {}
        for source in source_totals:
            if source in source_topics:
                nested[source] = {"total": source_totals[source], "topics": source_topics[source]}
            else:
                nested[source] = source_totals[source]

        nested_mixes.append(nested)
    logger.info(nested_mixes)

    return mixes
