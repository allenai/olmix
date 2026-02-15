import json
import logging
from collections import defaultdict
from copy import deepcopy

from olmix.aliases import GenerationConfig
from olmix.generate.synthesize_mixture import mk_mixtures

logger = logging.getLogger(__name__)


def prettify_mixes(mixes: list[dict[str, tuple[float, float]]]):
    result = {"mixes": mixes}
    return json.dumps(result, indent=2)


def mk_mixes(config: GenerationConfig) -> list[dict[str, tuple[float, float]]]:
    """Generate mixture configurations from a GenerationConfig.

    Returns the raw mixes and logs a nested summary.
    """
    mixes = mk_mixtures(config)

    display_mixes = deepcopy(mixes)

    nested_mixes = []
    for mix in display_mixes:
        mix = {k: v for k, v in mix.items() if v[0] > 0}

        # Organize into source -> topic -> weight
        source_totals: dict[str, float] = defaultdict(float)
        source_topics: dict[str, dict[str, float]] = defaultdict(dict)

        for domain, (weight, _) in mix.items():
            if ":" in domain:
                source, topic = domain.split(":", 1)
                source_totals[source] += weight
                source_topics[source][topic] = weight
            else:
                source_totals[domain] += weight

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
