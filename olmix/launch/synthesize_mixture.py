import concurrent.futures
import logging
import os
import pathlib
import random
from collections import defaultdict
from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

import gcsfs
import numpy as np
import pandas as pd
import s3fs
from olmo_core.aliases import PathOrStr
from olmo_core.data.types import NumpyDatasetDType
from olmo_core.io import get_file_size, is_url, normalize_path
from olmo_core.utils import OLMoEnvironmentError
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


import hashlib
import json

from olmix.aliases import ExperimentConfig, SourceConfig


class ConfigDefaults:
    min_strength: float = 0.1
    max_strength: float = 5.0
    sample_multiplier: int = 10
    maximum_repetition: int = 5
    minimum_weight: float = 2e-3  # 0.002


def leaf_to_source(leaf_weights: np.ndarray, domains: list[str]) -> dict[str, float]:
    """
    Given a flat vector of weights at the leaf level (like 'dclm:math' or 'wikipedia'),
    return a dict mapping each source to its total weight.
    """
    assert len(leaf_weights) == len(domains), "Vector and domain lengths must match"
    source_dist = defaultdict(float)
    for weight, domain in zip(leaf_weights, domains):
        source = domain.split(":", 1)[0]  # supports both 'source' and 'source:topic'
        source_dist[source] += weight
    return dict(source_dist)


def clip_candidates_by_level(
    candidates,
    idx_to_level,
    domains,
    minimum_source_weight,
    minimum_topic_weight,
    fixed_topic_weights,
):
    assert len(candidates[0]) == len(idx_to_level), "Mismatch between weights and level types."

    # get the weight per source
    source_totals = leaf_to_source(candidates[0], domains)

    # we clip topics to be minimum_topic_weight * source_totals[source] (relative), and we clip sources to be minimum_source_weight (absolute)
    for idx, level in enumerate(idx_to_level):
        weight = candidates[0][idx]
        domain = domains[idx]
        source = domain.split(":", 1)[0] if ":" in domain else domain

        if source in fixed_topic_weights:
            # this source has fixed topic weights, so we don't clip it
            continue

        if level == "source" and weight < minimum_source_weight:
            candidates[0][idx] = 0.0
        elif level == "topic":
            threshold = minimum_topic_weight * source_totals[source]
            if weight < threshold:
                candidates[0][idx] = 0.0

    # normalize
    total = candidates.sum()
    if total > 0:
        candidates /= total
    else:
        raise ValueError("All weights were clipped to zero.")

    return candidates


def sample_has_required_sources(sample_vector, domains, nonzero_sources, minimum_source_weight, minimum_topic_weight):
    # Convert leaf-level weights to source and topic level weights
    source_weights = leaf_to_source(sample_vector, domains)
    topic_weights = defaultdict(list)  # source -> list of (idx, weight)

    for idx, weight in enumerate(sample_vector):
        domain = domains[idx]
        source = domain.split(":", 1)[0] if ":" in domain else domain
        topic_weights[source].append((idx, weight))

    # Only count topics that are above the minimum topic weight * source_weights[source] towards the source weight.
    clipped_source_sums = defaultdict(float)
    for source, topics in topic_weights.items():
        total = source_weights[source]
        threshold = minimum_topic_weight * total
        for _, weight in topics:
            if weight >= threshold:
                clipped_source_sums[source] += weight

    # Check that all nonzero sources pass threshold
    return all(clipped_source_sums[source] > minimum_source_weight for source in nonzero_sources)


def sample_has_required_sources_and_topics(
    sample_vector, domains, nonzero_domains, minimum_source_weight, minimum_topic_weight
):
    # Compute source-level totals
    source_weights = leaf_to_source(sample_vector, domains)

    # Group topic weights under their sources
    topic_weights = defaultdict(list)  # source -> list of (idx, weight)
    for idx, weight in enumerate(sample_vector):
        domain = domains[idx]
        source = domain.split(":", 1)[0] if ":" in domain else domain
        topic_weights[source].append((idx, weight))

    # Clip topic weights below the dynamic per-topic threshold
    clipped_source_sums = defaultdict(float)
    topic_above_threshold = set()
    for source, topics in topic_weights.items():
        total = source_weights[source]
        threshold = minimum_topic_weight * total
        for idx, weight in topics:
            if weight >= threshold:
                clipped_source_sums[source] += weight
                topic_above_threshold.add(domains[idx])

    # Check all nonzero constraints
    for required in nonzero_domains:
        if ":" in required:
            # Topic-level requirement
            if required not in topic_above_threshold:
                return False
        else:
            # Source-level requirement
            if clipped_source_sums[required] <= minimum_source_weight:
                return False

    return True


def generate_weights_dirichlet(
    sources: list[SourceConfig],  # flat
    leaf_dist: dict[str, float],
    minimum_source_weight: float,
    minimum_topic_weight: float,
    num_samples_out: int,
    source_temperature: float,
    topic_temperature: float,
    min_source_strength: float,
    max_source_strength: float,
    min_topic_strength: float,
    max_topic_strength: float,
    max_tokens: int,
    available_tokens: int,
    allow_repetition: bool,
    manual_prior: dict[str, float] | None,
    sample_multiplier: int | None,
    enable_bound: bool = True,
    nonzero_weight: list[str] | None = None,
    fixed_source_weights: dict[str, float] | None = None,
    existing_mix_file: str | None = None,
):
    """
    Generate weights for each domain group using a dirichlet distribution.
    The list of domains is always sorted to be in alphabetical order (i.e. alphabetical on sources, then on topics).
    """

    token_scale = available_tokens / max_tokens
    logger.info(f"Source token population is {token_scale:.2f}:1 target population.")

    collected_samples: list[tuple[np.ndarray, np.ndarray]] = []
    weight_bounds = None

    prior_dist = np.array([v for _, v in leaf_dist.items()])
    logger.info(f"Dimension of leaf-level distribution: {len(prior_dist)}")
    domains = [k for k, _ in leaf_dist.items()]
    source_names = [source.name for source in sources]
    idx_to_level = ["source" if name in source_names else "topic" for name in leaf_dist]

    if enable_bound:
        # weight bounds are at the leaf level and computed using the number of available tokens per source/topic.
        weight_bounds = [(0.0, min(prior_dist[idx] * token_scale, 1.0)) for idx in range(len(prior_dist))]
        # raise ValueError("WARNING: need to make sure keys are aligned here and with other places. In cookbook implementation, just removed sorted() everywhere, and seems to be fine?")
        grouped_bounds = {domain: weight_bounds[idx] for idx, domain in enumerate(domains)}
        logger.info("Weight bounds:")
        logger.info(grouped_bounds)

    # split prior distribution into source and topic distributions and tweak it according to the manual prior
    # Note: "topic" here refers to any leaf-level unit (could be topic, quality bucket, or topic:quality)
    topic_distributions = {}
    source_distribution = []
    for source_config in sorted(sources, key=lambda x: x.name):
        if source_config.topics:
            # this source has topics - collect all leaf-level units
            unit_weights = []
            for topic in sorted(source_config.topics, key=lambda x: x.name):
                if topic.quality:
                    # Each quality bucket is a separate unit
                    for q in sorted(topic.quality, key=lambda x: x.name):
                        unit_weights.append(leaf_dist[f"{source_config.name}:{topic.name}:{q.name}"])
                else:
                    # Topic is the unit
                    unit_weights.append(leaf_dist[f"{source_config.name}:{topic.name}"])
            weights = np.array(unit_weights)
            normalized_weights = weights / weights.sum()
            topic_distributions[source_config.name] = normalized_weights

            if manual_prior is not None and source_config.name in manual_prior:
                source_distribution.append(manual_prior[source_config.name])
            else:
                source_distribution.append(weights.sum())
        elif source_config.quality:
            # Source-level quality buckets (no topics)
            unit_weights = [
                leaf_dist[f"{source_config.name}:{q.name}"] for q in sorted(source_config.quality, key=lambda x: x.name)
            ]
            weights = np.array(unit_weights)
            normalized_weights = weights / weights.sum()
            topic_distributions[source_config.name] = normalized_weights

            if manual_prior is not None and source_config.name in manual_prior:
                source_distribution.append(manual_prior[source_config.name])
            else:
                source_distribution.append(weights.sum())
        else:
            # this source does not have topics or quality
            topic_distributions[source_config.name] = np.array([1.0])
            if manual_prior is not None and source_config.name in manual_prior:
                source_distribution.append(manual_prior[source_config.name])
            else:
                source_distribution.append(leaf_dist[source_config.name])

    source_distribution = np.array(source_distribution)
    source_distribution /= source_distribution.sum()

    logger.info(f"Source prior: {source_distribution}")
    logger.info(f"Topic prior: {topic_distributions}")
    if source_temperature < 1.0:
        source_prior = source_distribution**source_temperature
        source_prior = source_prior / np.sum(source_prior)
        logger.info(f"Source prior after temperature scaling: {source_prior}")
    else:
        source_prior = source_distribution

    if topic_temperature < 1.0:
        topic_priors = deepcopy(topic_distributions)
        for source, topic_prior in topic_priors.items():
            topic_prior = topic_prior**topic_temperature
            topic_prior = topic_prior / np.sum(topic_prior)
            topic_priors[source] = topic_prior
        logger.info(f"Topic priors after temperature scaling: {topic_priors}")
    else:
        topic_priors = deepcopy(topic_distributions)

    if not allow_repetition and weight_bounds:
        logger.info("Limiting candidates to within bounds, repetition is disabled...")

    fixed_topic_weights = {}
    for source in sources:
        if source.topics:
            if source.topics[0].weight is not None:
                # this source has topics with a fixed weight, so we use that weight as the prior
                # Need to expand to leaf-level units if topics have quality buckets
                unit_weights = []
                for topic in sorted(source.topics, key=lambda x: x.name):
                    if topic.quality:
                        # Check if quality buckets have explicit weights
                        has_quality_weights = any(q.weight is not None for q in topic.quality)
                        if has_quality_weights:
                            # Use explicit quality weights (normalize to sum to 1)
                            quality_weights = [
                                q.weight if q.weight is not None else 0.0
                                for q in sorted(topic.quality, key=lambda x: x.name)
                            ]
                            total_quality_weight = sum(quality_weights)
                            for qw in quality_weights:
                                unit_weights.append(
                                    topic.weight * (qw / total_quality_weight) if total_quality_weight > 0 else 0
                                )
                        else:
                            # Distribute topic weight proportionally across quality buckets by token count
                            quality_tokens = [
                                leaf_dist[f"{source.name}:{topic.name}:{q.name}"]
                                for q in sorted(topic.quality, key=lambda x: x.name)
                            ]
                            total_quality_tokens = sum(quality_tokens)
                            for qt in quality_tokens:
                                unit_weights.append(
                                    topic.weight * (qt / total_quality_tokens) if total_quality_tokens > 0 else 0
                                )
                    else:
                        unit_weights.append(topic.weight)
                conditional_weight = np.array([unit_weights])
                logger.info(f"Using fixed topic weights for source '{source.name}': {conditional_weight[0]}")
                fixed_topic_weights[source.name] = conditional_weight

    sample_multiplier = sample_multiplier if sample_multiplier else ConfigDefaults.sample_multiplier

    if fixed_source_weights is not None:
        fixed_source_weights = [
            fixed_source_weights[source_config.name] for source_config in sorted(sources, key=lambda x: x.name)
        ]

    if existing_mix_file is not None:
        ratios = pd.read_pickle(existing_mix_file)
        valid_existing_mixes = ratios[ratios[domains].sum(axis=1) == 1][
            domains
        ].values  # keep the rows that have probabilities that add up to 1 on the domains we're mixing on
    for _ in tqdm(range(num_samples_out * sample_multiplier)):
        candidates = []

        # first, generate source-level weights
        if min_source_strength == max_source_strength:
            if fixed_source_weights is not None:
                # if we have fixed source weights, we use those
                source_samples = np.array([fixed_source_weights])
            else:
                source_samples = np.random.dirichlet(source_prior * min_source_strength, 1)
        else:
            source_samples = []
            if fixed_source_weights is not None:
                for _ in range(15):
                    source_samples.append(np.array([fixed_source_weights]))
            else:
                min_source_strength_log = np.log10(min_source_strength)
                max_source_strength_log = np.log10(max_source_strength)
                for strength in np.logspace(min_source_strength_log, max_source_strength_log, 15):
                    samples_per_strength = np.random.dirichlet(source_prior * strength, 1)
                    source_samples.append(samples_per_strength)

        # then, generate topic-level weights
        topic_samples = defaultdict(list)
        for source, topic_prior in topic_priors.items():
            if source in fixed_topic_weights:
                # this source has fixed topic weights, so we use those
                conditional_weight = fixed_topic_weights[source]
                if min_topic_strength == max_topic_strength:
                    topic_samples[source].append(conditional_weight)
                else:
                    for _ in range(15):
                        topic_samples[source].append(conditional_weight)
                continue

            if min_topic_strength == max_topic_strength:
                topic_samples[source].append(np.random.dirichlet(topic_prior * min_topic_strength, 1))
            else:
                min_topic_strength_log = np.log10(min_topic_strength)
                max_topic_strength_log = np.log10(max_topic_strength)
                for strength in np.logspace(min_topic_strength_log, max_topic_strength_log, 15):
                    samples_per_strength = np.random.dirichlet(topic_prior * strength, 1)
                    topic_samples[source].append(samples_per_strength)

        # convert from source_samples and topic_samples back to a set of leaf-level samples
        candidates = []
        for i, source_sample in enumerate(source_samples):
            leaf_level_sample = {
                source: samples[i][0] * source_sample[0, j] for j, (source, samples) in enumerate(topic_samples.items())
            }
            flattened_sample = np.concatenate([arr for arr in list(leaf_level_sample.values())]).reshape(1, -1)
            candidates.append(flattened_sample)

        filtered_candidates = []

        # If we don't allow repetition, we need to filter out candidates that are outside the bounds
        if weight_bounds and not allow_repetition:
            filtered_candidates = [
                sample
                for sample in candidates
                if all(lower <= sample[0][idx] <= upper for idx, (lower, upper) in enumerate(weight_bounds))
            ]
        else:
            filtered_candidates = candidates

        if nonzero_weight:
            source_names = set(nonzero_weight)
            # Filter candidates
            filtered_candidates = [
                sample
                for sample in filtered_candidates
                if sample_has_required_sources_and_topics(
                    sample[0], domains, source_names, minimum_source_weight, minimum_topic_weight
                )
            ]

        if not filtered_candidates:
            logger.warning("No candidates left after filtering according to weight bounds and nonzero weights!")
            continue

        candidates = random.choice(filtered_candidates)

        if minimum_source_weight == minimum_topic_weight:
            candidates = np.where(candidates < minimum_source_weight, 0, candidates)
            candidates = candidates / np.sum(candidates).reshape(-1, 1)
            candidates = np.round(candidates / minimum_source_weight) * minimum_source_weight
            candidates = candidates / np.sum(candidates)
        else:
            candidates = clip_candidates_by_level(
                candidates, idx_to_level, domains, minimum_source_weight, minimum_topic_weight, fixed_topic_weights
            )

        if weight_bounds and not allow_repetition:
            # need to check for out-of-bounds candidates again, in case normalization caused bounds to be violated.
            if any(
                candidates[0][idx] < lower or candidates[0][idx] > upper
                for idx, (lower, upper) in enumerate(weight_bounds)
            ):
                continue

        selected: tuple[np.ndarray, np.ndarray] = (
            candidates[0],
            np.ones(candidates.shape[1]),
        )

        # if selected is too close to an existing mix from existing_mix_file, discard it
        if existing_mix_file is not None:
            dists = np.linalg.norm(valid_existing_mixes - candidates[0], axis=1)
            if np.any(dists < 0.01):
                logger.info(f"Candidate swarm run is too close to the mixes at {existing_mix_file}, rejecting.")
                continue

        reject = False
        if allow_repetition:
            for idx, _ in enumerate(domains):
                available_tokens = int(prior_dist[idx] * available_tokens)
                required_tokens = int(selected[0][idx] * max_tokens)

                repetition = np.ceil(required_tokens / available_tokens * 1000) / 1000 if available_tokens != 0 else 0

                if repetition > ConfigDefaults.maximum_repetition:
                    reject = True
                    break

                selected[1][idx] = max(1, repetition)

        if not reject:
            collected_samples.append(selected)

    if len(collected_samples) == 0:
        raise ValueError("No valid samples were generated, please check the configuration!")

    if len(collected_samples) > 10000:
        # when we have a lot of samples, regular sort_and_deduplicate is O(n^2) and takes too long
        deduped = sort_and_deduplicate_with_hash(collected_samples)
    else:
        deduped = sort_and_deduplicate(collected_samples)

    if len(collected_samples) < num_samples_out:
        raise ValueError(
            f"The number of collected samples '{len(collected_samples)}' is less than the required number of samples '{num_samples_out}'!"
        )

    selected_samples = random.sample(deduped, num_samples_out)
    selected_samples = np.stack(selected_samples, axis=0)

    logger.info("Number of nonzero domains per swarm run: ")
    print([len(np.where(selected_samples[i][0] != 0)[0]) for i in range(len(selected_samples))])

    all_diffs = []
    for i in range(len(selected_samples)):
        for j in range(i + 1, len(selected_samples)):
            diff = np.linalg.norm(selected_samples[i][0] - selected_samples[j][0])
            if diff < 0.01:
                logger.info(f"Sample {i} and Sample {j} are too close to each other!")
                logger.info(f"Sample {i}: {selected_samples[i][0]}")
                logger.info(f"Sample {j}: {selected_samples[j][0]}")
            all_diffs.append(diff)

    return selected_samples


def mk_mixtures(
    config: ExperimentConfig, group_uuid: str, use_cache: bool = True
) -> list[dict[str, tuple[float, float]]]:
    random.seed(config.seed)
    np.random.seed(config.seed)

    num_samples = config.variants
    sources = config.sources
    leaf_dist, available_tokens, leaf_tokens = calculate_priors(sources, config.dtype, use_cache=use_cache)
    logger.info(f"Total tokens for config: {available_tokens:,}")
    logger.info(f"Using seed: {config.seed}")

    logger.info("Source distribution:")
    logger.info(leaf_dist)
    logger.info("Source tokens:")

    leaf_tokens = {k: f"{v:,}" for k, v in leaf_tokens.items() if v > 0}
    logger.info(leaf_tokens)

    leaf_items = list(leaf_dist.items())
    prior_dist = [v for _, v in leaf_items]
    domains = [k for k, _ in leaf_items]

    # renormalize the prior distribution
    prior_dist = prior_dist / np.sum(prior_dist)

    # convert single-level sampling params into topic/source level
    minimum_source_weight = (
        config.minimum_source_weight
        if config.minimum_source_weight
        else config.minimum_weight
        if config.minimum_weight
        else ConfigDefaults.minimum_weight
    )
    minimum_topic_weight = (
        config.minimum_topic_weight
        if config.minimum_topic_weight
        else config.minimum_weight
        if config.minimum_weight
        else ConfigDefaults.minimum_weight
    )

    source_mix_temperature = config.source_mix_temperature if config.source_mix_temperature else config.mix_temperature
    topic_mix_temperature = config.topic_mix_temperature if config.topic_mix_temperature else config.mix_temperature

    min_source_strength = config.min_source_strength if config.min_source_strength else config.min_strength
    max_source_strength = config.max_source_strength if config.max_source_strength else config.max_strength

    min_topic_strength = config.min_topic_strength if config.min_topic_strength else config.min_strength
    max_topic_strength = config.max_topic_strength if config.max_topic_strength else config.max_strength

    mixtures = generate_weights_dirichlet(
        sources=sources,
        leaf_dist=leaf_dist,
        minimum_source_weight=minimum_source_weight,
        minimum_topic_weight=minimum_topic_weight,
        num_samples_out=num_samples,
        source_temperature=source_mix_temperature,
        topic_temperature=topic_mix_temperature,
        min_source_strength=min_source_strength,
        max_source_strength=max_source_strength,
        min_topic_strength=min_topic_strength,
        max_topic_strength=max_topic_strength,
        allow_repetition=config.allow_repetition,
        max_tokens=config.get_max_tokens(),
        available_tokens=available_tokens,
        enable_bound=True,
        nonzero_weight=config.nonzero_weight,
        fixed_source_weights=config.fixed_source_weights,
        manual_prior=config.manual_prior,
        sample_multiplier=config.sample_multiplier,
        existing_mix_file=config.existing_mix_file,
    )

    weight_maps = []
    for mix in mixtures:
        weight_map = {}
        for idx in range(len(domains)):
            weight_map[domains[idx]] = (mix[0][idx], mix[1][idx])

        weight_maps.append(weight_map)

    # Log weight ranges for topics
    for i in range(len(domains)):
        if ":" in domains[i]:
            weights = np.array([mix[0][i] for mix in mixtures])
            logger.info(f"Topic {domains[i]}, min: {weights.min()}, max: {weights.max()}")

    # Log weight ranges for sources
    source_to_indices = defaultdict(list)
    for i, domain in enumerate(domains):
        source = domain.split(":", 1)[0]
        source_to_indices[source].append(i)

    for source, indices in source_to_indices.items():
        source_weights = []
        for mix in mixtures:
            total = sum(mix[0][i] for i in indices)
            source_weights.append(total)
        source_weights = np.array(source_weights)
        logger.info(f"Source {source}, min: {source_weights.min()}, max: {source_weights.max()}")

    return weight_maps


def _bytes_to_tokens(num_bytes: int, dtype: NumpyDatasetDType) -> int:
    """
    Convert bytes to tokens based on the dtype.
    """
    npdtype = dtype.as_np_dtype()
    return num_bytes // npdtype(0).itemsize


def _count_tokens_for_file(path: PathOrStr, dtype: NumpyDatasetDType) -> int:
    return _bytes_to_tokens(get_file_size(path), dtype)


def count_tokens(paths: list[str], dtype: NumpyDatasetDType, fs) -> int:
    """Helper to count tokens across a list of paths using glob expansion."""
    total = 0
    for path in paths:
        matches = fs.glob(path)
        for match in matches:
            total += _count_tokens_for_file(f"s3://{match}", dtype)
    return total


def get_leaf_configs(source_config):
    """Return a list of (name, paths) tuples representing the leaf nodes.

    Handles three nesting levels:
    - source.paths -> (source, paths)
    - source.quality[].paths -> (source:quality_name, paths)
    - source.topics[].paths -> (source:topic, paths)
    - source.topics[].quality[].paths -> (source:topic:quality_name, paths)
    """
    results = []

    if source_config.topics:
        for topic in source_config.topics:
            if topic.quality:
                # Topics with quality buckets
                for q in topic.quality:
                    results.append((f"{source_config.name}:{topic.name}:{q.name}", q.paths))
            else:
                # Topics with direct paths
                results.append((f"{source_config.name}:{topic.name}", topic.paths))
    elif source_config.quality:
        # Source-level quality buckets (no topics)
        for q in source_config.quality:
            results.append((f"{source_config.name}:{q.name}", q.paths))
    else:
        # Direct paths on source
        results.append((source_config.name, source_config.paths))

    return results


def calculate_priors(
    source_configs: list[SourceConfig], dtype: NumpyDatasetDType, use_cache: bool
) -> tuple[dict[str, float], int, dict[str, int]]:
    config_hash = hashlib.md5(
        json.dumps(
            [(sc.name, sc.paths) for sc in source_configs],
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    pathlib.Path("cache/").mkdir(parents=True, exist_ok=True)
    cache_path = pathlib.Path(f"cache/priors_cache_{config_hash}.json")
    if use_cache:
        try:
            with open(cache_path) as f:
                logger.info(
                    f"Source distribution cache found, using cached values at {cache_path}! This can be disabled by setting use_cache=False."
                )
                obj = json.load(f)
                return (obj["relative_sizes"], obj["total_tokens"], obj["token_counts"])
        except FileNotFoundError:
            logger.info("No cache file found, calculating from source files...")

    token_counts = defaultdict(int)

    # Count tokens in each "leaf": the prior distribution is represented at the leaf level.
    filesystems = {}
    leaf_configs: list[SourceConfig] = []
    for sc in source_configs:
        leaf_configs.extend(
            SourceConfig(name=leaf_name, paths=leaf_paths) for leaf_name, leaf_paths in get_leaf_configs(sc)
        )
    source_configs = leaf_configs

    for source in source_configs:
        schemes = {urlparse(path).scheme for path in source.paths}

        # Check for mixed schemes within a source
        if len(schemes) > 1 and any(scheme for scheme in schemes):
            raise OLMoEnvironmentError(
                f"Mixed URL schemes in source '{source.name}': {schemes}. Each source must use a consistent scheme."
            )

        # Get the scheme (or None for local paths)
        scheme = next(iter(schemes)) if schemes and next(iter(schemes)) else "local"

        if scheme not in filesystems:
            filesystems[scheme] = get_filesystem_for_scheme(scheme)

    # Multithreaded token counting at leaf level
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        for source in source_configs:
            scheme = next(iter({urlparse(path).scheme for path in source.paths}), "local")
            fs = filesystems.get(scheme)

            globs = [path for path in source.paths if "*" in path]
            paths = [path for path in source.paths if path not in globs]
            source.paths = paths + expand_globs(fs, globs) if globs else paths

        futures = {
            executor.submit(_count_tokens_for_file, path, dtype): source
            for source in source_configs
            for path in source.paths
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Counting tokens (leaf level)",
        ):
            source_future = futures[future]
            try:
                result = future.result()
                token_counts[source_future.name] += result
            except Exception as e:
                logger.info(f"Error processing {source_future.name}: {e!s}")
                token_counts[source_future.name] = 0

    # Calculate relative sizes
    total_tokens = sum(token_counts.values())

    token_counts = dict(sorted(token_counts.items()))

    if total_tokens == 0:
        raise Exception("Error processing config, no tokens found for sources!")

    relative_sizes = {path: count / total_tokens for path, count in token_counts.items()}

    if use_cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {
                    "relative_sizes": relative_sizes,
                    "total_tokens": total_tokens,
                    "token_counts": token_counts,
                },
                f,
            )

    return (relative_sizes, total_tokens, token_counts)


def sort_and_deduplicate(
    samples: list[tuple[np.ndarray, np.ndarray]], threshold=1e-5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Remove identical configs to avoid duplicated training.
    """
    unique_samples = []
    for sample in tqdm(samples):
        is_duplicate = any(np.allclose(sample[0], unique_sample[0], atol=threshold) for unique_sample in unique_samples)
        if not is_duplicate:
            unique_samples.append(sample)

    logger.info(f"Filtered {len(samples) - len(unique_samples)} duplicate distributions from candidate pool...")
    return unique_samples


def sort_and_deduplicate_with_hash(
    samples: list[tuple[np.ndarray, np.ndarray]], threshold=1e-5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Remove near-identical configs efficiently to avoid duplicated training.
    """
    unique_samples = []
    seen_hashes = set()

    for sample in tqdm(samples):
        vec = sample[0]
        rounded = tuple(np.round(vec / threshold).astype(int))

        if rounded not in seen_hashes:
            seen_hashes.add(rounded)
            unique_samples.append(sample)

    logger.info(f"Filtered {len(samples) - len(unique_samples)} duplicate distributions from candidate pool...")
    return unique_samples


def get_filesystem_for_scheme(scheme: str):
    """
    Get the appropriate filesystem for a given URL scheme.

    Args:
        scheme: The URL scheme (e.g., 's3', 'gs', 'local', 'weka')

    Returns:
        The appropriate filesystem object for the scheme or None for local paths

    Raises:
        OLMoEnvironmentError: If the scheme is not supported or not configured correctly
        NotImplementedError: If the scheme is recognized but not currently supported
    """
    if scheme in ("s3", "weka"):
        client_kwargs = {}
        profile_name = os.environ.get("AWS_PROFILE", None)

        if scheme == "weka":
            profile_name = "WEKA"
            client_kwargs["endpoint_url"] = os.environ.get("WEKA_ENDPOINT_URL")

        return s3fs.S3FileSystem(client_kwargs={**client_kwargs}, profile=profile_name)

    elif scheme == "gs":
        try:
            gs_project = os.environ.get("GOOGLE_CLOUD_PROJECT", None)

            if not gs_project:
                raise OLMoEnvironmentError("GOOGLE_CLOUD_PROJECT environment variable is not set!")

            try:
                return gcsfs.GCSFileSystem(token="google_default")
            except Exception as e:
                logger.warning(
                    f"Failed to create GCS filesystem with default credentials: {e!s}. Retrying with metadata server..."
                )
                return gcsfs.GCSFileSystem()

        except Exception as e:
            raise OLMoEnvironmentError(
                f"Failed to create GCS filesystem: {e!s}. Ensure GOOGLE_APPLICATION_CREDENTIALS_JSON and GOOGLE_CLOUD_PROJECT are set correctly."
            )

    elif scheme in ("r2", "http", "https"):
        raise NotImplementedError(f"'{scheme}' scheme is not currently supported")

    elif scheme == "local":
        return None  # No remote filesystem needed for local paths

    else:
        raise OLMoEnvironmentError(f"Unsupported URL scheme: {scheme}")


def expand_globs(
    fs: s3fs.S3FileSystem | gcsfs.GCSFileSystem | None = s3fs.S3FileSystem(), sources: list[str] | None = None
) -> Any:
    if sources is None:
        sources = []
    results = []

    for source in sources:
        if is_url(source):
            results.extend(_expand_remote(source, fs))
        else:
            results.extend(_expand_local(source))

    # Filter the globs from the expanded list
    return [r for r in results if "*" not in r]


def _expand_local(pattern: str) -> list[str]:
    """
    Expand a local glob pattern.
    """
    from glob import glob

    logger.info(f"Expanding '{pattern}'...")
    matches = sorted(glob(pattern, recursive=True))

    if not matches:
        raise FileNotFoundError(pattern)

    return [normalize_path(match) for match in matches]


def _expand_remote(pattern: str, fs: s3fs.S3FileSystem | gcsfs.GCSFileSystem | None) -> list[str]:
    """
    Expand a remote glob pattern.
    """
    if not fs:
        fs = s3fs.S3FileSystem()

    parsed = urlparse(pattern)
    logger.info(f"Expanding remote glob '{pattern}'...")

    if parsed.scheme == "s3":
        return [f"s3://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "weka":
        return [f"weka://{obj}" for obj in fs.glob(pattern.replace("weka://", "s3://"))]
    elif parsed.scheme == "gs":
        return [f"gs://{obj}" for obj in fs.glob(pattern)]
    elif parsed.scheme == "r2":
        raise NotImplementedError("'r2' types are not currently supported")
    elif parsed.scheme in ("http", "https"):
        raise NotImplementedError("'http' types are not currently supported")
    elif parsed.scheme == "file":
        raise NotImplementedError("Remote 'file' types are not currently supported")
    else:
        raise NotImplementedError(f"Glob expansion is not currently supported for '{parsed.scheme}' files")
