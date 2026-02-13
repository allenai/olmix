"""Data loaders for the olmix fit module.

Provides two loader functions:
- load_from_wandb: Load ratios/metrics from WandB using a launch output directory
- load_from_csv: Load ratios/metrics from local CSV files
"""

import json
import logging
import os
import pathlib

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

from olmix.aliases import ExperimentConfig
from olmix.fit.constants import ALL_WANDB_METRICS
from olmix.fit.utils import (
    calculate_priors_with_manual,
    get_runs_from_api,
    mk_run_from_json,
    mk_run_metrics,
    mk_weights_from_config,
)

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"


def _find_metadata_json(launch_output_dir: str) -> str:
    """Find the metadata JSON file in a launch output directory.

    Looks for 'metadata.json' first, then falls back to the most recent
    timestamped JSON file (e.g., '20260204_072937-f4d79447.json').
    """
    metadata_path = os.path.join(launch_output_dir, "metadata.json")
    if os.path.exists(metadata_path):
        return metadata_path

    # Fall back to timestamped JSON files
    json_files = sorted(
        [f for f in os.listdir(launch_output_dir) if f.endswith(".json")],
        reverse=True,
    )
    if json_files:
        return os.path.join(launch_output_dir, json_files[0])

    raise FileNotFoundError(f"No metadata JSON file found in {launch_output_dir}")


def load_from_wandb(
    launch_output_dir: str,
    *,
    workspace: str = "ai2-llm/regmixer",
    no_cache: bool = False,
    dashboard: list[str] | None = None,
    patched: bool = False,
    fixed_weight_dict: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[ExperimentConfig], tuple, tuple, list[str]]:
    """Load ratios and metrics from WandB using a launch output directory.

    Reads the metadata JSON from the launch output directory to auto-resolve
    the group_id and experiment config.

    Args:
        launch_output_dir: Path to the launch output directory containing metadata JSON
        workspace: WandB workspace
        no_cache: Disable caching
        dashboard: Dashboard names for eval results
        patched: Apply patched domain name logic
        fixed_weight_dict: Dict of fixed domain weights (already parsed from JSON)

    Returns:
        Tuple of (ratios, metrics, launch_configs, priors, original_priors, experiment_groups)
    """
    if dashboard is None:
        dashboard = ["regmixer"]

    # Read metadata JSON
    metadata_path = _find_metadata_json(launch_output_dir)
    with open(metadata_path) as f:
        metadata = json.load(f)

    group_id = metadata["metadata"]["group_id"]
    config_data = metadata["config"]
    experiment_groups = [group_id]

    # Reconstruct ExperimentConfig from inline config
    launch_config = ExperimentConfig(**config_data)
    launch_configs = [launch_config]

    # Use saved priors from metadata if available, otherwise compute from S3
    if "priors" in metadata:
        logger.info("Using saved priors from launch metadata (skipping S3 prior calculation)")
        saved = metadata["priors"]
        priors = (saved["relative_sizes"], saved["total_tokens"], saved["token_counts"])
        from copy import deepcopy

        original_priors = deepcopy(priors)
    else:
        logger.info("No saved priors in metadata, calculating from source files...")
        priors, original_priors = calculate_priors_with_manual(
            source_configs=launch_config.sources,
            dtype=launch_config.dtype,
            use_cache=(not no_cache),
            manual_prior=launch_config.manual_prior if hasattr(launch_config, "manual_prior") else None,
            fixed_source_weights=launch_config.fixed_source_weights
            if hasattr(launch_config, "fixed_source_weights")
            else None,
        )

    if fixed_weight_dict is not None:
        new_priors = {k: v for k, v in priors[0].items() if k not in fixed_weight_dict}
        total = sum(new_priors.values())
        new_priors = {k: v / total for k, v in new_priors.items()}
        priors[0].clear()
        priors[0].update(new_priors)

    logger.info("Source weights:")
    logger.info(priors[0])

    # Initialize WandB API and fetch runs
    api = wandb.Api()
    eval_metrics = ALL_WANDB_METRICS
    eval_metric_group_name = "all_wandb_metrics"

    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_runs_cache.json"
    )
    full_group_names = [f"{lc.name}-{group}" for group, lc in zip(experiment_groups, launch_configs)]

    if no_cache:
        logger.info("Cache disabled, will not use cache for run samples...")
        run_instances = get_runs_from_api(api, workspace, full_group_names, cache_path, no_cache, eval_metrics)
    else:
        try:
            with open(cache_path) as f:
                run_dict = json.load(f)
                run_instances = [mk_run_from_json(run) for run in run_dict]
            logger.info(f"Loaded cached runs from {cache_path}")
        except FileNotFoundError:
            logger.warning(f"Failed to load cache from {cache_path}, fetching runs from API...")
            run_instances = get_runs_from_api(api, workspace, full_group_names, cache_path, no_cache, eval_metrics)

    logger.info(f"Found {len(run_instances)} runs in {workspace} that match group id filter {experiment_groups}...")

    # Build DataFrames (with pickle caching)
    ratios_cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_ratios.pkl"
    )
    metrics_cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_metrics.pkl"
    )

    if os.path.exists(ratios_cache_path) and os.path.exists(metrics_cache_path) and not no_cache:
        logger.info(f"Loading cached ratios and metrics from {ratios_cache_path} and {metrics_cache_path}")
        with open(ratios_cache_path, "rb") as f:
            ratios = pd.read_pickle(f)
        with open(metrics_cache_path, "rb") as f:
            metrics = pd.read_pickle(f)
        ratios = ratios[ratios["run"].isin(metrics["run"])]
    else:
        run_ratios = [
            {
                "run": run.id,
                "name": run.display_name,
                "index": idx,
                **mk_weights_from_config(run.config, priors, run.display_name, patched),
            }
            for idx, run in enumerate(run_instances)
        ]

        run_metrics = [
            {
                "run": run.id,
                "name": run.display_name,
                "index": idx,
                **mk_run_metrics(
                    history=run.samples,
                    metrics=(eval_metric_group_name, eval_metrics),
                    display_name=run.display_name,
                    dashboard=dashboard,
                ),
            }
            for idx, run in tqdm(enumerate(run_instances))
            if len(run.samples) > 0
        ]

        ratios = pd.DataFrame(run_ratios)
        metrics = pd.DataFrame(run_metrics)
        numerical_cols = metrics.columns[3:]
        metrics[numerical_cols] = metrics[numerical_cols].apply(pd.to_numeric, errors="coerce")
        ratios = ratios[ratios["run"].isin(metrics["run"])]

        if fixed_weight_dict is not None:
            domains = ratios.columns[3:]
            ratios[domains] = ratios[domains].div(ratios[domains].sum(axis=1), axis=0)

        pd.to_pickle(ratios, ratios_cache_path)
        pd.to_pickle(metrics, metrics_cache_path)
        logger.info(f"Saved ratios to {ratios_cache_path} and metrics to {metrics_cache_path}")

    return ratios, metrics, launch_configs, priors, original_priors, experiment_groups


def load_priors_from_json(
    json_path: str,
) -> tuple[dict[str, float], int, dict[str, int]]:
    """Load priors from a JSON file.

    Accepts either:
    - A launch metadata JSON (reads from the "priors" key)
    - A standalone priors JSON with "relative_sizes", "total_tokens", "token_counts"

    Args:
        json_path: Path to the JSON file

    Returns:
        Tuple of (relative_sizes, total_tokens, token_counts)

    Raises:
        ValueError: If the JSON file doesn't contain priors data
    """
    with open(json_path) as f:
        data = json.load(f)

    # Support both launch metadata JSON and standalone priors JSON
    if "priors" in data:
        priors_data = data["priors"]
    elif "relative_sizes" in data:
        priors_data = data
    else:
        raise ValueError(
            f"JSON file {json_path} does not contain priors data. "
            "Expected a 'priors' key (launch metadata) or 'relative_sizes' key (standalone priors)."
        )

    relative_sizes = priors_data["relative_sizes"]
    total_tokens = priors_data["total_tokens"]
    token_counts = priors_data["token_counts"]

    logger.info(f"Loaded priors from {json_path}: {len(relative_sizes)} domains, {total_tokens:,} total tokens")
    return (relative_sizes, total_tokens, token_counts)


def load_from_csv(
    csv_dir: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratios and metrics from CSV files in a directory.

    The directory must contain:
    - ratios.csv: columns [run_id, <domain columns...>]
    - metrics.csv: columns [run_id, <metric columns...>]

    Both files are joined on run_id. The returned DataFrames use the canonical
    format [run, name, index, ...data columns].

    Args:
        csv_dir: Path to directory containing ratios.csv and metrics.csv

    Returns:
        Tuple of (ratios, metrics) DataFrames

    Raises:
        FileNotFoundError: If ratios.csv or metrics.csv is missing
        ValueError: If run_id column is missing or ratios don't sum to ~1.0
    """
    ratios_path = os.path.join(csv_dir, "ratios.csv")
    metrics_path = os.path.join(csv_dir, "metrics.csv")

    if not os.path.exists(ratios_path):
        raise FileNotFoundError(f"ratios.csv not found in {csv_dir}")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.csv not found in {csv_dir}")

    ratios_raw = pd.read_csv(ratios_path)
    metrics_raw = pd.read_csv(metrics_path)

    if "run_id" not in ratios_raw.columns:
        raise ValueError("ratios.csv must have a 'run_id' column")
    if "run_id" not in metrics_raw.columns:
        raise ValueError("metrics.csv must have a 'run_id' column")

    # Validate ratios sum to ~1.0
    domain_cols = [c for c in ratios_raw.columns if c not in ("run_id", "name")]
    row_sums = ratios_raw[domain_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.01):
        bad_rows = ratios_raw.index[~np.isclose(row_sums, 1.0, atol=0.01)].tolist()
        raise ValueError(
            f"Ratios do not sum to ~1.0 for rows: {bad_rows}. Got sums: {row_sums.iloc[bad_rows].tolist()}"
        )

    # Build canonical format: [run, name, index, ...data columns]
    has_name = "name" in ratios_raw.columns

    ratios = pd.DataFrame()
    ratios["run"] = ratios_raw["run_id"]
    ratios["name"] = ratios_raw["name"] if has_name else ratios_raw["run_id"]
    ratios["index"] = range(len(ratios_raw))
    for col in domain_cols:
        ratios[col] = ratios_raw[col]

    metric_cols = [c for c in metrics_raw.columns if c not in ("run_id", "name")]

    metrics = pd.DataFrame()
    metrics["run"] = metrics_raw["run_id"]
    metrics["name"] = metrics_raw["name"] if "name" in metrics_raw.columns else metrics_raw["run_id"]
    metrics["index"] = range(len(metrics_raw))
    for col in metric_cols:
        metrics[col] = pd.to_numeric(metrics_raw[col], errors="coerce")

    # Filter to common run_ids
    common_runs = set(ratios["run"]) & set(metrics["run"])
    ratios = ratios[ratios["run"].isin(common_runs)].reset_index(drop=True)
    metrics = metrics[metrics["run"].isin(common_runs)].reset_index(drop=True)

    logger.info(f"Loaded {len(ratios)} runs from CSV: {len(domain_cols)} domains, {len(metric_cols)} metrics")

    return ratios, metrics
