"""Data loaders for the olmix fit module.

Provides load_from_csv() to load ratios/metrics from local CSV files.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_from_csv(
    ratios_path: str,
    metrics_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratios and metrics from CSV files.

    Accepts both ``run_id`` and ``run`` as the ID column (the first one found
    is used).  The returned DataFrames use the canonical format
    ``[run, name, index, ...data columns]``.

    Args:
        ratios_path: Path to the ratios CSV file.
        metrics_path: Path to the metrics CSV file.

    Returns:
        Tuple of (ratios, metrics) DataFrames.

    Raises:
        FileNotFoundError: If either CSV file is missing.
        ValueError: If no recognised ID column or ratios don't sum to ~1.0.
    """
    ratios_raw = pd.read_csv(ratios_path)
    metrics_raw = pd.read_csv(metrics_path)

    # Resolve ID column — accept both "run_id" and "run"
    id_col = _resolve_id_column(ratios_raw, "ratios")
    _resolve_id_column(metrics_raw, "metrics")  # validate it exists

    # Determine domain / metric columns (everything except id, name, index, unnamed index cols)
    skip_cols = {id_col, "run_id", "run", "name", "index"}
    domain_cols = [c for c in ratios_raw.columns if c not in skip_cols and not c.startswith("Unnamed")]
    metric_cols = [c for c in metrics_raw.columns if c not in skip_cols and not c.startswith("Unnamed")]

    # Validate ratios sum to ~1.0
    row_sums = ratios_raw[domain_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=0.01):
        bad_rows = ratios_raw.index[~np.isclose(row_sums, 1.0, atol=0.01)].tolist()
        raise ValueError(
            f"Ratios do not sum to ~1.0 for rows: {bad_rows}. Got sums: {row_sums.iloc[bad_rows].tolist()}"
        )

    # Build canonical format: [run, name, index, ...data columns]
    has_name = "name" in ratios_raw.columns

    ratios = pd.DataFrame()
    ratios["run"] = ratios_raw[id_col]
    ratios["name"] = ratios_raw["name"] if has_name else ratios_raw[id_col]
    ratios["index"] = range(len(ratios_raw))
    for col in domain_cols:
        ratios[col] = ratios_raw[col]

    metrics = pd.DataFrame()
    metrics_id_col = "run" if "run" in metrics_raw.columns else "run_id"
    metrics["run"] = metrics_raw[metrics_id_col]
    metrics["name"] = metrics_raw["name"] if "name" in metrics_raw.columns else metrics_raw[metrics_id_col]
    metrics["index"] = range(len(metrics_raw))
    for col in metric_cols:
        metrics[col] = pd.to_numeric(metrics_raw[col], errors="coerce")

    # Filter to common run IDs
    common_runs = set(ratios["run"]) & set(metrics["run"])
    ratios = ratios[ratios["run"].isin(common_runs)].reset_index(drop=True)
    metrics = metrics[metrics["run"].isin(common_runs)].reset_index(drop=True)

    logger.info(f"Loaded {len(ratios)} runs from CSV: {len(domain_cols)} domains, {len(metric_cols)} metrics")

    return ratios, metrics


def _resolve_id_column(df: pd.DataFrame, label: str) -> str:
    """Return the name of the ID column in *df*, preferring ``run`` over ``run_id``."""
    if "run" in df.columns:
        return "run"
    if "run_id" in df.columns:
        return "run_id"
    raise ValueError(f"{label} CSV must have a 'run' or 'run_id' column")
