"""Core fitting and proposing logic for olmix fit module.

Provides run_fit() which takes pre-loaded ratios/metrics DataFrames and runs
the regression fitting and mixture proposing pipeline.
"""

import hashlib
import json
import logging
import os
import pathlib
import pickle
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

from olmix.aliases import LaunchConfig
from olmix.fit.utils import (
    PROPOSER_TYPES,
    REGRESSION_TYPES,
    aggregate_mmlu,
    build_regression,
    expand_collapsed_weights,
)
from olmix.plots import (
    plot_and_log_weights,
    plot_correlation,
    plot_interaction_matrix,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"


class RegressionCacheConfig(BaseModel):
    """Configuration for regression model caching.

    Captures all parameters that affect the regression model, including
    hashes of input dataframes to ensure cache uniqueness.
    """

    ratios_hash: str
    metrics_hash: str
    regression_type: str
    train_split: float
    n_test: int
    seed: int
    early_stopping: float
    aggregate_task_families: bool

    def get_hash(self) -> str:
        """Get hash string for cache key."""
        config_dict = self.model_dump(exclude_none=True, exclude_defaults=True)
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


def run_fit(
    ratios: pd.DataFrame,
    metrics: pd.DataFrame,
    priors: tuple,
    original_priors: tuple,
    output_dir: str,
    *,
    domain_cols: list[str],
    metric_cols: list[str],
    eval_metrics: list[str] | None = None,
    experiment_groups: list[str] | None = None,
    launch_configs: list[LaunchConfig] | None = None,
    full_group_names: list[str] | None = None,
    token_counts: dict[str, int] | None = None,
    # Regression options
    regression_type: str = "log_linear",
    train_split: float = 1.0,
    n_test: int = 0,
    seed: int = 0,
    early_stopping: float = 0.0,
    # Proposer options
    proposer_type: str = "exact",
    constrain_objective: bool = False,
    temperature: float | None = None,
    # Filtering options
    drop_metrics: tuple[str, ...] = (),
    obj_weights: dict[str, float] | None = None,
    # Other options
    fit_only: bool = False,
    make_worst_mix: bool = False,
    kl_reg: float | None = None,
    target_tokens: int | None = None,
    repetition_factor: float = 5.0,
    test_ratios_path: tuple[str, ...] = (),
    test_metrics_path: tuple[str, ...] = (),
    aggregate_task_families: bool = False,
    task_families: dict[str, list[str]] | None = None,
) -> None:
    """Run the regression fitting and mixture proposing pipeline.

    Args:
        ratios: DataFrame [run, name, index, ...domain weights]
        metrics: DataFrame [run, name, index, ...metric values]
        priors: Priors tuple from calculate_priors_with_manual
        original_priors: Original priors before collapsing
        output_dir: Directory for saving outputs
        domain_cols: Domain weight column names (from load_from_csv).
        metric_cols: Metric value column names (from load_from_csv).
        eval_metrics: List of metric names to fit on. None = derive from metric_cols
        experiment_groups: Group IDs (used for hardcoded outlier drops; None in CSV mode)
        launch_configs: ExperimentConfig objects (needed for constrain_objective)
        full_group_names: Full WandB group names (needed for multi-split training; None in CSV mode)
        regression_type: Regression model type
        train_split: Fraction or number of samples of dataset used for training
        n_test: Number of test samples
        seed: Random state for train-test split
        early_stopping: Epsilon for early stopping
        proposer_type: Proposer type (simulation, search, exact)
        constrain_objective: Constrain proposal by token budget
        temperature: Dirichlet temperature
        drop_metrics: Metrics to exclude from fitting
        fit_only: Only fit regression, don't propose
        make_worst_mix: Invert objective for counterfactual
        kl_reg: KL regularization lambda for log-linear exact proposer
        target_tokens: number of tokens (R) requested
        test_ratios_path: Optional paths to ratios DataFrames for test set
        test_metrics_path: Optional paths to metrics DataFrames for test set
        aggregate_task_families: Whether to aggregate metrics by task family rather than per task
    """
    if eval_metrics is not None:
        metric_cols = list(eval_metrics)

    # Build regression config for caching
    # Hash input dataframes to ensure cache uniqueness
    ratios_hash = hashlib.sha256(pd.util.hash_pandas_object(ratios, index=True).values.tobytes()).hexdigest()
    metrics_hash = hashlib.sha256(pd.util.hash_pandas_object(metrics, index=True).values.tobytes()).hexdigest()

    regression_config = RegressionCacheConfig(
        ratios_hash=ratios_hash,
        metrics_hash=metrics_hash,
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        seed=seed,
        early_stopping=early_stopping,
        aggregate_task_families=aggregate_task_families,
    )

    if all("mmlu_stem" not in s for s in metrics.columns) and any("mmlu" in s for s in metrics.columns):
        metrics, metric_cols = aggregate_mmlu(metrics, metric_cols)

    if len(ratios[domain_cols]) > len(ratios):
        raise ValueError("The number of swarm runs is fewer than the number of mixing sources.")

    bad_rows = metrics[metrics[metric_cols].isna().any(axis=1)]
    if not bad_rows.empty:
        logger.warning(f"Found NaNs in the following rows, dropping them! {bad_rows.index.tolist()}")
        metrics = metrics.drop(index=bad_rows.index)
        ratios = ratios.drop(index=bad_rows.index)

    if aggregate_task_families:
        if task_families is None:
            raise ValueError("task_families must be provided when aggregate_task_families=True")
        meta_cols = [c for c in metrics.columns if c not in metric_cols]
        metrics_new = metrics.loc[:, meta_cols].copy()

        for family, tasks in task_families.items():
            # Only keep tasks that actually exist in the dataframe
            existing = [t for t in tasks if t in metric_cols]

            if not existing:
                raise ValueError(f"No columns found for task family '{family}'")

            # Row-wise mean across the family
            metrics_new[family] = metrics[existing].mean(axis=1)
        metrics = metrics_new
        metric_cols = list(task_families.keys())

    # X = Domain weights
    X_train = ratios[domain_cols].values
    # Y = Metric values
    Y_train = metrics[metric_cols].values

    if len(test_ratios_path) != 0 and len(test_metrics_path) != 0:
        test_ratios = [pd.read_pickle(ratios_path) for ratios_path in test_ratios_path]
        test_metrics = []
        for metrics_path in test_metrics_path:
            tm = pd.read_pickle(metrics_path)
            if all("mmlu_stem" not in s for s in tm.columns) and any("mmlu" in s for s in tm.columns):
                tm, _ = aggregate_mmlu(tm, metric_cols)

            if aggregate_task_families:
                assert task_families is not None
                # we need to aggregate the test set metrics as well
                meta_cols = [c for c in tm.columns if c not in metric_cols]
                task_cols = [c for c in metric_cols if c in tm.columns]
                metrics_new = tm.loc[:, meta_cols].copy()

                for family, tasks in task_families.items():
                    # Only keep tasks that actually exist in the dataframe
                    existing = [t for t in tasks if t in task_cols]

                    if not existing:
                        raise ValueError(f"No columns found for task family '{family}'")

                    # Row-wise mean across the family
                    metrics_new[family] = tm[existing].mean(axis=1)
                tm = metrics_new

            test_metrics.append(tm)

        X_test = np.concatenate([tr[[c for c in domain_cols if c in tr.columns]].values for tr in test_ratios])
        Y_test = np.concatenate([tm[[c for c in metric_cols if c in tm.columns]].values for tm in test_metrics])

    if n_test > 0 and (len(test_ratios_path) == 0 or len(test_metrics_path) == 0):
        logger.info(f"Using {n_test} samples for test data")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, test_size=n_test / len(Y_train), random_state=seed
        )

    if train_split != 1.0:
        logger.info(f"Subsampling training data to {train_split} of original size")
        train_split = int(train_split) if train_split > 1 else train_split
        if train_split == len(Y_train):
            logger.info("Train split is the same as the dataset size, not subsampling...")
        else:
            X_train, _, Y_train, _ = train_test_split(X_train, Y_train, train_size=train_split, random_state=seed)

    if n_test == 0 and (len(test_ratios_path) == 0 or len(test_metrics_path) == 0):
        X_test = deepcopy(X_train)
        Y_test = deepcopy(Y_train)

    logger.info(f"Number of train samples: {len(Y_train)}. Number of test samples: {len(Y_test)}.")

    predictors = []

    indexed_metrics = list(enumerate(metric_cols))
    logger.info(f"Fitting {regression_type} regression for metrics:")
    logger.info(indexed_metrics)

    obj_weights_list: list[float] | None = None
    if obj_weights:
        obj_weights_list = [obj_weights.get(metric, 1) for idx, metric in indexed_metrics]
        logger.info(f"Minimizing weighted average: {obj_weights}")
    else:
        obj_weights_list = None

    # Reorder priors to match ratios order
    priors_reordered = {domain: priors[0][domain] for domain in domain_cols if domain in priors[0]}
    if set(priors_reordered.keys()) != set(priors[0].keys()):
        missing_in_csv = set(priors[0].keys()) - set(domain_cols)
        missing_in_priors = set(domain_cols) - set(priors[0].keys())
        if missing_in_csv:
            logger.warning(f"Domains in priors but not in CSV columns: {missing_in_csv}")
        if missing_in_priors:
            raise ValueError(f"Domains in CSV columns but not in priors: {missing_in_priors}")
    priors[0].clear()
    priors[0].update(priors_reordered)

    assert set(domain_cols) == set(original_priors[0].keys()), "Mismatch between CSV columns and original priors keys"
    original_priors_reordered = {
        domain: original_priors[0][domain] for domain in domain_cols if domain in original_priors[0]
    }
    original_priors[0].clear()
    original_priors[0].update(original_priors_reordered)

    # Caching logic for regression model
    experiment_groups_key = "_".join(experiment_groups) if experiment_groups else "csv"
    hash_str = regression_config.get_hash()
    regression_model_cache_folder = pathlib.Path(BASE_CACHE_DIR) / experiment_groups_key / hash_str
    regression_model_cache_folder.mkdir(parents=True, exist_ok=True)
    regression_model_cache_path = regression_model_cache_folder / "regression_params.pkl"
    if os.path.exists(regression_model_cache_path) and regression_type == "log_linear":
        logger.info(f"Using log-linear regression model at {regression_model_cache_path}")
        with open(regression_model_cache_path, "rb") as f:
            params = pickle.load(f)

        # link the regression model cache to the run that uses it
        with open(os.path.join(output_dir, "path_to_regression_model.txt"), "w") as f:
            f.write(str(regression_model_cache_path))

        # initialize the regression models using the cached parameters
        for idx, metric in indexed_metrics:
            reg = REGRESSION_TYPES[regression_type](
                params=params[metric], requested_tokens=target_tokens if regression_type == "autoscale" else None
            )
            predictors.append(reg)
    elif (
        not os.path.exists(regression_model_cache_path)
        and regression_type == "log_linear"
        and os.path.exists(os.path.join(output_dir, "path_to_regression_model.txt"))
    ):
        logger.info(
            f"Model at {regression_model_cache_path} not found, but {os.path.join(output_dir, 'path_to_regression_model.txt')} exists. Attempting to load regression model from output directory link..."
        )
        # look in output_dir
        with open(os.path.join(output_dir, "path_to_regression_model.txt")) as f:
            regression_model_cache_path = pathlib.Path(f.read().strip())

        if not os.path.exists(regression_model_cache_path):
            raise ValueError(
                f"You may have deleted the cached regression model since the last run, but the output directory still links to it. Please delete {os.path.join(output_dir, 'path_to_regression_model.txt')}."
            )

        logger.info(f"Using log-linear regression model at {regression_model_cache_path}")
        with open(regression_model_cache_path, "rb") as f:
            params = pickle.load(f)

        # initialize the regression models using the cached parameters
        for idx, metric in indexed_metrics:
            reg = REGRESSION_TYPES[regression_type](
                params=params[metric], requested_tokens=target_tokens if regression_type == "autoscale" else None
            )
            predictors.append(reg)

    else:
        logger.info(f"Will save regression model to {regression_model_cache_path}")
        with open(regression_model_cache_folder / "config.json", "w") as f:
            json.dump(regression_config.model_dump(), f, indent=2, default=str)

        for idx, metric in indexed_metrics:
            predictors.append(
                build_regression(
                    idx,
                    Y_train,
                    X_train,
                    regression_type,
                    early_stopping,
                    target_tokens if regression_type == "autoscale" else None,
                )
            )

        if regression_type == "log_linear":
            parameters = {metric: predictors[idx].model for idx, metric in indexed_metrics}
            with open(regression_model_cache_path, "wb") as f:
                pickle.dump(parameters, f)
            logger.info(f"Log linear regression model saved to {regression_model_cache_path}")
            with open(os.path.join(output_dir, "path_to_regression_model.txt"), "w") as f:
                f.write(str(regression_model_cache_path))

    if len(drop_metrics) != 0:
        drop_indices = [list(metric_cols).index(m) for m in drop_metrics if m in metric_cols]
        logger.info(f"Dropping metrics {drop_metrics} at indices {drop_indices}")
        # Remove those predictors by index
        predictors = [p for i, p in enumerate(predictors) if i not in drop_indices]
        metrics = metrics.drop(columns=list(drop_metrics), errors="ignore")
        Y_train = np.delete(Y_train, drop_indices, axis=1)
        Y_test = np.delete(Y_test, drop_indices, axis=1)

        metric_cols = [m for m in metric_cols if m not in drop_metrics]
        indexed_metrics = list(enumerate(metric_cols))

    plot_interaction_matrix(
        output_dir,
        predictors,
        regression_type,
        domain_cols,
        metric_cols,
        ratios,
    )

    for idx, metric in indexed_metrics:
        plot_correlation(
            Y_test,
            X_test,
            Y_train,
            X_train,
            idx,
            predictors=predictors,
            train_split=train_split,
            metric_name=metric,
            regression_type=regression_type,
            n_test=n_test,
            split_seed=seed,
            output_dir=output_dir,
        )

    # plot correlation for average BPB as well
    plot_correlation(
        Y_test,
        X_test,
        Y_train,
        X_train,
        idx,
        predictors=predictors,
        train_split=train_split,
        metric_name=metric,
        regression_type=regression_type,
        n_test=n_test,
        split_seed=seed,
        output_dir=output_dir,
        average_bpb=True,
        test_ratios_path=test_ratios_path,
    )

    if fit_only or n_test > 0:
        logger.info("Either fit only mode or n_test>0, not proposing a mix.")
        logger.info(f"Results saved to {output_dir}")
        return

    weights = PROPOSER_TYPES[proposer_type]().propose(
        predictor=predictors,
        prior_distributions=priors[0],
        constrain_objective=constrain_objective,
        obj_weights=obj_weights_list,
        temperature=temperature,
        make_worst_mix=make_worst_mix,
        kl_reg=kl_reg,
        target_tokens=target_tokens,
        repetition_factor=repetition_factor,
        token_counts=token_counts,
    )
    plot_and_log_weights(
        prior=priors[0],
        original_prior=original_priors[0],
        prediction=weights,
        metric_name="opt_avg_all_metrics",
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        split_seed=seed,
        domain_cols=domain_cols,
        output_dir=output_dir,
        expand_collapsed_weights_fn=expand_collapsed_weights,
    )

    metric = "opt_avg_all_metrics"
    predictions = np.array([p.predict(weights[None])[0] for p in predictors])
    if obj_weights_list is not None:
        predicted_performance = np.average(predictions, axis=0, weights=obj_weights_list)
    else:
        predicted_performance = predictions.mean(axis=0)
    logger.info(f"Metric: {metric}. Predicted performance using regression model: {predicted_performance}")

    with open(f"{output_dir}/predicted_performance.json", "w") as f:
        json.dump(float(predicted_performance), f)

    logger.info(f"Results saved to {output_dir}")
