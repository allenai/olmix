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
from sklearn.model_selection import train_test_split

from olmix.aliases import ExperimentConfig
from olmix.fit.constants import ALL_TASK_FAMILIES
from olmix.fit.utils import (
    PROPOSER_TYPES,
    REGRESSION_TYPES,
    add_back_in_fixed_source_weights,
    aggregate_mmlu,
    build_regression,
    expand_collapsed_weights,
)
from olmix.plots import (
    plot_and_log_weights,
    plot_correlation,
    plot_interaction_matrix,
    plot_interaction_matrix_signed_evidence,
)

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"


def run_fit(
    ratios: pd.DataFrame,
    metrics: pd.DataFrame,
    priors: tuple,
    original_priors: tuple,
    output_dir: str,
    *,
    eval_metrics: list[str] | None = None,
    experiment_groups: list[str] | None = None,
    launch_configs: list[ExperimentConfig] | None = None,
    full_group_names: list[str] | None = None,
    token_counts: dict[str, int] | None = None,
    # Regression options
    regression_type: str = "log_linear",
    train_split: tuple[float, ...] = (1.0,),
    n_test: int = 0,
    seed: int = 0,
    early_stopping: float = 0.0,
    # Proposer options
    opt_avg_metric: bool = False,
    proposer_type: str = "exact",
    simulation_samples: int = 100_000,
    constrain_objective: bool = False,
    temperature: float | None = None,
    # Filtering options
    keep_sources: list[str] | None = None,
    support_domains: tuple[str, ...] = (),
    drop_metrics: tuple[str, ...] = (),
    fixed_weight: str | None = None,
    obj_weights: dict[str, float] | None = None,
    # Other options
    alpha: float = 1.0,
    fit_only: bool = False,
    make_worst_mix: bool = False,
    kl_reg: float | None = None,
    # WandB params (needed only for DRO reference model via WandB)
    wandb_api=None,
    workspace: str = "ai2-llm/regmixer",
    target_tokens: int | None = None,
    repetition_factor: float = 5.0,
    natural_kl: tuple | None = None,
    test_ratios_path: tuple[str, ...] = (),
    test_metrics_path: tuple[str, ...] = (),
    aggregate_task_families: bool = False,
) -> None:
    """Run the regression fitting and mixture proposing pipeline.

    Args:
        ratios: DataFrame [run, name, index, ...domain weights]
        metrics: DataFrame [run, name, index, ...metric values]
        priors: Priors tuple from calculate_priors_with_manual
        original_priors: Original priors before collapsing
        output_dir: Directory for saving outputs
        eval_metrics: List of metric names to fit on. None = derive from metrics.columns[3:]
        experiment_groups: Group IDs (used for hardcoded outlier drops; None in CSV mode)
        launch_configs: ExperimentConfig objects (needed for constrain_objective)
        full_group_names: Full WandB group names (needed for multi-split training; None in CSV mode)
        regression_type: Regression model type
        train_split: Fraction(s) of dataset used for training
        n_test: Number of test samples
        seed: Random state for train-test split
        early_stopping: Epsilon for early stopping
        opt_avg_metric: Optimize average of all metrics
        proposer_type: Proposer type (simulation, search, exact)
        simulation_samples: Number of simulation samples
        constrain_objective: Constrain proposal by token budget
        temperature: Dirichlet temperature
        keep_sources: Only use runs with nonzero weight on these sources
        support_domains: Only use runs where these domains sum to 1
        drop_metrics: Metrics to exclude from fitting
        fixed_weight: JSON string of fixed domain weights
        alpha: Alpha for simulated distributions
        fit_only: Only fit regression, don't propose
        make_worst_mix: Invert objective for counterfactual
        kl_reg: KL regularization lambda for log-linear exact proposer
        wandb_api: Optional WandB API instance (for DRO reference model)
        workspace: WandB workspace (for DRO reference model)
        target_tokens: number of tokens (R) requested
        natural_kl: whether to use natural distribution for KL regularization, even when swarm dirichlet prior is different
        test_ratios_path: Optional paths to ratios DataFrames for test set
        test_metrics_path: Optional paths to metrics DataFrames for test set
        aggregate_task_families: Whether to aggregate metrics by task family rather than per task
    """
    if eval_metrics is None:
        eval_metrics = list(metrics.columns[3:])

    fixed_weight_dict = json.loads(fixed_weight) if fixed_weight is not None else None

    # Build regression config for caching
    regression_config: dict = {
        "regression_type": regression_type,
        "train_split": train_split[0] if len(train_split) == 1 else train_split,
        "n_test": n_test,
        "seed": seed,
        "keep_sources": keep_sources,
        "early_stopping": early_stopping,
    }
    if fixed_weight is not None:
        regression_config["fixed_weight"] = fixed_weight
    if len(support_domains) != 0:
        regression_config["support_domains"] = support_domains

    metrics_to_index = eval_metrics

    if len(support_domains) != 0:
        keep_idxs = np.where(np.isclose(ratios[list(support_domains)].sum(axis=1), 1))[0]
        ratios = ratios.iloc[keep_idxs]
        drop_col = list(set(ratios.columns[3:]).difference(set(support_domains)))
        ratios = ratios.drop(columns=drop_col)
        metrics = metrics.iloc[keep_idxs]

        new_priors = {k: v for k, v in priors[0].items() if k in list(support_domains)}
        total = sum(list(new_priors.values()))
        new_priors = {k: v / total for k, v in new_priors.items()}
        priors[0].clear()
        priors[0].update(new_priors)

    if all("mmlu_stem" not in s for s in metrics.columns) and any("mmlu" in s for s in metrics.columns):
        metrics, metrics_to_index = aggregate_mmlu(metrics, metrics_to_index)

    if len(ratios[ratios.columns[3:]]) > len(ratios):
        raise ValueError("The number of swarm runs is fewer than the number of mixing sources.")

    if keep_sources:
        old_len = len(ratios)
        other_columns = list(set(ratios.columns[3:]).difference(set(keep_sources)))
        ratios = ratios[
            ratios[list(keep_sources)].ne(0).all(axis=1)  # all specified columns nonzero
            & ratios[other_columns].eq(0).all(axis=1)
        ]
        logger.info(f"Filtered out {old_len - len(ratios)} runs that were not only on {keep_sources}")
        metrics = metrics[metrics["name"].isin(ratios["name"])]
        ratios.drop(columns=other_columns, inplace=True)

    cols_to_check = metrics.columns[3:]
    bad_rows = metrics[metrics[cols_to_check].isna().any(axis=1)]

    if not bad_rows.empty:
        logger.warning(f"Found NaNs in the following rows, dropping them! {bad_rows.index.tolist()}")
        metrics = metrics.drop(index=bad_rows.index)
        ratios = ratios.drop(index=bad_rows.index)

    if regression_type == "log_linear":
        if (metrics[metrics.columns[3:]] < 0).any().any():
            logger.info("Log-linear regression requires non-negative metrics, shifting metrics to be non-negative.")
            metrics[metrics.columns[3:]] = metrics[metrics.columns[3:]].subtract(metrics[metrics.columns[3:]].min())

    if aggregate_task_families:
        meta_cols = metrics.columns[:3]
        task_cols = metrics.columns[3:]
        metrics_new = metrics.loc[:, meta_cols].copy()

        for family, tasks in ALL_TASK_FAMILIES.items():
            # Only keep tasks that actually exist in the dataframe
            existing = [t for t in tasks if t in task_cols]

            if not existing:
                raise ValueError(f"No columns found for task family '{family}'")

            # Row-wise mean across the family
            metrics_new[family] = metrics[existing].mean(axis=1)
        metrics = metrics_new
        metrics_to_index = list(ALL_TASK_FAMILIES.keys())

    # X = Domain weights
    X_train = ratios[ratios.columns[3:]].values
    # Y = Metric values
    Y_train = metrics[metrics.columns[3:]].values

    if len(test_ratios_path) != 0 and len(test_metrics_path) != 0:
        test_ratios = [pd.read_pickle(ratios_path) for ratios_path in test_ratios_path]
        test_metrics = []
        for metrics_path in test_metrics_path:
            tm = pd.read_pickle(metrics_path)
            if all("mmlu_stem" not in s for s in tm.columns) and any("mmlu" in s for s in tm.columns):
                tm, _ = aggregate_mmlu(tm, metrics_to_index)

            if aggregate_task_families:
                # we need to aggregate the test set metrics as well
                meta_cols = tm.columns[:3]
                task_cols = tm.columns[3:]
                metrics_new = tm.loc[:, meta_cols].copy()

                for family, tasks in ALL_TASK_FAMILIES.items():
                    # Only keep tasks that actually exist in the dataframe
                    existing = [t for t in tasks if t in task_cols]

                    if not existing:
                        raise ValueError(f"No columns found for task family '{family}'")

                    # Row-wise mean across the family
                    metrics_new[family] = tm[existing].mean(axis=1)
                tm = metrics_new

            test_metrics.append(tm)

        X_test = np.concatenate([tr[tr.columns[3:]].values for tr in test_ratios])
        Y_test = np.concatenate([tm[tm.columns[3:]].values for tm in test_metrics])

    if n_test > 0 and (len(test_ratios_path) == 0 or len(test_metrics_path) == 0):
        logger.info(f"Using {n_test} samples for test data")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, test_size=n_test / len(Y_train), random_state=seed
        )

    if train_split[0] != 1.0:
        # If we also want to subsample the training_data to study the effect of number of proxy runs
        logger.info(f"Subsampling training data to {train_split} of original size")

        train_split = tuple(int(t) if t > 1 else t for t in train_split)
        assert len(train_split) > 0

        # we IID subselect training data

        if len(train_split) > 1:
            if full_group_names is None:
                raise ValueError("full_group_names required for multi-split training (not available in CSV mode)")
            all_x = []
            all_y = []
            for i, t in enumerate(train_split):
                ratios_subset = ratios[ratios["name"].str.contains(full_group_names[i])]
                metrics_subset = metrics[metrics["name"].str.contains(full_group_names[i])]

                X_train_subset = ratios_subset[ratios_subset.columns[3:]].values
                Y_train_subset = metrics_subset[metrics_subset.columns[3:]].values

                X_train_subset, _, Y_train_subset, _ = train_test_split(
                    X_train_subset, Y_train_subset, train_size=t, random_state=seed
                )

                all_x.append(X_train_subset)
                all_y.append(Y_train_subset)
            X_train = np.concatenate(all_x)
            Y_train = np.concatenate(all_y)
        else:
            if train_split[0] == len(Y_train):
                logger.info("Train split is the same as the dataset size, not subsampling...")
            else:
                X_train, _, Y_train, _ = train_test_split(
                    X_train, Y_train, train_size=train_split[0], random_state=seed
                )

    if n_test == 0 and (len(test_ratios_path) == 0 or len(test_metrics_path) == 0):
        X_test = deepcopy(X_train)
        Y_test = deepcopy(Y_train)

    logger.info(f"Number of train samples: {len(Y_train)}. Number of test samples: {len(Y_test)}.")

    predictors = []

    indexed_metrics = list(enumerate(metrics_to_index))
    logger.info(f"Fitting {regression_type} regression for metrics:")
    logger.info(indexed_metrics)

    if obj_weights:
        obj_weights = [obj_weights.get(metric, 1) for idx, metric in indexed_metrics]
        logger.info(f"Minimizing weighted average: {obj_weights}")

    # Caching logic for regression model
    experiment_groups_key = "_".join(experiment_groups) if experiment_groups else "csv"
    regression_config_str = json.dumps(regression_config, sort_keys=True)
    hash_str = hashlib.sha256(regression_config_str.encode("utf-8")).hexdigest()[:16]
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
        # look in output_dir
        with open(os.path.join(output_dir, "path_to_regression_model.txt")) as f:
            regression_model_cache_path = pathlib.Path(f.read().strip())
        if os.path.exists(regression_model_cache_path):
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
            # save intermediate progress after each regression model
            if regression_type == "log_linear":
                parameters = {indexed_metrics[i][-1]: predictors[i].model for i in range(len(predictors))}
                with open(str(regression_model_cache_path).split(".pkl")[0] + f"_{idx}.pkl", "wb") as f:
                    pickle.dump(parameters, f)
                logger.info(
                    f"First {idx} regression models saved to {str(regression_model_cache_path).split('.pkl')[0] + f'_{idx}.pkl'}"
                )
                with open(os.path.join(output_dir, "path_to_regression_model.txt"), "w") as f:
                    f.write(str(regression_model_cache_path))

        if regression_type == "log_linear":
            parameters = {metric: predictors[idx].model for idx, metric in indexed_metrics}
            with open(regression_model_cache_path, "wb") as f:
                pickle.dump(parameters, f)
            logger.info(f"Log linear regression model saved to {regression_model_cache_path}")
            with open(os.path.join(output_dir, "path_to_regression_model.txt"), "w") as f:
                f.write(str(regression_model_cache_path))

    if len(drop_metrics) != 0:
        drop_indices = [
            metrics.columns.get_loc(m) - 3  # shift because metrics start at col 3
            for m in drop_metrics
            if m in metrics.columns[3:]
        ]
        logger.info(f"Dropping metrics {drop_metrics} at indices {drop_indices}")
        # Remove those predictors by index
        predictors = [p for i, p in enumerate(predictors) if i not in drop_indices]
        metrics = metrics.drop(columns=list(drop_metrics), errors="ignore")
        Y_train = np.delete(Y_train, drop_indices, axis=1)
        Y_test = np.delete(Y_test, drop_indices, axis=1)

        metrics_to_index = [m for i, m in indexed_metrics if i not in drop_indices]
        indexed_metrics = list(enumerate(metrics_to_index))

    plot_interaction_matrix(
        output_dir,
        predictors,
        regression_type,
        ratios.columns[3:].tolist(),
        metrics.columns[3:].tolist(),
        ratios,
    )
    plot_interaction_matrix_signed_evidence(
        output_dir,
        predictors,
        regression_type,
        ratios.columns[3:].tolist(),
        metrics.columns[3:].tolist(),
        ratios,
    )
    results = []

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
            alpha=alpha,
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
        alpha=alpha,
        output_dir=output_dir,
        average_bpb=True,
        test_ratios_path=test_ratios_path,
    )

    if fit_only or n_test > 0:
        logger.info("Either fit only mode or n_test>0, not proposing a mix.")
        return

    if opt_avg_metric:
        weights = PROPOSER_TYPES[proposer_type]().propose(
            index=-1,
            predictor=predictors,
            prior_distributions=priors[0],
            num_samples=simulation_samples,
            opt_avg_metric=opt_avg_metric,
            constrain_objective=constrain_objective,
            swarm_config=launch_configs[0] if constrain_objective and launch_configs else None,
            obj_weights=obj_weights,
            temperature=temperature,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
            make_worst_mix=make_worst_mix,
            kl_reg=kl_reg,
            target_tokens=target_tokens,
            repetition_factor=repetition_factor,
            token_counts=token_counts,
            manual_kl=natural_kl,
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
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
            expand_collapsed_weights_fn=expand_collapsed_weights,
            add_back_in_fixed_source_weights_fn=add_back_in_fixed_source_weights,
        )

        results.append(("opt_avg_all_metrics", weights))

    metric, weights = results[-1]
    predictions = np.array([p.predict(weights[None])[0] for p in predictors])
    if obj_weights is not None:
        predicted_performance = np.average(predictions, axis=0, weights=obj_weights)
    else:
        predicted_performance = predictions.mean(axis=0)
    logger.info(f"Metric: {metric}. Predicted performance using regression model: {predicted_performance}")

    with open(f"{output_dir}/predicted_performance.json", "w") as f:
        json.dump(float(predicted_performance), f)

    logger.info(f"Results saved to {output_dir}")
