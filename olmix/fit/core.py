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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from olmix.aliases import ExperimentConfig
from olmix.fit.utils import (
    PROPOSER_TYPES,
    LogLinearRegressor,
    add_back_in_fixed_source_weights,
    aggregate_mmlu,
    build_regression,
    compute_mixture_neighborhood,
    expand_collapsed_weights,
    get_runs_from_api,
    mk_run_metrics,
    mk_weights_from_config,
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
    # Regression options
    regression_type: str = "log_linear",
    train_split: tuple[float, ...] = (1.0,),
    n_test: int = 0,
    seed: int = 0,
    early_stopping: float = 0.0,
    interactions: tuple[str, ...] = (),
    # Proposer options
    opt_avg_metric: bool = False,
    proposer_type: str = "exact",
    simulation_samples: int = 100_000,
    constrain_objective: bool = False,
    temperature: float | None = None,
    neighborhood: str | None = None,
    # Filtering options
    keep_sources: list[str] | None = None,
    support_domains: tuple[str, ...] = (),
    drop_metrics: tuple[str, ...] = (),
    select_top_k_runs: float = 1.0,
    fixed_weight: str | None = None,
    metric_type: str | None = None,
    # DRO options
    dro_reference_model_id: str | None = None,
    use_reference_model_predicted_scores: bool = False,
    use_reference_model_as_search_prior: bool = False,
    use_hardcoded_reference_ratio: bool = False,
    # Other options
    alpha: float = 1.0,
    num_samples: int = 1,
    fit_only: bool = False,
    tol: float | None = None,
    fixed_search_weight: str | None = None,
    make_worst_mix: bool = False,
    min_weight_per_domain: float = 0.0,
    kl_reg: float | None = None,
    # WandB params (needed only for DRO reference model via WandB)
    wandb_api=None,
    workspace: str = "ai2-llm/regmixer",
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
        interactions: Feature interactions
        opt_avg_metric: Optimize average of all metrics
        proposer_type: Proposer type (simulation, search, exact)
        simulation_samples: Number of simulation samples
        constrain_objective: Constrain proposal by token budget
        temperature: Dirichlet temperature
        neighborhood: Training run name for neighborhood selection
        keep_sources: Only use runs with nonzero weight on these sources
        support_domains: Only use runs where these domains sum to 1
        drop_metrics: Metrics to exclude from fitting
        select_top_k_runs: Fraction of top runs to use
        fixed_weight: JSON string of fixed domain weights
        metric_type: Metric type for evaluation
        dro_reference_model_id: Reference model for DRO (YAML path or WandB run ID)
        use_reference_model_predicted_scores: Use predicted (not true) scores for reference
        use_reference_model_as_search_prior: Center search around reference model weights
        use_hardcoded_reference_ratio: Use hardcoded reference ratio
        alpha: Alpha for simulated distributions
        num_samples: Number of evaluation samples per metric
        fit_only: Only fit regression, don't propose
        tol: Pareto constraint tolerance
        fixed_search_weight: JSON string of fixed weights during search
        make_worst_mix: Invert objective for counterfactual
        min_weight_per_domain: Minimum weight per domain in proposal
        kl_reg: KL regularization lambda for log-linear exact proposer
        wandb_api: Optional WandB API instance (for DRO reference model)
        workspace: WandB workspace (for DRO reference model)
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
        "neighborhood": neighborhood,
        "keep_sources": keep_sources,
        "early_stopping": early_stopping,
    }
    if select_top_k_runs < 1.0:
        regression_config["select_top_k_runs"] = select_top_k_runs
    if fixed_weight is not None:
        regression_config["fixed_weight"] = fixed_weight
    if metric_type is not None:
        regression_config["metric_type"] = metric_type
    if len(interactions) != 0:
        regression_config["interactions"] = interactions
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

    # Hardcoded outlier drops (WandB mode only)
    if experiment_groups:
        if experiment_groups[0] == "870881c8":
            # hardcoded logic: drop outlier for reasoning swarm
            ratios = ratios.drop(index=27)
            metrics = metrics.drop(index=27)

        if experiment_groups[0] == "a09b2bf1":
            ratios = ratios.drop(index=[30, 47, 49])
            metrics = metrics.drop(index=[30, 47, 49])

        if experiment_groups == ["a3e06472", "515eaf2d"]:
            ratios = ratios.drop(index=[11, 12, 25, 27, 30, 35, 55])
            metrics = metrics.drop(index=[11, 12, 25, 27, 30, 35, 55])

    if select_top_k_runs < 1.0:
        metrics["all_bpb"] = metrics[metrics.columns[3:]].mean(axis=1)
        keep_runs = metrics.sort_values(by="all_bpb").run.values[: int(len(metrics) * select_top_k_runs)]
        metrics = metrics[metrics.run.isin(keep_runs)]
        ratios = ratios[ratios.run.isin(keep_runs)]

    if metric_type == "primary_score":
        logger.info("Doing z-score normalization on the primary scores...")
        cols_to_normalize = metrics.columns[3:]
        metrics[cols_to_normalize] = metrics[cols_to_normalize].apply(pd.to_numeric, errors="coerce")
        metrics[cols_to_normalize] = metrics[cols_to_normalize].apply(lambda col: (col - col.mean()) / col.std(ddof=0))

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

    # X = Domain weights
    X_train = ratios[ratios.columns[3:]].values
    # Y = Metric values
    Y_train = metrics[metrics.columns[3:]].values

    if n_test > 0:
        logger.info(f"Using {n_test} samples for test data")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X_train, Y_train, test_size=n_test / len(Y_train), random_state=seed
        )

    if train_split[0] != 1.0:
        # If we also want to subsample the training_data to study the effect of number of proxy runs
        logger.info(f"Subsampling training data to {train_split} of original size")

        train_split = tuple(int(t) if t > 1 else t for t in train_split)
        assert len(train_split) > 0

        if neighborhood is None:
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
        else:
            assert len(train_split) == 1, "If neighborhood is not set, train_split must be a single float"
            X_train, Y_train = compute_mixture_neighborhood(X_train, Y_train, ratios, neighborhood, train_split[0])

    if n_test == 0:
        X_test = deepcopy(X_train)
        Y_test = deepcopy(Y_train)

    logger.info(f"Number of train samples: {len(Y_train)}. Number of test samples: {len(Y_test)}.")

    predictors = []

    indexed_metrics = list(enumerate(metrics_to_index))
    logger.info(f"Fitting {regression_type} regression for metrics:")
    logger.info(indexed_metrics)

    # Objective weights are now fixed at uniform weighting
    obj_weights = None

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
            reg = LogLinearRegressor(params[metric])
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
                reg = LogLinearRegressor(params[metric])
                predictors.append(reg)
    else:
        logger.info(f"Will save regression model to {regression_model_cache_path}")
        for idx, metric in indexed_metrics:
            predictors.append(
                build_regression(
                    idx, Y_train, X_train, regression_type, early_stopping, list(interactions) if interactions else None
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
        metric_type,
    )
    plot_interaction_matrix_signed_evidence(
        output_dir,
        predictors,
        regression_type,
        ratios.columns[3:].tolist(),
        metrics.columns[3:].tolist(),
        ratios,
        metric_type,
    )
    results = []

    reference_ratio = None
    reference_scores = None
    if dro_reference_model_id is not None:
        # load in metrics of the reference model
        if dro_reference_model_id.endswith("yaml"):
            with open(dro_reference_model_id) as f:
                dro_config = yaml.safe_load(f)

            assert all([entry["domain"] == ratios.columns[3:][i] for i, entry in enumerate(dro_config["sources"])])

            reference_ratio = np.array([entry["weight"] for entry in dro_config["sources"]])
            reference_ratio /= np.sum(reference_ratio)  # normalize the weights
            reference_scores = [pred.predict(reference_ratio)[0] for pred in predictors]
            reference_scores = np.array(reference_scores)

        else:
            if wandb_api is None:
                import wandb

                wandb_api = wandb.Api()

            eval_metric_group_name = "all_wandb_metrics"
            cache_path = (
                pathlib.Path(BASE_CACHE_DIR)
                / f"{'_'.join(experiment_groups or ['csv'])}_{eval_metric_group_name}_runs_cache.json"
            )
            reference_model_run_instance = get_runs_from_api(
                wandb_api, workspace, [dro_reference_model_id], cache_path, True, num_samples, eval_metrics
            )[0]

            if use_reference_model_predicted_scores:
                # get reference model's mix and pass this through the regression model
                reference_run_ratio = {
                    "run": reference_model_run_instance.id,
                    "name": reference_model_run_instance.display_name,
                    "index": 0,
                    **mk_weights_from_config(
                        reference_model_run_instance.config, priors, reference_model_run_instance.display_name
                    ),
                }
                reference_ratio_df = pd.DataFrame([reference_run_ratio])
                reference_ratio = reference_ratio_df[reference_ratio_df.columns[3:]].values
                reference_scores = [pred.predict(reference_ratio)[0] for pred in predictors]
                reference_scores = np.array(reference_scores)
            else:
                # load in the reference model's true performance
                reference_run_metric = {
                    "run": reference_model_run_instance.id,
                    "name": reference_model_run_instance.display_name,
                    "index": 0,
                    **mk_run_metrics(
                        history=reference_model_run_instance.samples,
                        samples=num_samples,
                        metrics=(eval_metric_group_name, eval_metrics),
                        display_name=reference_model_run_instance.display_name,
                    ),
                }
                reference_scores = []
                for idx, metric in indexed_metrics:
                    reference_scores.append(reference_run_metric[metric])
                reference_scores = np.array(reference_scores)

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
            n_samples=num_samples,
            alpha=alpha,
            output_dir=output_dir,
        )

        if not opt_avg_metric and n_test == 0:
            weights = PROPOSER_TYPES[proposer_type]().propose(
                index=idx,
                predictor=predictors,
                prior_distributions=priors[0],
                original_prior=original_priors[0],
                num_samples=simulation_samples,
                opt_avg_metric=opt_avg_metric,
                constrain_objective=constrain_objective,
                swarm_config=launch_configs[0] if constrain_objective and launch_configs else None,
                obj_weights=obj_weights,
                temperature=temperature,
                reference_scores=reference_scores if dro_reference_model_id is not None else None,
                fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
                metric_type=metric_type,
                ratios=ratios,
                tol=tol,
                fixed_search_weight=fixed_search_weight,
                reference_ratio=reference_ratio if use_reference_model_as_search_prior else None,
                make_worst_mix=make_worst_mix,
                min_weight_per_domain=min_weight_per_domain,
            )

            plot_and_log_weights(
                prior=priors[0],
                original_prior=original_priors[0],
                prediction=weights,
                metric_name=metric,
                regression_type=regression_type,
                train_split=train_split,
                n_test=n_test,
                split_seed=seed,
                n_samples=num_samples,
                alpha=alpha,
                df_config=ratios,
                output_dir=output_dir,
                fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
                expand_collapsed_weights_fn=expand_collapsed_weights,
                add_back_in_fixed_source_weights_fn=add_back_in_fixed_source_weights,
            )

            results.append((metric, weights))

    if fit_only:
        logger.info("Fit only mode, not proposing a mix.")
        return

    if opt_avg_metric and n_test == 0:
        if (
            experiment_groups
            and experiment_groups[0] in ["5c712b3b", "daf37f03", "f5a3ff58"]
            and use_hardcoded_reference_ratio
        ):
            logger.info("Using hardcoded reference ratio for s2pdf + web one node swarms...")
            reference_ratio = np.array(
                [
                    0.75,
                    0.00528905,
                    0.00994264,
                    0.01429609,
                    0.01794769,
                    0.00955301,
                    0.00601014,
                    0.01521388,
                    0.00796704,
                    0.00801173,
                    0.01751576,
                    0.00872822,
                    0.01303348,
                    0.01352303,
                    0.01408978,
                    0.0128043,
                    0.02283908,
                    0.01021371,
                    0.01405873,
                    0.0094072,
                    0.01173855,
                    0.0078169,
                ]
            )

        weights = PROPOSER_TYPES[proposer_type]().propose(
            index=-1,
            predictor=predictors,
            prior_distributions=priors[0],
            original_prior=original_priors[0],
            num_samples=simulation_samples,
            opt_avg_metric=opt_avg_metric,
            constrain_objective=constrain_objective,
            swarm_config=launch_configs[0] if constrain_objective and launch_configs else None,
            obj_weights=obj_weights,
            temperature=temperature,
            reference_scores=reference_scores if dro_reference_model_id is not None else None,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
            metric_type=metric_type,
            ratios=ratios,
            tol=tol,
            fixed_search_weight=fixed_search_weight,
            reference_ratio=reference_ratio
            if use_reference_model_as_search_prior or reference_ratio is not None
            else None,
            make_worst_mix=make_worst_mix,
            min_weight_per_domain=min_weight_per_domain,
            kl_reg=kl_reg,
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
            n_samples=num_samples,
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
            expand_collapsed_weights_fn=expand_collapsed_weights,
            add_back_in_fixed_source_weights_fn=add_back_in_fixed_source_weights,
        )

        results.append(("opt_avg_all_metrics", weights))

    elif not opt_avg_metric and n_test == 0:
        # If we're not optimizing for the average of the metric group, then we average the reweighted distributions after fitting
        eval_metric_group_name = "all_wandb_metrics" if experiment_groups else "csv_metrics"
        avg_name = f"avg_{eval_metric_group_name}"
        average = np.mean([result[1] for result in results], axis=0)
        plot_and_log_weights(
            prior=priors[0],
            original_prior=original_priors[0],
            prediction=average,
            metric_name=avg_name,
            regression_type=regression_type,
            train_split=train_split,
            n_test=n_test,
            split_seed=seed,
            n_samples=num_samples,
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
            expand_collapsed_weights_fn=expand_collapsed_weights,
            add_back_in_fixed_source_weights_fn=add_back_in_fixed_source_weights,
        )

        results.append((avg_name, average))

    if n_test == 0:
        metric, weights = results[-1]
        predictions = np.array([p.predict(weights[None])[0] for p in predictors])
        if obj_weights is not None:
            predicted_performance = np.average(predictions, axis=0, weights=obj_weights)
        else:
            predicted_performance = predictions.mean(axis=0)
        logger.info(f"Metric: {metric}. Predicted performance using regression model: {predicted_performance}")

        with open(f"{output_dir}/predicted_performance.json", "w") as f:
            json.dump(float(predicted_performance), f)

        if dro_reference_model_id is not None and use_reference_model_predicted_scores:
            if metric_type == "primary_score":
                diff = predictions - reference_scores
            else:
                diff = reference_scores - predictions
            colors = ["green" if val > 0 else "red" for val in diff]
            x = np.arange(len(diff))

            plt.figure(figsize=(10, 6))
            plt.bar(x, diff, color=colors)
            plt.title("Pareto Improvement")

            plt.ylabel("PREDICTED Improvements over reference model")
            plt.axhline(0, color="black", linewidth=0.8)
            plt.xticks(ticks=x, labels=metrics.columns[3:].tolist(), rotation=90)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/predicted_pareto_improvement.png")
            plt.close()

    logger.info(f"Results saved to {output_dir}")
