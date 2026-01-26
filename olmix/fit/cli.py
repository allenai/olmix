import hashlib
import json
import logging
import os
import pathlib
import pickle
import re
import subprocess
import warnings
from copy import deepcopy
from io import StringIO
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
import yaml
from olmo_core.utils import prepare_cli_environment
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


from tqdm import tqdm

from olmix.fit.constants import GroupedWandbMetrics, ObjectiveWeights
from olmix.fit.utils import (
    PROPOSER_TYPES,
    LogLinearRegressor,
    aggregate_mmlu,
    build_regression,
    calculate_priors_with_manual,
    compute_mixture_neighborhood,
    filter_constrained_swarm,
    get_output_dir,
    get_runs_from_api,
    mk_run_from_json,
    mk_run_metrics,
    mk_weights_from_config,
    plot_and_log_weights,
    plot_correlation,
    plot_interaction_matrix,
    plot_interaction_matrix_signed_evidence,
    save_eval_config,
    swarm_config_from_path,
)

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"


@click.group()
def cli():
    prepare_cli_environment()


@cli.command()
@click.option(
    "--experiment-groups",
    "-g",
    type=str,
    multiple=True,
    help="The group ID(s) to fit the regression model against",
    required=True,
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    multiple=True,
    help="Relative path to the experiment configuration file.",
    required=True,
)
@click.option(
    "-a",
    "--alpha",
    type=float,
    default=1.0,
    help="Alpha to apply to simulated distributions",
    required=False,
)
@click.option(
    "-A",
    "--group-average",
    type=str,
    help="The metric group to average for the regression task (regression will be fit against the average)",
    required=False,
    default=None,
)
@click.option(
    "-G",
    "--group-metrics",
    type=str,
    help="The metric group to fit regressions against (each metric will be fit separately)",
    default=None,
)
@click.option(
    "-s",
    "--num-samples",
    type=int,
    default=1,
    help="The number of evaluation samples per metric to collect from the run history",
    required=False,
)
@click.option(
    "-w",
    "--workspace",
    type=str,
    default=DEFAULT_WORKSPACE,
    help="The Wandb workspace to query for the runs",
    required=False,
)
@click.option(
    "-N",
    "--no-cache",
    is_flag=True,
    help="Do not use the cache for the runs",
    required=False,
    default=False,
)
@click.option(
    "-e",
    "--use-entropy",
    is_flag=True,
    help="Select highest entropy samples for simulation.",
    required=False,
    default=False,
)
@click.option(
    "-S",
    "--simulation-samples",
    type=int,
    default=100_000,
    help="Number of simulation samples to generate for each metric",
    required=False,
)
@click.option(
    "-r",
    "--regression-type",
    type=str,
    default="lightgbm",
    help="Whether to use LightGBM or linear regression for fitting",
    required=False,
)
@click.option(
    "-t",
    "--train-split",
    multiple=True,
    type=float,
    default=[1.0],
    help="Fraction of dataset/number of samples used for training. Default = 1.0 means that train equals test.",
    required=False,
)
@click.option(
    "--n-test",
    type=int,
    default=0,
    help="Number of test samples we evaluate regression model on, primarily used for analysis.",
    required=False,
)
@click.option(
    "--seed",
    type=int,
    default=0,
    help="Random state for train-test split",
    required=False,
)
@click.option(
    "--opt-avg-metric",
    is_flag=True,
    help="If set, each metric is fit separately, and then a mixture is selected to minimize the average of all metrics",
    required=False,
    default=False,
)
@click.option(
    "--proposer-type",
    type=str,
    help="Proposer type: simulation, search, or exact (for log linear only)",
    required=False,
    default="simulation",
)
@click.option(
    "--neighborhood",
    type=str,
    help="the training run display name that defines the neighborhood of subselected mixtures we regress on",
    required=False,
    default=None,
)
@click.option(
    "--final-cookbook-path",
    type=click.Path(exists=True),
    help="Path to cookbook config containing information about the full run for which the proposed mix is used (number of tokens, dataset being used)",
    required=False,
    default=None,
)
@click.option(
    "--manual-token-constraint-path",
    type=click.Path(exists=True),
    help="Rather than using the requested and available tokens specified by a cookbook config, use a manually specified config",
    required=False,
    default=None,
)
@click.option(
    "--repetition-factor",
    type=float,
    help="Adjusts the document repetition constraint; i.e., find an optimal mix when we are allowed to 2x each domain",
    required=False,
    default=1,
)
@click.option(
    "--constrain-swarm",
    is_flag=True,
    help="If set, we only use swarm runs that are unconstrained according to the final cookbook config.",
    required=False,
    default=False,
)
@click.option(
    "--constrain-objective",
    is_flag=True,
    help="If set, we produce a proposed mix that is unconstrained according to the final cookbook config.",
    required=False,
    default=False,
)
@click.option(
    "--obj-weights",
    type=str,
    help="The non-uniform weights used to average BPB over all tasks. If not set, uniform weights are used.",
    required=False,
    default=None,
)
@click.option(
    "--temperature",
    type=float,
    help="The temperature used to adjust the dirichlet prior in the simulation process. Closer to 0 = more uniform.",
    required=False,
    default=None,
)
@click.option(
    "--keep-sources",
    type=str,
    multiple=True,
    help="If set, we only use swarm runs that have nonzero weight on keep_sources for the regression.",
    required=False,
    default=None,
)
@click.option(
    "--early-stopping",
    type=float,
    help="The epsilon for early stopping",
    required=False,
    default=0.0,
)
@click.option(
    "--dro-reference-model-id",
    type=str,
    help="If we want to enforce pareto improvements, this is the id of the initial model we want to do better than",
    required=False,
    default=None,
)
@click.option(
    "--use-reference-model-predicted-scores",
    is_flag=True,
    help="If true, we use the predicted performance of the reference model, not the true performance",
    required=False,
    default=False,
)
@click.option(
    "--use-reference-model-as-search-prior",
    is_flag=True,
    help="If true, we center our proposal/simulation around the reference model weights",
    required=False,
    default=False,
)
@click.option(
    "--select-top-k-runs",
    type=float,
    help="If set, only use the metrics and ratios of the top k runs, where performance is the average BPB across all tasks",
    required=False,
    default=1.0,
)
@click.option(
    "--fixed-weight", type=str, help="string dict of domains and their weights to fix", required=False, default=None
)
@click.option(
    "--pull-from-dashboard",
    is_flag=True,
    help="if set, pull eval results from dashboard",
    required=False,
    default=False,
)
@click.option(
    "--dashboard",
    type=str,
    help="the dashboard where offline evals are stored",
    required=False,
    multiple=True,
    default=["regmixer"],
)
@click.option("--metric-type", type=str, help="the metric type to use for evaluation", required=False, default=None)
@click.option(
    "--use-cookbook",
    is_flag=True,
    help="if set, use a series of params designed for olmo-cookbook, not regmixer swarm",
    required=False,
    default=False,
)
@click.option(
    "--fit-only",
    is_flag=True,
    help="if set, only fit the regression model, do not propose a mix",
    required=False,
    default=False,
)
@click.option(
    "--custom-name", type=str, help="if set, use this custom name for the experiment", required=False, default=None
)
@click.option("--interactions", multiple=True, help="Feature interactions, like 1,2 ", type=str, default=None)
@click.option("--tol", type=float, help="Pareto constraint tolerance", default=None, required=False)
@click.option(
    "--fixed-search-weight",
    type=str,
    help="If set, this states that certain elements of our proposed mix must have a specific weight",
    required=False,
)
@click.option(
    "--support-domains",
    multiple=True,
    help="if set, we only select runs where the ratios on the support domains add to 1 ",
    type=str,
    default=None,
)
@click.option(
    "--drop-metrics",
    multiple=True,
    help="if set, we do not fit regression models on certain metrics",
    type=str,
    default=None,
)
@click.option(
    "--make-worst-mix",
    is_flag=True,
    help="if set, we invert the objective function and produce a bad mix (for counterfactual)",
    required=False,
    default=False,
)
@click.option(
    "--min-weight-per-domain",
    type=float,
    help="if set, we propose a mix where the minimum weight for each domain is above this threshold",
    required=False,
    default=0.0,
)
@click.option(
    "--use-hardcoded-reference-ratio",
    is_flag=True,
    help="if set, we use a hardcoded reference ratio for the mix",
    required=False,
    default=False,
)
@click.option(
    "--kl-reg",
    type=float,
    help="the lambda for KL regularization, only used for log-linear regression",
    required=False,
    default=None,
)
@click.option(
    "--patched",
    is_flag=True,
    help="if set, we need to patch multiple swarms. We just need this to hardcode an edge case (adding 'dclm:' prefix to the dclm only swarm)",
    required=False,
    default=False,
)
def fit(
    experiment_groups: list[str],
    config: list[pathlib.Path],
    alpha: float,
    num_samples: int,
    simulation_samples: int,
    group_average: str | None,
    group_metrics: str | None,
    workspace: str,
    no_cache: bool,
    use_entropy: bool,
    regression_type: str,
    train_split: tuple[float],
    n_test: int,
    seed: int,
    opt_avg_metric: bool,
    proposer_type: str,
    neighborhood: str | None,
    final_cookbook_path: Path | None,
    manual_token_constraint_path: Path | None,
    repetition_factor: float,
    constrain_swarm: bool,
    constrain_objective: bool,
    obj_weights: str | None,
    temperature: float | None,
    keep_sources: list[str] | None,
    dashboard: list[str],
    support_domains: tuple[str],
    drop_metrics: tuple[str],
    interactions: tuple[str],
    early_stopping: float = 0.0,
    dro_reference_model_id: str | None = None,
    use_reference_model_predicted_scores: bool = False,
    use_reference_model_as_search_prior: bool = False,
    select_top_k_runs: float = 1.0,
    fixed_weight: str | None = None,
    pull_from_dashboard: bool = False,
    metric_type: str | None = None,
    use_cookbook: bool = False,
    fit_only: bool = False,
    custom_name: str | None = None,
    tol: float | None = None,
    fixed_search_weight: str | None = None,
    make_worst_mix: bool = False,
    min_weight_per_domain: float = 0.0,
    use_hardcoded_reference_ratio: bool = False,
    kl_reg: float | None = None,
    patched: bool = False,
):
    output_dir = get_output_dir(experiment_groups)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    if group_average and group_metrics:
        raise ValueError("Cannot provide both group-average and group-metrics")

    if proposer_type == "search" and regression_type != "search":
        raise ValueError("Proposer type search only works with regression type search")

    if use_cookbook:
        workspace = "ai2-llm/olmo-cookbook"
        assert pull_from_dashboard, "If using olmo-cookbook, pull_from_dashboard must be set to True"

    eval_config = {
        "config": config[0] if len(config) == 1 else config,
        "alpha": alpha,
        "num_samples": num_samples,
        "simulation_samples": simulation_samples,
        "group_average": group_average,
        "group_metrics": group_metrics,
        "workspace": workspace,
        "regression_type": regression_type,
        "train_split": train_split[0] if len(train_split) == 1 else train_split,
        "n_test": n_test,
        "seed": seed,
        "opt_avg_metric": opt_avg_metric,
    }
    if proposer_type != "simulation":
        eval_config["proposer_type"] = proposer_type
    if neighborhood is not None:
        eval_config["neighborhood"] = neighborhood
    if constrain_swarm:
        eval_config["constrain_swarm"] = True
        eval_config["final_cookbook_path"] = final_cookbook_path
    if constrain_objective:
        eval_config["constrain_objective"] = True
        if final_cookbook_path is not None:
            eval_config["final_cookbook_path"] = final_cookbook_path
        elif manual_token_constraint_path is not None:
            eval_config["manual_token_constraint_path"] = manual_token_constraint_path

        if repetition_factor != 1:
            eval_config["repetition_factor"] = repetition_factor
    if obj_weights is not None:
        eval_config["obj_weights"] = obj_weights
    if temperature is not None:
        eval_config["temperature"] = temperature
    if len(keep_sources) != 0:
        eval_config["keep_sources"] = keep_sources
    if early_stopping > 0.0:
        eval_config["early_stopping"] = early_stopping
    if dro_reference_model_id is not None:
        eval_config["dro_reference_model_id"] = dro_reference_model_id
    if use_reference_model_predicted_scores:
        eval_config["use_reference_model_predicted_scores"] = use_reference_model_predicted_scores
    if use_reference_model_as_search_prior:
        eval_config["use_reference_model_as_search_prior"] = use_reference_model_as_search_prior
    if fixed_weight is not None:
        eval_config["fixed_weight"] = fixed_weight
        fixed_weight_dict = json.loads(fixed_weight)
    if pull_from_dashboard:
        eval_config["pull_from_dashboard"] = pull_from_dashboard
    if dashboard[0] != "regmixer":
        eval_config["dashboard"] = dashboard
    if metric_type is not None:
        eval_config["metric_type"] = metric_type
    if tol is not None:
        eval_config["tol"] = tol
    if fixed_search_weight is not None:
        eval_config["fixed_search_weight"] = fixed_search_weight
    if len(support_domains) != 0:
        eval_config["support_domains"] = support_domains
    if len(drop_metrics) != 0:
        eval_config["drop_metrics"] = drop_metrics
    if make_worst_mix:
        eval_config["make_worst_mix"] = True
    if min_weight_per_domain > 0.0:
        eval_config["min_weight_per_domain"] = min_weight_per_domain
    if use_hardcoded_reference_ratio:
        eval_config["use_hardcoded_reference_ratio"] = True
    if kl_reg is not None:
        assert proposer_type == "exact"
        eval_config["kl_reg"] = kl_reg

    # used for caching regression model
    regression_config = {
        "group_average": group_average,
        "group_metrics": group_metrics,
        "regression_type": regression_type,
        "train_split": train_split[0] if len(train_split) == 1 else train_split,
        "n_test": n_test,
        "seed": seed,
        "neighborhood": neighborhood,
        "constrain_swarm": constrain_swarm,
        "keep_sources": keep_sources,
        "early_stopping": early_stopping,
    }
    if select_top_k_runs < 1.0:
        eval_config["select_top_k_runs"] = select_top_k_runs
        regression_config["select_top_k_runs"] = select_top_k_runs
    if fixed_weight is not None:
        regression_config["fixed_weight"] = fixed_weight
    if metric_type is not None:
        regression_config["metric_type"] = metric_type
    if len(interactions) != 0:
        regression_config["interactions"] = interactions

    if len(support_domains) != 0:
        regression_config["support_domains"] = support_domains

    output_dir = save_eval_config(eval_config, output_dir, custom_name)

    api = wandb.Api()

    eval_metric_group = GroupedWandbMetrics.all_metrics
    eval_metric_group_name = eval_metric_group.name

    if group_average:
        eval_metric_group = GroupedWandbMetrics[group_average]
        eval_metric_group_name = f"avg_{group_average}"

    if group_metrics:
        eval_metric_group = GroupedWandbMetrics[group_metrics]
        eval_metric_group_name = group_metrics

    cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_runs_cache.json"
    )
    launch_configs = [swarm_config_from_path(c, use_cookbook) for c in config]
    full_group_names = [
        f"{launch_config.name}-{group}" for group, launch_config in zip(experiment_groups, launch_configs)
    ]
    if no_cache:
        logger.info("Cache disabled, will not use cache for run samples...")
        run_instances = get_runs_from_api(
            api, workspace, full_group_names, cache_path, no_cache, num_samples, eval_metric_group
        )
    else:
        try:
            # TODO: Add partitioned cache per group maybe?
            with open(cache_path) as f:
                run_dict = json.load(f)
                run_instances = [mk_run_from_json(run) for run in run_dict]
            logger.info(f"Loaded cached runs from {cache_path}")

        except FileNotFoundError:
            logger.warning(f"Failed to load cache from {cache_path}, fetching runs from API...")
            run_instances = get_runs_from_api(
                api, workspace, full_group_names, cache_path, no_cache, num_samples, eval_metric_group
            )

    # Filter out failed runs or runs without evals
    # run_instances = [run for run in run_instances if run.samples.shape[0] > 0]

    logger.info(f"Found {len(run_instances)} runs in {workspace} that match group id filter {experiment_groups}...")

    logger.info("Calculating source weights...")
    priors, original_priors = calculate_priors_with_manual(
        source_configs=launch_configs[0].dataset.sources if use_cookbook else launch_configs[0].sources,
        dtype=launch_configs[0].dataset.dtype if use_cookbook else launch_configs[0].dtype,
        use_cache=(not no_cache),
        manual_prior=launch_configs[0].manual_prior if hasattr(launch_configs[0], "manual_prior") else None,
        fixed_source_weights=launch_configs[0].fixed_source_weights
        if hasattr(launch_configs[0], "fixed_source_weights")
        else None,
    )

    if fixed_weight is not None:
        # remove the fixed weight domains from the priors, and renormalize the remaining domains to add to 1
        new_priors = {k: v for k, v in priors[0].items() if k not in fixed_weight_dict}
        total = sum(list(new_priors.values()))
        new_priors = {k: v / total for k, v in new_priors.items()}  # normalize the weights
        # hack to update the tuple
        priors_list = list(priors)
        priors_list[0] = new_priors
        priors = tuple(priors_list)

    logger.info("Source weights:")
    logger.info(priors[0])

    ratios_cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_ratios.pkl"
    )
    metrics_cache_path = (
        pathlib.Path(BASE_CACHE_DIR) / f"{'_'.join(experiment_groups)}_{eval_metric_group_name}_metrics.pkl"
    )
    if os.path.exists(ratios_cache_path) and os.path.exists(metrics_cache_path):
        logger.info(f"Loading cached ratios and metrics from {ratios_cache_path} and {metrics_cache_path}")
        with open(ratios_cache_path, "rb") as f:
            ratios = pd.read_pickle(f)
        with open(metrics_cache_path, "rb") as f:
            metrics = pd.read_pickle(f)
        ratios = ratios[ratios["run"].isin(metrics.run)]
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
        if pull_from_dashboard:
            all_dashboard_results = pd.DataFrame()
            for d in dashboard:
                logger.info(f"Pulling results from dashboard {d}...")
                command = [
                    "olmo-cookbook-eval",
                    "results",
                    "--dashboard",
                    f"{d}",
                ]
                for task in eval_metric_group.value:
                    command.append("--tasks")
                    command.append(task)
                command.extend(["--format", "csv", "--skip-on-fail"])
                result = subprocess.run(command, capture_output=True, text=True)
                # Check for errors
                if result.returncode != 0:
                    print("Error:", result.stderr)
                else:
                    # Load CSV content into a DataFrame
                    csv_data = result.stdout
                    df = pd.read_csv(StringIO(csv_data))
                    all_dashboard_results = pd.concat([all_dashboard_results, df], ignore_index=True)

            run_metrics = []
            for idx, run in tqdm(enumerate(run_instances)):
                # Filter the dashboard results
                matched = all_dashboard_results[
                    all_dashboard_results["name"].str.contains(re.escape(run.display_name), regex=True)
                ]

                if matched.empty:
                    logger.warning(f"No matching results found for run {run.display_name}")
                    continue

                try:
                    metrics = {k: next(iter(v.values())) for k, v in matched[eval_metric_group.value].to_dict().items()}
                except StopIteration:
                    logger.warning(f"Empty values found when parsing metrics for {run.display_name}")
                    continue

                run_metrics.append(
                    {
                        "run": run.id,
                        "name": run.display_name,
                        "index": idx,
                        **metrics,
                    }
                )

        else:
            run_metrics = [
                {
                    "run": run.id,
                    "name": run.display_name,
                    "index": idx,
                    **mk_run_metrics(
                        history=run.samples,
                        samples=num_samples,
                        metrics=(eval_metric_group_name, eval_metric_group.value),
                        display_name=run.display_name
                        if experiment_groups[0] != "ee22e17f"
                        else run.display_name.replace("all-dressed", "dclmv2"),
                        average=group_average is not None,
                        pull_from_dashboard=pull_from_dashboard,
                        dashboard=dashboard,
                        metric_type=metric_type,
                    ),
                }
                for idx, run in tqdm(enumerate(run_instances))
                if eval_metric_group_name
                in [
                    "superswarm_offline",
                    "olmo3_offline_tasks",
                    "pdf_tasks",
                    "code_tasks_offline",
                    "code_tasks_offline_fixed",
                    "olmo3_offline_tasks_0630",
                    "midtraining_aggregate_evals",
                    "midtraining_finegrained_evals",
                    "pretraining_tasks_for_paper",
                    "math_tasks",
                    "code_tasks_new",
                    "qa_tasks",
                ]
                or len(run.samples) > 0
            ]

        if constrain_swarm:
            raise NotImplementedError(
                "Constrained swarm is implemented but out of date. We concluded that this is not the right way to enforce token repetition constraints."
            )
            run_ratios, run_metrics = filter_constrained_swarm(final_cookbook_path, run_ratios, run_metrics)

        ratios = pd.DataFrame(run_ratios)
        metrics = pd.DataFrame(run_metrics)
        numerical_cols = metrics.columns[3:]
        metrics[numerical_cols] = metrics[numerical_cols].apply(pd.to_numeric, errors="coerce")
        ratios = ratios[ratios["run"].isin(metrics.run)]

        if len(support_domains) == 0 and len(train_split) == 1:
            assert np.isclose(ratios[ratios.columns[3:]].sum(axis=1).sum(), len(ratios)), "Ratios do not add up to 1!"
        if fixed_weight is not None:
            # normalize the non-fixed-weight domains to add to 1
            domains = ratios.columns[3:]
            ratios[domains] = ratios[domains].div(ratios[domains].sum(axis=1), axis=0)

        pd.to_pickle(ratios, ratios_cache_path)
        pd.to_pickle(metrics, metrics_cache_path)
        logger.info(f"Saved ratios to {ratios_cache_path} and metrics to {metrics_cache_path}")

    metrics_to_index = eval_metric_group.value
    if len(support_domains) != 0:
        # only keep ratios/
        keep_idxs = np.where(np.isclose(ratios[list(support_domains)].sum(axis=1), 1))[0]
        ratios = ratios.iloc[keep_idxs]
        drop_col = list(set(ratios.columns[3:]).difference(set(support_domains)))
        ratios = ratios.drop(columns=drop_col)
        metrics = metrics.iloc[keep_idxs]

        new_priors = {k: v for k, v in priors[0].items() if k in list(support_domains)}
        total = sum(list(new_priors.values()))
        new_priors = {k: v / total for k, v in new_priors.items()}
        # hack to update prior tuple
        priors_list = list(priors)
        priors_list[0] = new_priors
        priors = tuple(priors_list)

    if group_average:
        metrics_to_index = [eval_metric_group_name]

    if all("mmlu_stem" not in s for s in metrics.columns) and any("mmlu" in s for s in metrics.columns):
        metrics, metrics_to_index = aggregate_mmlu(metrics, metrics_to_index)

    if len(ratios[ratios.columns[3:]]) > len(ratios):
        raise ValueError("The number of swarm runs is fewer than the number of mixing sources.")

    if len(keep_sources) != 0:
        old_len = len(ratios)
        other_columns = list(set(ratios.columns[3:]).difference(set(keep_sources)))
        ratios = ratios[
            ratios[list(keep_sources)].ne(0).all(axis=1)  # all specified columns nonzero
            & ratios[other_columns].eq(0).all(axis=1)
        ]
        logger.info(f"Filtered out {old_len - len(ratios)} runs that were not only on {keep_sources}")
        metrics = metrics[metrics["name"].isin(ratios["name"])]
        ratios.drop(columns=other_columns, inplace=True)

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

    """ cols_with_nans = metrics[cols_to_check].columns[metrics[cols_to_check].isna().any()].tolist()
    if len(cols_with_nans) > 0:
        logger.warning(f"Found NaNs in the following columns, dropping them! {cols_with_nans}")
        metrics = metrics.drop(columns=cols_with_nans)
        metrics_to_index = [m for m in metrics_to_index if m not in cols_with_nans] """

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

        train_split = [int(t) if t > 1 else t for t in train_split]

        if neighborhood is None:
            # we IID subselect training data

            if len(train_split) > 1:
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

    if obj_weights:
        obj_weights = ObjectiveWeights[obj_weights]
        obj_weights = [obj_weights.value.get(metric, 1) for idx, metric in indexed_metrics]
        logger.info(f"Minimizing weighted average: {obj_weights}")

    """ if regression_type=="lightgbm":
        # debugging - just fit on first metric
        drop_indices = np.arange(4, len(metrics.columns))
        metrics = metrics.drop(columns=metrics.columns[drop_indices])
        Y_train = np.delete(Y_train, drop_indices-3, axis=1)
        Y_test = np.delete(Y_test, drop_indices-3, axis=1)
        metrics_to_index = [m for i, m in indexed_metrics if i not in drop_indices-3]
        indexed_metrics = list(enumerate(metrics_to_index))
    """

    # caching logic for regression model. Note that one regression model can be used for many different proposed mixes,
    # which is why we need to cache based on a separate subconfig, regression_config
    regression_config_str = json.dumps(regression_config, sort_keys=True)
    hash_str = hashlib.sha256(regression_config_str.encode("utf-8")).hexdigest()[:16]
    regression_model_cache_folder = pathlib.Path(BASE_CACHE_DIR) / "_".join(experiment_groups) / hash_str
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
            predictors.append(build_regression(idx, Y_train, X_train, regression_type, early_stopping, interactions))
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
            reference_model_run_instance = get_runs_from_api(
                api, workspace, [dro_reference_model_id], cache_path, True, num_samples, eval_metric_group
            )[0]

            if use_reference_model_predicted_scores:
                # get reference model's mix and pass this through the regression model
                reference_run_ratio = {
                    "run": reference_model_run_instance.id,
                    "name": reference_model_run_instance.display_name,
                    "index": 0,
                    **mk_weights_from_config(reference_model_run_instance.config, priors),
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
                        metrics=(eval_metric_group_name, eval_metric_group.value),
                        display_name=reference_model_run_instance.display_name,
                        average=group_average is not None,
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
                final_cookbook_path=final_cookbook_path,
                manual_token_constraint_path=manual_token_constraint_path,
                repetition_factor=repetition_factor,
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
                kl_reg=kl_reg,
            )

            results.append((metric, weights))

    if fit_only:
        logger.info("Fit only mode, not proposing a mix.")
        return

    if opt_avg_metric and n_test == 0:
        assert group_metrics is not None and group_average is None  # need to have this set

        if experiment_groups[0] in ["5c712b3b", "daf37f03", "f5a3ff58"] and use_hardcoded_reference_ratio:
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
            final_cookbook_path=final_cookbook_path,
            manual_token_constraint_path=manual_token_constraint_path,
            repetition_factor=repetition_factor,
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
            metric_name=group_metrics,
            regression_type=regression_type,
            train_split=train_split,
            n_test=n_test,
            split_seed=seed,
            n_samples=num_samples,
            alpha=alpha,
            df_config=ratios,
            output_dir=output_dir,
            fixed_weight=fixed_weight_dict if fixed_weight is not None else None,
        )

        results.append((group_metrics, weights))

    elif not group_average and n_test == 0:
        # If we're not optimizing for the average of the metric group, then we average the reweighted distributions after fitting
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


if __name__ == "main":
    cli(obj={})
