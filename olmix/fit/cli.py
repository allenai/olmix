import json
import logging
import os
import pathlib
import warnings

import click

warnings.filterwarnings("ignore", category=UserWarning)


from olmix.fit.constants import ALL_WANDB_METRICS
from olmix.fit.core import run_fit
from olmix.fit.loaders import load_from_csv, load_from_wandb, load_priors_from_json
from olmix.fit.utils import (
    calculate_priors_with_manual,
    get_output_dir,
    save_eval_config,
    swarm_config_from_path,
)
from olmix.plots import BASE_OUTPUT_DIR

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"
DEFAULT_WORKSPACE = "ai2-llm/regmixer"


@click.command()
@click.option(
    "--from-csv",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to directory containing ratios.csv and metrics.csv",
)
@click.option(
    "--from-wandb",
    type=click.Path(exists=True, file_okay=False),
    default=None,
    help="Path to launch output directory containing metadata JSON (auto-resolves group_id and config)",
)
@click.option(
    "--experiment-groups",
    "-g",
    type=str,
    multiple=True,
    help="The group ID(s) to fit the regression model against (auto-resolved when using --from-wandb)",
)
@click.option(
    "--priors",
    type=click.Path(exists=True),
    default=None,
    help="Path to JSON file containing prior distribution (e.g., from launch output metadata). "
    "Avoids S3 access for prior calculation when using --from-csv.",
)
@click.option(
    "-c",
    "--config",
    type=click.Path(exists=True),
    multiple=True,
    help="Relative path to the experiment configuration file. Required with --from-csv unless --priors is given.",
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
    default="log_linear",
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
    default="exact",
)
@click.option(
    "--neighborhood",
    type=str,
    help="the training run display name that defines the neighborhood of subselected mixtures we regress on",
    required=False,
    default=None,
)
@click.option(
    "--constrain-objective",
    is_flag=True,
    help="If set, we produce a proposed mix that is unconstrained according to the final cookbook config.",
    required=False,
    default=False,
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
    from_csv: str | None,
    from_wandb: str | None,
    priors: str | None,
    experiment_groups: tuple[str, ...],
    config: tuple[str, ...],
    alpha: float,
    num_samples: int,
    simulation_samples: int,
    workspace: str,
    no_cache: bool,
    use_entropy: bool,
    regression_type: str,
    train_split: tuple[float, ...],
    n_test: int,
    seed: int,
    opt_avg_metric: bool,
    proposer_type: str,
    neighborhood: str | None,
    constrain_objective: bool,
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
    # ── Validate data source flags ───────────────────────────────────────
    priors_file = priors  # rename to avoid shadowing the priors data variable

    if from_csv and from_wandb:
        raise click.UsageError("--from-csv and --from-wandb are mutually exclusive")
    if not from_csv and not from_wandb:
        raise click.UsageError("Must specify either --from-csv or --from-wandb")
    if from_csv and not config and not priors_file:
        raise click.UsageError("--from-csv requires either --config or --priors")

    if proposer_type == "search" and regression_type != "search":
        raise ValueError("Proposer type search only works with regression type search")

    if use_cookbook:
        workspace = "ai2-llm/olmo-cookbook"
        assert pull_from_dashboard, "If using olmo-cookbook, pull_from_dashboard must be set to True"

    fixed_weight_dict = json.loads(fixed_weight) if fixed_weight is not None else None

    # ── Load data ────────────────────────────────────────────────────────
    if from_wandb:
        if experiment_groups:
            logger.warning("--experiment-groups is ignored when using --from-wandb (auto-resolved from metadata JSON)")

        ratios, metrics, launch_configs, priors, original_priors, experiment_groups_list = load_from_wandb(
            from_wandb,
            workspace=workspace,
            num_samples=num_samples,
            no_cache=no_cache,
            use_cookbook=use_cookbook,
            pull_from_dashboard=pull_from_dashboard,
            dashboard=list(dashboard),
            metric_type=metric_type,
            patched=patched,
            fixed_weight_dict=fixed_weight_dict,
        )
        eval_metrics: list[str] | None = ALL_WANDB_METRICS
    else:
        assert from_csv is not None
        ratios, metrics = load_from_csv(from_csv)

        experiment_groups_list = list(experiment_groups) if experiment_groups else []

        # Load or calculate priors
        if priors_file:
            loaded_priors = load_priors_from_json(priors_file)
            from copy import deepcopy

            priors = loaded_priors
            original_priors = deepcopy(loaded_priors)
            launch_configs = [swarm_config_from_path(c, use_cookbook) for c in config] if config else []
        else:
            launch_configs = [swarm_config_from_path(c, use_cookbook) for c in config]
            priors, original_priors = calculate_priors_with_manual(
                source_configs=launch_configs[0].dataset.sources if use_cookbook else launch_configs[0].sources,
                dtype=launch_configs[0].dataset.dtype if use_cookbook else launch_configs[0].dtype,
                use_cache=(not no_cache),
                manual_prior=launch_configs[0].manual_prior if hasattr(launch_configs[0], "manual_prior") else None,
                fixed_source_weights=launch_configs[0].fixed_source_weights
                if hasattr(launch_configs[0], "fixed_source_weights")
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

        # Validate ratios sum to ~1.0 (CSV data was already validated in loader,
        # but apply fixed_weight normalization here if needed)
        if fixed_weight_dict is not None:
            domains = ratios.columns[3:]
            ratios[domains] = ratios[domains].div(ratios[domains].sum(axis=1), axis=0)

        eval_metrics = None  # derived from CSV columns in run_fit

    # ── Setup output directory ───────────────────────────────────────────
    if experiment_groups_list:
        output_dir = get_output_dir(experiment_groups_list)
    else:
        csv_name = os.path.basename(os.path.normpath(from_csv)) if from_csv else "unknown"
        output_dir = f"{BASE_OUTPUT_DIR}csv_{csv_name}/"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(BASE_CACHE_DIR).mkdir(parents=True, exist_ok=True)

    # ── Validate constrain_objective ─────────────────────────────────────
    if constrain_objective:
        swarm_config = launch_configs[0]
        target_tokens = swarm_config.get_target_tokens()
        if target_tokens is None:
            raise click.UsageError(
                "For --constrain-objective, set target_tokens or target_chinchilla_multiple in config"
            )

    # ── Build and save eval config ───────────────────────────────────────
    full_group_names = None
    if experiment_groups_list:
        full_group_names = [
            f"{launch_config.name}-{group}" for group, launch_config in zip(experiment_groups_list, launch_configs)
        ]

    eval_config: dict = {
        "config": str(config[0]) if len(config) == 1 else [str(c) for c in config] if config else None,
        "alpha": alpha,
        "num_samples": num_samples,
        "simulation_samples": simulation_samples,
        "workspace": workspace,
        "regression_type": regression_type,
        "train_split": train_split[0] if len(train_split) == 1 else train_split,
        "n_test": n_test,
        "seed": seed,
        "opt_avg_metric": opt_avg_metric,
    }
    if from_csv:
        eval_config["from_csv"] = from_csv
    if from_wandb:
        eval_config["from_wandb"] = from_wandb
    if priors_file:
        eval_config["priors"] = priors_file
    if proposer_type != "simulation":
        eval_config["proposer_type"] = proposer_type
    if neighborhood is not None:
        eval_config["neighborhood"] = neighborhood
    if constrain_objective:
        eval_config["constrain_objective"] = True
        swarm_config = launch_configs[0]
        target_tokens = swarm_config.get_target_tokens()
        eval_config["target_tokens"] = target_tokens
        eval_config["repetition_factor"] = swarm_config.repetition_factor
    if temperature is not None:
        eval_config["temperature"] = temperature
    if keep_sources:
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
    if select_top_k_runs < 1.0:
        eval_config["select_top_k_runs"] = select_top_k_runs

    output_dir = save_eval_config(eval_config, output_dir, custom_name)

    # ── Run fit ──────────────────────────────────────────────────────────
    run_fit(
        ratios,
        metrics,
        priors,
        original_priors,
        output_dir,
        eval_metrics=eval_metrics,
        experiment_groups=experiment_groups_list if experiment_groups_list else None,
        launch_configs=launch_configs,
        full_group_names=full_group_names,
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        seed=seed,
        early_stopping=early_stopping,
        interactions=interactions,
        opt_avg_metric=opt_avg_metric,
        proposer_type=proposer_type,
        simulation_samples=simulation_samples,
        constrain_objective=constrain_objective,
        temperature=temperature,
        neighborhood=neighborhood,
        keep_sources=keep_sources,
        support_domains=support_domains,
        drop_metrics=drop_metrics,
        select_top_k_runs=select_top_k_runs,
        fixed_weight=fixed_weight,
        metric_type=metric_type,
        dro_reference_model_id=dro_reference_model_id,
        use_reference_model_predicted_scores=use_reference_model_predicted_scores,
        use_reference_model_as_search_prior=use_reference_model_as_search_prior,
        use_hardcoded_reference_ratio=use_hardcoded_reference_ratio,
        alpha=alpha,
        num_samples=num_samples,
        fit_only=fit_only,
        tol=tol,
        fixed_search_weight=fixed_search_weight,
        make_worst_mix=make_worst_mix,
        min_weight_per_domain=min_weight_per_domain,
        kl_reg=kl_reg,
        workspace=workspace,
    )
