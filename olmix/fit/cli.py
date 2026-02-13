import json
import logging
import os
import pathlib
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

import click

warnings.filterwarnings("ignore", category=UserWarning)


from olmix.fit.constants import ALL_WANDB_METRICS
from olmix.fit.core import run_fit
from olmix.fit.loaders import load_from_csv, load_from_wandb, load_priors_from_json
from olmix.fit.utils import (
    calculate_priors_with_manual,
    get_output_dir,
    save_fit_config,
    swarm_config_from_path,
)
from olmix.plots import BASE_OUTPUT_DIR

logger = logging.getLogger(__name__)

BASE_CACHE_DIR = "cache/"


@dataclass
class FitConfig:
    """Configuration capturing all CLI parameters."""

    # Required/base fields (always included)
    config: str | list[str] | None
    alpha: float
    simulation_samples: int
    workspace: str
    regression_type: str
    n_test: int
    seed: int
    opt_avg_metric: bool

    # Optional fields with defaults (only included if non-default)
    from_csv: str | None = None
    from_wandb: str | None = None
    priors: str | None = None
    proposer_type: str = "simulation"
    constrain_objective: bool = False
    target_tokens: int | None = None
    repetition_factor: float | None = None
    obj_weights: str | None = None
    temperature: float | None = None
    keep_sources: list[str] = field(default_factory=list)
    early_stopping: float = 0.0
    fixed_weight: str | None = None
    dashboard: tuple[str, ...] | None = None
    support_domains: tuple[str, ...] = field(default_factory=tuple)
    drop_metrics: tuple[str, ...] = field(default_factory=tuple)
    make_worst_mix: bool = False
    kl_reg: float | None = None
    requested_tokens: int | None = None
    use_natural_kl: bool = False
    test_paths: str | None = None
    aggregate_task_families: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict, filtering out None values and empty/default collections."""
        result = {}

        for key, value in asdict(self).items():
            # Always include base fields
            if key in {
                "config",
                "alpha",
                "simulation_samples",
                "workspace",
                "regression_type",
                "n_test",
                "seed",
                "opt_avg_metric",
            }:
                result[key] = value
            # Include optional fields only if they have meaningful values
            elif value is not None and value is not False and value != () and value != [] and value != 0.0:
                # Special case: only include proposer_type if not default "simulation"
                if key == "proposer_type" and value == "simulation":
                    continue
                # Special case: only include dashboard if not default ("regmixer",)
                if key == "dashboard" and value == ("regmixer",):
                    continue
                result[key] = value
            # Include boolean flags explicitly if True
            elif isinstance(value, bool) and value is True:
                result[key] = value

        return result


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
    "--fixed-weight", type=str, help="string dict of domains and their weights to fix", required=False, default=None
)
@click.option(
    "--dashboard",
    type=str,
    help="the dashboard where offline evals are stored",
    required=False,
    multiple=True,
    default=["regmixer"],
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
    "--kl-reg",
    type=float,
    help="the lambda for KL regularization with exact solver",
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
@click.option(
    "--requested-tokens",
    type=int,
    help="if --constrain-objective and --manual-token-constraint-path are set, this overrides the number of requested tokens to use in the constraint",
    required=False,
    default=None,
)
@click.option(
    "--use-natural-kl",
    is_flag=True,
    help="if set to true and we use an exact solver, the reference distribution for the KL penalty will be the natural distribution, not the prior (which could be manually set).",
    required=False,
    default=False,
)
@click.option(
    "--test-ratios-path",
    type=str,
    multiple=True,
    help="paths to ratios of held out mixtures to evaluate fit on.",
    required=False,
    default=[],
)
@click.option(
    "--test-metrics-path",
    type=str,
    multiple=True,
    help="paths to metrics of held out mixtures to evaluate fit on.",
    required=False,
    default=[],
)
@click.option(
    "--aggregate-task-families",
    is_flag=True,
    help="if set to true, we fit one model per task family (math, code, qa)",
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
    simulation_samples: int,
    workspace: str,
    no_cache: bool,
    regression_type: str,
    train_split: tuple[float, ...],
    n_test: int,
    seed: int,
    opt_avg_metric: bool,
    proposer_type: str,
    constrain_objective: bool,
    obj_weights: str | None,
    temperature: float | None,
    keep_sources: list[str] | None,
    dashboard: list[str],
    support_domains: tuple[str],
    drop_metrics: tuple[str],
    early_stopping: float = 0.0,
    fixed_weight: str | None = None,
    fit_only: bool = False,
    custom_name: str | None = None,
    make_worst_mix: bool = False,
    kl_reg: float | None = None,
    patched: bool = False,
    requested_tokens: int | None = None,
    use_natural_kl: bool = False,
    test_ratios_path: tuple[str, ...] = (),
    test_metrics_path: tuple[str, ...] = (),
    aggregate_task_families: bool = False,
):
    # Validate data source flags
    priors_file = priors  # rename to avoid shadowing the priors data variable

    if from_csv and from_wandb:
        raise click.UsageError("--from-csv and --from-wandb are mutually exclusive")
    if not from_csv and not from_wandb:
        raise click.UsageError("Must specify either --from-csv or --from-wandb")
    if from_csv and not config and not priors_file:
        raise click.UsageError("--from-csv requires either --config or --priors")

    if proposer_type == "search" and regression_type != "search":
        raise ValueError("Proposer type search only works with regression type search")

    fixed_weight_dict = json.loads(fixed_weight) if fixed_weight is not None else None

    # Load data
    if from_wandb:
        if experiment_groups:
            logger.warning("--experiment-groups is ignored when using --from-wandb (auto-resolved from metadata JSON)")

        ratios, metrics, launch_configs, priors_data, original_priors_data, experiment_groups_list = load_from_wandb(
            from_wandb,
            workspace=workspace,
            no_cache=no_cache,
            dashboard=list(dashboard),
            patched=patched,
            fixed_weight_dict=fixed_weight_dict,
        )
        eval_metrics: list[str] | None = ALL_WANDB_METRICS
        natural_kl: tuple | None = None
    else:
        assert from_csv is not None
        ratios, metrics = load_from_csv(from_csv)

        experiment_groups_list = list(experiment_groups) if experiment_groups else []

        # Load or calculate priors
        if priors_file:
            priors_data = load_priors_from_json(priors_file)
            from copy import deepcopy

            original_priors_data = deepcopy(priors_data)
            launch_configs = [swarm_config_from_path(c) for c in config] if config else []
        else:
            launch_configs = [swarm_config_from_path(c) for c in config]
            priors_data, original_priors_data = calculate_priors_with_manual(
                source_configs=launch_configs[0].sources,
                dtype=launch_configs[0].dtype,
                use_cache=(not no_cache),
                manual_prior=launch_configs[0].manual_prior if hasattr(launch_configs[0], "manual_prior") else None,
                fixed_source_weights=launch_configs[0].fixed_source_weights
                if hasattr(launch_configs[0], "fixed_source_weights")
                else None,
            )

        if use_natural_kl and proposer_type == "exact":
            logger.info("Calculating natural source weights for KL regularization...")
            natural_kl, _ = calculate_priors_with_manual(
                source_configs=launch_configs[0].sources,
                dtype=launch_configs[0].dtype,
                use_cache=(not no_cache),
                manual_prior=None,
                fixed_source_weights=launch_configs[0].fixed_source_weights
                if hasattr(launch_configs[0], "fixed_source_weights")
                else None,
            )
        else:
            natural_kl = None

        if fixed_weight_dict is not None:
            new_priors = {k: v for k, v in priors_data[0].items() if k not in fixed_weight_dict}
            total = sum(new_priors.values())
            new_priors = {k: v / total for k, v in new_priors.items()}
            priors_data[0].clear()
            priors_data[0].update(new_priors)

        logger.info("Source weights:")
        logger.info(priors_data[0])

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
    experiment_groups_list = None
    if experiment_groups_list:
        full_group_names = [
            f"{launch_config.name}-{group}" for group, launch_config in zip(experiment_groups_list, launch_configs)
        ]

    # Prepare values for FitConfig
    config_value = str(config[0]) if len(config) == 1 else [str(c) for c in config] if config else None
    dashboard_value = tuple(dashboard) if dashboard[0] != "regmixer" else None
    test_paths_value = (
        "_".join([tr.split("/")[-1].split("_")[0] for tr in test_ratios_path])
        if test_ratios_path and test_metrics_path
        else None
    )

    # Handle constrain_objective special case
    target_tokens_value = None
    repetition_factor_value = None
    if constrain_objective:
        swarm_config = launch_configs[0]
        target_tokens_value = swarm_config.get_target_tokens()
        repetition_factor_value = swarm_config.repetition_factor

    # Validate kl_reg constraint
    if kl_reg is not None:
        assert proposer_type == "exact", "kl_reg requires proposer_type='exact'"

    # Create FitConfig dataclass
    fit_config_obj = FitConfig(
        config=config_value,
        alpha=alpha,
        simulation_samples=simulation_samples,
        workspace=workspace,
        regression_type=regression_type,
        n_test=n_test,
        seed=seed,
        opt_avg_metric=opt_avg_metric,
        from_csv=from_csv,
        from_wandb=from_wandb,
        priors=priors_file,
        proposer_type=proposer_type,
        constrain_objective=constrain_objective,
        target_tokens=target_tokens_value,
        repetition_factor=repetition_factor_value,
        obj_weights=obj_weights,
        temperature=temperature,
        keep_sources=list(keep_sources) if keep_sources else [],
        early_stopping=early_stopping,
        fixed_weight=fixed_weight,
        dashboard=dashboard_value,
        support_domains=support_domains,
        drop_metrics=drop_metrics,
        make_worst_mix=make_worst_mix,
        kl_reg=kl_reg,
        requested_tokens=requested_tokens,
        use_natural_kl=use_natural_kl,
        test_paths=test_paths_value,
        aggregate_task_families=aggregate_task_families,
    )

    # Convert to dict, filtering out None/default values
    fit_config = fit_config_obj.to_dict()

    output_dir = save_fit_config(fit_config, output_dir, custom_name)

    # ── Run fit ──────────────────────────────────────────────────────────
    run_fit(
        ratios,
        metrics,
        priors_data,
        original_priors_data,
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
        opt_avg_metric=opt_avg_metric,
        proposer_type=proposer_type,
        simulation_samples=simulation_samples,
        constrain_objective=constrain_objective,
        temperature=temperature,
        keep_sources=keep_sources,
        support_domains=support_domains,
        drop_metrics=drop_metrics,
        fixed_weight=fixed_weight,
        alpha=alpha,
        fit_only=fit_only,
        make_worst_mix=make_worst_mix,
        kl_reg=kl_reg,
        workspace=workspace,
        requested_tokens=requested_tokens,
        natural_kl=natural_kl,
        test_ratios_path=test_ratios_path,
        test_metrics_path=test_metrics_path,
        aggregate_task_families=aggregate_task_families,
    )
