import hashlib
import json
import logging
import os
import pathlib
import warnings

import click

warnings.filterwarnings("ignore", category=UserWarning)

from olmix.fit.config import FitConfig
from olmix.fit.core import run_fit
from olmix.fit.loaders import load_from_csv

logger = logging.getLogger(__name__)


def _save_fit_config(cfg: FitConfig, output_dir: str) -> str:
    """Serialize FitConfig to JSON inside a hashed subdirectory of *output_dir*."""
    config_str = json.dumps(cfg.model_dump(), sort_keys=True, default=str)
    hash_str = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]
    folder_path = os.path.join(output_dir, hash_str)
    os.makedirs(folder_path, exist_ok=True)
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg.model_dump(), f, indent=2, default=str)
    logger.info(f"Saved config to {config_path}")
    return folder_path


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the YAML fit configuration file.",
)
@click.option(
    "--output-dir",
    "output_dir_arg",
    type=click.Path(),
    required=True,
    help="Directory for saving fit outputs.",
)
def fit(config_path: str, output_dir_arg: str):
    cfg = FitConfig.from_yaml(config_path)

    # ── Load CSVs ─────────────────────────────────────────────────────────
    ratios, metrics = load_from_csv(cfg.swarm.ratios, cfg.swarm.metrics)

    # ── Priors ────────────────────────────────────────────────────────────
    priors_data = cfg.priors.to_tuple()
    from copy import deepcopy

    original_priors_data = deepcopy(priors_data)

    # Apply fixed_weight adjustments to priors and ratios
    fixed_weight_dict = cfg.filtering.fixed_weight if cfg.filtering.fixed_weight else None
    if fixed_weight_dict:
        new_priors = {k: v for k, v in priors_data[0].items() if k not in fixed_weight_dict}
        total = sum(new_priors.values())
        new_priors = {k: v / total for k, v in new_priors.items()}
        priors_data[0].clear()
        priors_data[0].update(new_priors)

        domains = ratios.columns[3:]
        ratios[domains] = ratios[domains].div(ratios[domains].sum(axis=1), axis=0)

    logger.info("Source weights:")
    logger.info(priors_data[0])

    # ── Validate constraints ─────────────────────────────────────────────
    if cfg.constraints.enabled and cfg.constraints.target_tokens is None:
        raise click.UsageError("constraints.enabled requires constraints.target_tokens")

    # ── Validate proposer ─────────────────────────────────────────────────
    if cfg.proposer.type == "search" and cfg.regression.type != "search":
        raise click.UsageError("proposer.type 'search' only works with regression.type 'search'")
    if cfg.proposer.kl_reg is not None and cfg.proposer.type != "exact":
        raise click.UsageError("proposer.kl_reg requires proposer.type 'exact'")

    # ── Output directory ──────────────────────────────────────────────────
    pathlib.Path(output_dir_arg).mkdir(parents=True, exist_ok=True)
    output_dir = _save_fit_config(cfg, output_dir_arg)

    # ── Convert fixed_weight to JSON string for run_fit ───────────────────
    fixed_weight_str = json.dumps(fixed_weight_dict) if fixed_weight_dict else None

    # ── Run fit ───────────────────────────────────────────────────────────
    run_fit(
        ratios,
        metrics,
        priors_data,
        original_priors_data,
        output_dir,
        eval_metrics=None,
        experiment_groups=None,
        launch_configs=None,
        full_group_names=None,
        regression_type=cfg.regression.type,
        train_split=tuple(cfg.regression.train_split),
        n_test=cfg.regression.n_test,
        seed=cfg.regression.seed,
        early_stopping=0.0,
        proposer_type=cfg.proposer.type,
        constrain_objective=cfg.constraints.enabled,
        temperature=cfg.proposer.temperature,
        keep_sources=cfg.filtering.keep_sources or None,
        support_domains=tuple(cfg.filtering.support_domains),
        drop_metrics=tuple(cfg.filtering.drop_metrics),
        fixed_weight=fixed_weight_str,
        fit_only=cfg.proposer.fit_only,
        make_worst_mix=cfg.proposer.make_worst_mix,
        kl_reg=cfg.proposer.kl_reg,
        workspace="",
        target_tokens=cfg.constraints.target_tokens,
        repetition_factor=cfg.constraints.repetition_factor,
        token_counts=cfg.priors.token_counts,
        natural_kl=None,
        test_ratios_path=(),
        test_metrics_path=(),
        aggregate_task_families=cfg.regression.aggregate_task_families,
        obj_weights=cfg.filtering.obj_weights,
    )
