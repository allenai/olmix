import json
import logging
import random
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
import platform
from typing import Any, Optional, Tuple, Union, List
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.optim as optim
from tqdm import tqdm
from wandb.apis.public import Run
import hashlib
import os
import yaml
import boto3
from copy import deepcopy
from collections import defaultdict
import pydantic_core
import statsmodels.api as sm
from matplotlib.colors import TwoSlopeNorm
import cvxpy as cp

from scipy.stats import norm  # for probit transform
from matplotlib.cm import ScalarMappable


import subprocess
from io import StringIO

from cookbook.aliases import SwarmConfig as CookbookExperimentConfig
from cookbook.utils.data import get_token_counts_and_ratios

from regmixer.synthesize_mixture import calculate_priors
from regmixer.eval.constants import WandbMetrics, GroupedWandbMetrics
from regmixer.eval.law import ScalingLaw
from regmixer.aliases import SourceConfig, ExperimentConfig

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


if platform.system() == "Darwin":  # Darwin is the system name for macOS
    import multiprocessing as mp

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    mp.set_start_method("spawn", force=True)

BASE_OUTPUT_DIR = "output/"
# Match regmix setup: https://github.com/sail-sg/regmix/blob/main/regression_fitting/regression.ipynb
LGBM_HPS = {
    "task": "train",
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": ["l1", "l2"],
    "seed": 42,
    "num_iterations": 10000,
    "learning_rate": 1e-2,
    "verbosity": -1,
    "early_stopping_round": 3,
}


class Regressor:
    def fit(self, x, y, idx, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, x):
        if not hasattr(self, "model"):
            raise AttributeError("Subclasses must define self.model before calling predict()")
        return self.model.predict(x)


class LightGBMRegressor(Regressor):
    def __init__(self, **kwargs):
        self.model = lgb.LGBMRegressor(**LGBM_HPS)

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            eval_set=[(x, target)],
            eval_metric="l2",
        )


class LinearRegressor(Regressor):
    #def __init__(self, **kwargs):
    #    self.model = LinearRegression(fit_intercept=False)

    def __init__(self, **kwargs):
        self.model = None 

    #def fit(self, x, y, idx, **kwargs):
    #    target = y[:, idx]
    #    self.model = self.model.fit(x, target)

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        # we do not add intercept because this would make X linearly dependent.
        self.model = sm.OLS(target, x).fit()

class QuadraticRegressor(Regressor):
    def __init__(self, interactions, **kwargs):
        self.model = None
        assert interactions is not None, "Interactions must be provided for quadratic regression."
        self.interactions = []
        for interaction in interactions:
            self.interactions.append(tuple([int(var) for var in interaction.split(",")]))

    def _transform(self, x):
        """Add interaction terms to x based on self.interactions"""
        x = np.asarray(x)
        interaction_terms = []
        for i, j in self.interactions:
            interaction = (x[:, i] * x[:, j]).reshape(-1, 1)
            interaction_terms.append(interaction)
        if interaction_terms:
            interaction_matrix = np.hstack(interaction_terms)
            x_augmented = np.hstack([x, interaction_matrix])
        else:
            x_augmented = x
        return x_augmented

    def fit(self, x, y, idx, **kwargs):
        x_augmented = self._transform(x)
        target = y[:, idx]
        self.model = sm.OLS(target, x_augmented).fit()

    def predict(self, x):
        x_augmented = self._transform(x)
        return self.model.predict(x_augmented)


class LogLinearRegressor(Regressor):
    def __init__(self, params=None, **kwargs):
        np.random.seed(42)
        random.seed(42)

        if params is None:
            self.model = ScalingLaw(mixing_law)
        else:
            self.model = params

    def fit(self, x, y, idx, early_stopping=0.0, max_step=100, delta=0.02):
        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            init_params_log_linear_law(idx, num_domains=x.shape[-1]),
            max_step=max_step,
            delta=delta,
            eps=early_stopping,
        )

    def predict(self, x):
        return mixing_law(
            torch.tensor(x, dtype=torch.float), torch.tensor(self.model, dtype=torch.float)
        ).numpy()


class LogNonLinearRegressor(Regressor):
    def __init__(self, params=None, B_mask=None, **kwargs):
        np.random.seed(42)
        random.seed(42)

        self.B_mask = B_mask
        if params is None:
            self.model = ScalingLaw(nonlinear_mixing_law)
        else:
            self.model = params

    def fit(self, x, y, idx, early_stopping=0.0, max_step=100, delta=0.02):
        for i, params in enumerate(
            init_params_log_nonlinear_law(idx, self.B_mask, num_domains=x.shape[-1])
        ):
            print(
                nonlinear_mixing_law(
                    torch.tensor(x, dtype=torch.float),
                    torch.tensor(params, dtype=torch.float),
                    B_mask=self.B_mask,
                )
            )
            if i == 0:
                break

        target = y[:, idx]
        self.model = self.model.fit(
            x,
            target,
            init_params_log_nonlinear_law(idx, self.B_mask, num_domains=x.shape[-1]),
            B_mask=self.B_mask,
            max_step=max_step,
            delta=delta,
            eps=early_stopping,
        )

    def predict(self, x):
        return nonlinear_mixing_law(
            torch.tensor(x, dtype=torch.float),
            torch.tensor(self.model, dtype=torch.float),
            B_mask=self.B_mask,
        ).numpy()


class SearchRegressor(Regressor):
    def __init__(self, **kwargs):
        pass

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        self.model = {tuple(row): target[i] for i, row in enumerate(x)}

    def predict(self, x):
        preds = []
        for row in x:
            if tuple(row) in self.model:
                preds.append(self.model[tuple(row)])
            else:
                preds.append(np.inf)
        return preds

    def get_searched_weights(self):
        return [np.array(weight) for weight, _ in self.model.items()]



def nonlinear_mixing_law(x, param, B_mask=None):
    """
    Vectorized implementation of nonlinear mixing law:
    Y = exp(log_c + x @ t + sum_k B_k * x_i_k * x_j_k)

    Parameters:
        x:       (batch_size, d) input data
        param:   tensor of shape (1 + d + len(B_mask),)
                 [log_c, t (d,), B_params (len(B_mask),)]
        B_mask:  list of (i, j) tuples specifying quadratic terms

    Returns:
        Tensor of shape (batch_size,) with predicted outputs
    """
    log_c_i = param[0]
    d = x.shape[1]
    t_i = param[1 : 1 + d]  # Linear weights len(domains)
    B_params = param[1 + d :]  # Quadratic weights len(B_mask)

    lin_term = torch.matmul(x, t_i)  # (batch_size,)

    quad_term = None
    if B_mask is not None and len(B_mask) > 0:
        B_mask = torch.tensor(B_mask, dtype=torch.long, device=x.device)  # (num_terms, 2)
        x_i = x[:, B_mask[:, 0]]  # (batch_size, num_terms)
        x_j = x[:, B_mask[:, 1]]  # (batch_size, num_terms)
        quad_term = (x_i * x_j) @ B_params  # (batch_size,)

    if quad_term is None:
        return torch.exp(log_c_i) + torch.exp(lin_term)
    else:
        return torch.exp(log_c_i) + torch.exp(lin_term) + torch.exp(quad_term)

def mixing_law(x, param, **kwargs):
    log_c_i = param[0]
    t_i = param[1:]
    result = torch.exp(log_c_i) + torch.exp(torch.matmul(x, t_i))
    return result



def init_params_log_linear_law(idx, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10):  # originally (-2, 1.5, 10)
        for _ in range(30):
            ts = [
                -np.random.rand() if i == idx else np.random.rand() * 0.1
                for i in range(num_domains)
            ]
            yield [log_c_i] + ts


def init_params_log_nonlinear_law(idx, B_mask, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10):  # originally (-2, 1.5, 10)
        for _ in range(30):
            lin_params = [
                -np.random.rand() if i == idx else np.random.rand() * 0.1
                for i in range(num_domains)
            ]
            quadratic_params = [np.random.rand() * 0.1 for i in range(len(B_mask))]
            yield [log_c_i] + lin_params + quadratic_params


REGRESSION_TYPES = {
    "lightgbm": LightGBMRegressor,
    "linear": LinearRegressor,
    "log_linear": LogLinearRegressor,
    "log_nonlinear": LogNonLinearRegressor,
    "search": SearchRegressor,
    "quadratic": QuadraticRegressor,
}


class Proposer:
    def __init__(self):
        pass

    def propose(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")


class SimulationProposer(Proposer):
    def propose(
        self,
        index: int,
        predictor: list[Regressor],
        prior_distributions: dict,
        original_prior: dict,
        ratios: pd.DataFrame,
        num_samples: int = 1_000_000,
        seed: int = 1337,
        search_iterations: int = 10,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        final_cookbook_path: Optional[Path] = None,
        manual_token_constraint_path: Optional[Path] = None,
        repetition_factor: float = 1.0,
        obj_weights: Optional[list] = None,
        temperature: Optional[float] = None,
        reference_scores: Optional[np.ndarray] = None,
        fixed_weight: Optional[dict[str, float]] = None,
        metric_type: Optional[str] = None,
        tol: Optional[float] = None,
        fixed_search_weight: Optional[str] = None,
        reference_ratio: Optional[float] = None,
        make_worst_mix: bool = False,
        min_weight_per_domain: float = 0.0,
        **kwargs
    ) -> np.ndarray:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        min_weight = 1e-5
        min_dirichlet = 1
        max_dirichlet = 100
        search_dirichlet_factor = 2.0

        if reference_ratio is not None:
            search_prior = np.array(reference_ratio)
        else:
            search_prior = np.array(list(prior_distributions.values()))

        if temperature is not None:
            search_prior = search_prior**temperature
            search_prior = search_prior / np.sum(search_prior)

        if fixed_search_weight is not None:
            fixed_dict = json.loads(fixed_search_weight)
            fixed_indices = {list(prior_distributions.keys()).index(domain): weight for domain, weight in fixed_dict.items()}
            free_indices = [i for i in range(len(search_prior)) if i not in fixed_indices]
            free_prior = search_prior[free_indices]
            free_prior /= np.sum(free_prior)
            logger.info(f"Prior for sampling is {free_prior}")
        else:
            logger.info(f"Prior for sampling is {search_prior}")

        best_weights = np.zeros(len(prior_distributions))

        if constrain_objective:
            # just need a desired token count and available token count
            assert final_cookbook_path is not None or manual_token_constraint_path is not None
            if final_cookbook_path is not None:
                with open(final_cookbook_path, "r") as f:
                    data = yaml.safe_load(f)

                final_config = CookbookExperimentConfig(**data, path=final_cookbook_path)
                desired_tokens = final_config.max_tokens

                token_universe = get_token_counts_and_ratios(
                    final_config.dataset.sources, final_config.dataset.dtype, True
                )
                available_tokens_per_source = {
                    path: relative_size * token_universe[1]
                    for path, relative_size in token_universe[0].items()
                }
                # ensures that order of sources in simulations and in the constraint dictionary are aligned
                available_tokens_per_source = {
                    source: available_tokens_per_source[source]
                    for source, _ in prior_distributions.items()
                }
            elif manual_token_constraint_path is not None:
                with open(manual_token_constraint_path, "r") as f:
                    data = yaml.safe_load(f)
                desired_tokens = data["requested_tokens"]
                group_ids = data.get("group_ids", None)


                

                leaf_level_constraints = data["leaf_level_constraints"]
                granular_constraints = data.get("granular_constraints", False)
                if leaf_level_constraints and not granular_constraints:
                    # if the manual constraints are at the same granularity as the prior distributions, we can use them directly
                    available_tokens_per_source = {
                        source: data["available_tokens"][source]
                        for source, _ in prior_distributions.items()
                    }

                    if group_ids is not None:
                        group_ids = {source: group_ids[source] for source in prior_distributions} # order the group ids dict
                        groups = np.array(list(group_ids.values()))
                        desired_tokens = np.array([desired_tokens[idx] for idx in groups])

                elif leaf_level_constraints and granular_constraints:
                    # if the manual constraints are at a finer granularity, we need to aggregate them
                    available_tokens_per_source = {
                        source: data["available_tokens"][source]
                        for source, _ in original_prior.items()
                    }

                    if group_ids is not None:
                        group_ids = {source: group_ids[source] for source in original_prior} # order the group ids dict
                        groups = np.array(list(group_ids.values()))
                        desired_tokens_per_source = np.array([desired_tokens[idx] for idx in groups])
                else:
                    # otherwise, our constraints are at the source level while prior_distribution is at the leaf level
                    available_tokens_per_source = data["available_tokens"]

        # Multi-step search leveraging iterative prior results
        for search_step in tqdm(
            range(search_iterations), desc=f"Searching in {num_samples} candidate samples"
        ):
            offset = np.log(search_dirichlet_factor * (search_step + 1))
            alphas = np.exp(
                np.random.uniform(
                    low=np.log(min_dirichlet) + offset,
                    high=np.log(max_dirichlet) + offset,
                    size=num_samples,
                )
            )

            # generate simulations by sampling from dirichlet distribution with parameter prior * alpha
            if fixed_search_weight is not None:
                free_indices_simulations = (
                    torch.distributions.Dirichlet(torch.from_numpy(alphas[:, None] * free_prior))
                    .sample()
                    .numpy()
                )
                simulations = np.zeros((num_samples, len(search_prior)))
                simulations[:, free_indices] = free_indices_simulations * (1 - sum(list(fixed_indices.values())))
                idx  = np.array(list(fixed_indices.keys()))
                vals = np.array(list(fixed_indices.values()))
                simulations[:, idx] = vals  # broadcasts across rows
            elif constrain_objective and isinstance(desired_tokens, list):
                total = sum(desired_tokens)
                simulations = np.zeros((num_samples, len(search_prior)))

                if granular_constraints:
                    coarse_groups = np.array([group_ids[k] if k in group_ids else np.unique([stuff for g, stuff in group_ids.items() if g.startswith(k)])[0]  for k, v in prior_distributions.items()])
                else:
                    coarse_groups = groups
                for g in np.unique(coarse_groups):
                    cols = np.where(coarse_groups==g)[0]

                    # deterministic mass for the whole group
                    w_g = desired_tokens[g] / total                    # scalar

                    # fold desired & prior to shape within-group
                    base_norm = search_prior[cols] / search_prior[cols].sum()                        # simplex over group
                    conc = alphas[:, None] * base_norm[None, :]          # (n_samples, |cols|)

                    # sample a simplex for the group, then scale the whole group by w_g
                    samp = torch.distributions.Dirichlet(torch.from_numpy(conc)).sample().numpy()  # (n_samples, |cols|)
                    simulations[:, cols] = samp * w_g
            else:
                simulations = (
                    torch.distributions.Dirichlet(torch.from_numpy(alphas[:, None] * search_prior))
                    .sample()
                    .numpy()
                )

            # Filter out invalid simulations from the population
            if False: #temperature is None:
                # keep this for reproducibility...
                simulations = simulations[
                    np.all(
                        simulations <= 6.5 * np.array(list(prior_distributions.values())), axis=1
                    )
                ]

            # only search over mixes that are within bounds of the swarm ratios. We don't want the regression model to extrapolate.
            """ratios_max = ratios[ratios.columns[3:]].max().values
            simulations = simulations[
                np.all(
                    simulations < ratios_max,
                    axis=1,
                )  
            ]"""

            if constrain_objective:
                original_simulation_size = len(simulations)

                if leaf_level_constraints and not granular_constraints:
                    # we can just directly do elementwise comparisons
                    # Check which simulations will be filtered out

                    if isinstance(desired_tokens, list):
                        #group_ids = {source: group_ids[source] for source in prior_distributions} # order the group ids dict
                        #groups = np.array(list(group_ids.values()))
                        #desired_tokens = np.array([desired_tokens[idx] for idx in groups])

                        group_normalized_simulations = np.zeros_like(simulations)
                        for g in np.unique(groups):
                            cols = np.where(groups == g)[0]
                            block = simulations[:, cols]                               # (n_samples, n_cols_in_group)
                            denom = block.sum(axis=1, keepdims=True)         # row sums within group
                            # Safe division: leave rows with denom==0 as zeros
                            np.divide(block, denom, out=block, where=(denom != 0))
                            group_normalized_simulations[:, cols] = block

                        token_usage = group_normalized_simulations * desired_tokens_per_source 
                    else:
                        token_usage = simulations * desired_tokens

                    token_limits = (
                        np.array(list(available_tokens_per_source.values())) * repetition_factor
                    )
                    valid_mask = (token_usage <= token_limits).all(axis=1)

                    filtered_count = np.sum(~valid_mask)
                    if filtered_count > 0:
                        logger.info(
                            f"Filtering out {filtered_count} simulations that exceed token constraints"
                        )

                    simulations = simulations[valid_mask]
                elif leaf_level_constraints and granular_constraints:
                    constrained_domains = list(original_prior.keys())
                    original_prior_mix = np.array(list(original_prior.values()))

                    source_to_indices = defaultdict(list)
                    for i, domain in enumerate(constrained_domains):
                        source = domain.split(":", 1)[0]
                        source_to_indices[source].append(i)

                    per_source_weights = {}
                    for source, indices in source_to_indices.items():
                        topic_mix = original_prior_mix[indices]
                        total = topic_mix.sum()
                        per_source_weights[source] = (np.array(indices), topic_mix/total)

                    expanded_simulations = np.zeros((len(simulations), len(constrained_domains)))
                    for j, source in enumerate(prior_distributions):
                        indices, weights = per_source_weights[source]
                        expanded_simulations[:, indices] = simulations[:, [j]] * weights

                    if isinstance(desired_tokens, list):
                        #group_ids = {source: group_ids[source] for source in original_prior} # order the group ids dict
                        #groups = np.array(list(group_ids.values()))
                        #desired_tokens = np.array([desired_tokens[idx] for idx in groups])

                        group_normalized_simulations = np.zeros_like(expanded_simulations)
                        for g in np.unique(groups):
                            cols = np.where(groups == g)[0]
                            block = expanded_simulations[:, cols]                               # (n_samples, n_cols_in_group)
                            denom = block.sum(axis=1, keepdims=True)         # row sums within group
                            # Safe division: leave rows with denom==0 as zeros
                            np.divide(block, denom, out=block, where=(denom != 0))
                            group_normalized_simulations[:, cols] = block

                        token_usage = group_normalized_simulations * desired_tokens_per_source 
                    else:
                        token_usage = expanded_simulations * desired_tokens

                    token_limits =  (
                        np.array(list(available_tokens_per_source.values())) * repetition_factor
                    )

                    valid_mask = (token_usage <= token_limits).all(axis=1)

                    filtered_count = np.sum(~valid_mask)
                    if filtered_count > 0:
                        logger.info(
                            f"Filtering out {filtered_count} simulations that exceed token constraints"
                        )

                    simulations = simulations[valid_mask]
                else:
                    raise NotImplementedError("Code is out of date.")
                    # the constraints are at the source level, so we need to aggregate the simulation vectors by source
                    # we map from source to their indices in the probability vector
                    source_to_indices = defaultdict(list)
                    for idx, domain in enumerate(prior_distributions):
                        source = domain.split(":", 1)[0]
                        source_to_indices[source].append(idx)

                    def passes_constraints(sim):
                        # compute source weights
                        for source, indices in source_to_indices.items():
                            source_weight = sim[indices].sum()
                            if (
                                source_weight * desired_tokens
                                > available_tokens_per_source[source] * repetition_factor
                            ):
                                return False
                        return True

                    simulations = np.array([sim for sim in simulations if passes_constraints(sim)])

                logger.info(
                    f"Removed {original_simulation_size - len(simulations)} out of {original_simulation_size} simulations that would repeat tokens at the final run scale."
                )

                if len(simulations) == 0:
                    continue


            if min_weight_per_domain > 0.0:
                simulations = simulations[np.all(simulations >= min_weight_per_domain, axis=1)]
                if len(simulations) == 0:
                    logger.info(f"No simulations remain after enforcing min weight per domain of {min_weight_per_domain}.")
                    continue

            predictions = np.array([reg.predict(simulations) for reg in predictor])
            if reference_scores is not None:
                # If reference scores are provided, filter simulations based on them
                if tol is not None:
                    tol_range = [tol]
                else:
                    tol_range = [0, 0.05, 0.1, 0.15, 0.2]
                for t in tol_range:
                    # we allow for predicted scores to be within a tolerance of the reference scores
                    # if the current tol results in no remaining simulations, we increase tol
                    if (metric_type == "primary_score" and not make_worst_mix) or (metric_type != "primary_score" and make_worst_mix):
                        pareto_idxs = np.where(np.all(predictions.T > reference_scores - t, axis=1))[0]
                    elif (metric_type == "primary_score" and make_worst_mix) or (metric_type != "primary_score" and not make_worst_mix):
                        pareto_idxs = np.where(np.all(predictions.T < reference_scores + t, axis=1))[0]
                    if len(pareto_idxs) != 0:
                        logger.info(f"Using eps={t} for enforcing pareto improvements")
                        break

                if len(pareto_idxs) == 0:
                    logger.info(f"No simulations passed the pareto filter.")
                    continue

                # filter both the simulations and corresponding predictions down
                simulations = simulations[pareto_idxs]
                predictions = predictions[:, pareto_idxs]
                logger.info(
                    f"Filtered simulations to {len(simulations)} based on reference scores."
                )

            if opt_avg_metric:
                if obj_weights is not None:
                    objs = np.average(predictions, axis=0, weights=obj_weights)
                    logger.info(f"Computing weighted average of predictions using {obj_weights}")
                else:
                    objs = predictions.mean(axis=0)
            else:
                objs = predictor[index].predict(simulations)

            if (metric_type == "primary_score" and not make_worst_mix) or (metric_type != "primary_score" and make_worst_mix):
                best_mask = (objs.max() - objs) < 1e-3
                print(objs.max())
            elif (metric_type == "primary_score" and make_worst_mix) or (metric_type != "primary_score" and not make_worst_mix):
                best_mask = (objs - objs.min()) < 1e-3
                print(objs.min())
            best_weights = simulations[best_mask].mean(0)

            # Zero out weights below min_weight threshold and normalize
            best_weights[best_weights < min_weight] = 0.0
            best_weights /= best_weights.sum()

            search_prior = (best_weights + search_prior) / 2

            if fixed_search_weight is not None:
                free_prior = search_prior[free_indices]
                free_prior /= np.sum(free_prior)

            predicted_best_performance = sum([pred.predict([best_weights]) for pred in predictor])
            logger.info(
                f"Current best weights is: {best_weights} with predicted performance {predicted_best_performance}, and search prior is: {search_prior}"
            )

        if constrain_objective:
            if leaf_level_constraints and not granular_constraints:
                # we can just directly do elementwise comparisons
                if not all(
                    best_weights * desired_tokens
                    <= np.array(list(available_tokens_per_source.values())) * repetition_factor
                ):
                    raise ValueError(f"Best weights are out of bounds!")
            elif leaf_level_constraints and granular_constraints:
                expanded_best_weights = np.zeros(len(constrained_domains))
                for j, source in enumerate(prior_distributions):
                    indices, weights = per_source_weights[source]
                    expanded_best_weights[indices] = best_weights[j] * weights

                if False:#isinstance(desired_tokens, list):
                    #group_ids = {source: group_ids[source] for source in original_prior} # order the group ids dict
                    #groups = np.array(list(group_ids.values()))
                    #desired_tokens = np.array([desired_tokens[idx] for idx in groups])

                    group_normalized_best_weights = np.zeros_like(expanded_best_weights)
                    for g in np.unique(groups):
                        cols = np.where(groups == g)[0]
                        block = expanded_best_weights[:, cols]                               # (n_samples, n_cols_in_group)
                        denom = block.sum(axis=1, keepdims=True)         # row sums within group
                        # Safe division: leave rows with denom==0 as zeros
                        np.divide(block, denom, out=block, where=(denom != 0))
                        group_normalized_best_weights[:, cols] = block

                    token_usage = group_normalized_best_weights * desired_tokens_per_source
                    assert all(token_usage <= np.array(list(available_tokens_per_source.values())) * repetition_factor)

            else:
                if not passes_constraints(best_weights):
                    raise ValueError(f"Best weights are out of bounds!")
                
        # if best_weights was collapsed (because of conditioning on a topic-level p* mix), we need to expand it to the full prior distribution

        return best_weights


class SearchProposer(Proposer):
    def propose(
        self, index: int, predictor: list[SearchRegressor], prior_distributions: dict,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        manual_token_constraint_path: Optional[Path] = None,
        repetition_factor: float = 1.0,
        **kwargs
    ):


        if constrain_objective:
            # just need a desired token count and available token count
            if manual_token_constraint_path is not None:
                with open(manual_token_constraint_path, "r") as f:
                    data = yaml.safe_load(f)
                desired_tokens = data["requested_tokens"]


                # if the manual constraints are at the same granularity as the prior distributions, we can use them directly
                available_tokens_per_source = {
                    source: data["available_tokens"][source]
                    for source, _ in prior_distributions.items()
                }
                logger.info(f"Using manual token constraints from {manual_token_constraint_path}")


        searched_weights = predictor[0].get_searched_weights()
        best_performance = np.inf
        best_weights = np.zeros(len(searched_weights[0]))
        for weight in searched_weights:
            if opt_avg_metric:
                pred = np.array([reg.predict(weight[None]) for reg in predictor]).mean(axis=0)[0]
            else:
                pred = predictor[index].predict(weight[None])[0]


            if constrain_objective:
                token_usage = weight * desired_tokens
                token_limits = (
                    np.array(list(available_tokens_per_source.values())) * repetition_factor
                )

                if (token_usage <= token_limits).all() and pred < best_performance:
                    best_performance = pred
                    best_weights = weight
            else:
                if pred < best_performance:
                    best_performance = pred
                    best_weights = weight

        return best_weights


class LogLinearExactProposer(Proposer):

    def propose(self, predictor: list[SearchRegressor], prior_distributions: dict,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        manual_token_constraint_path: Optional[Path] = None,
        repetition_factor: float = 1.0,
        kl_reg: Optional[float] = 0.1,
        obj_weights: Optional[list] = None,
        **kwargs
    ):
        assert opt_avg_metric, "LogLinearExactProposer only supports opt_avg_metric=True"
        if kl_reg is None:
            raise ValueError("kl_reg must be provided for LogLinearExactProposer")

        if constrain_objective:
            # just need a desired token count and available token count
            if manual_token_constraint_path is not None:
                with open(manual_token_constraint_path, "r") as f:
                    data = yaml.safe_load(f)
                desired_tokens = data["requested_tokens"]


                # if the manual constraints are at the same granularity as the prior distributions, we can use them directly
                available_tokens_per_source = {
                    source: data["available_tokens"][source]
                    for source, _ in prior_distributions.items()
                }
                logger.info(f"Using manual token constraints from {manual_token_constraint_path}")


                caps = np.array(list(available_tokens_per_source.values())) * repetition_factor / desired_tokens

        
        C = np.array([p.model[0] for p in predictor])                 # (n,)
        A = np.array([p.model[1:] for p in predictor])                # (n, d)
        n, d = A.shape
        weights = np.ones(n) / n if obj_weights is None else np.array(obj_weights)

        x = cp.Variable(d)

        q = np.array(list(prior_distributions.values()))
        q = np.asarray(q, dtype=float)
        eps=1e-12
        q = np.maximum(q, eps)         # ensure strictly positive
        q = q / q.sum()


        # c_i doesn’t affect the argmin, but harmless to include if you want the value
        # obj = cp.sum(cp.multiply(weights, C) + cp.exp(A @ x))
        #obj = cp.sum(cp.multiply(weights, cp.exp(A @ x)))       # identical argmin
        loss = cp.sum(cp.multiply(weights, cp.exp(A @ x)))

        # KL(x || q) = sum x*log(x/q) = sum rel_entr(x, q)
        kl = cp.sum(cp.rel_entr(x, q))

        obj = loss + kl_reg * kl

        constraints = [
            x >= 0,      
            cp.sum(x) == 1
        ]
        if constrain_objective:
            constraints.append(x <= caps)


        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver="ECOS", verbose=True)              # ECOS or SCS are good

        print(prob.value, prob.status)

        return x.value


PROPOSER_TYPES = {
    "simulation": SimulationProposer,
    "search": SearchProposer,
    "exact": LogLinearExactProposer
}


@dataclass
class RunInstance:
    id: str
    display_name: str
    config: dict
    samples: pd.DataFrame
    state: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "config": self.config,
            "samples": self.samples.to_dict(),
            "state": self.state,
        }


def get_output_dir(groups: list[str]) -> str:
    return f"{BASE_OUTPUT_DIR}{'_'.join(groups)}/"


def build_regression(
    idx: int,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    regression_type: str,
    early_stopping: float,
    interactions: Optional[List[str]] = None,
) -> Regressor:
    logger.info(f"Building regression model, index: {idx}")
    reg = REGRESSION_TYPES[regression_type](interactions=interactions)
    reg.fit(X_train, Y_train, idx, early_stopping=early_stopping)
    return reg


def get_runs_without_wandb(full_group_name, dashboard="olmo-3-evals"): #"regmixer"):
    bucket = "ai2-llm"
    base_prefix = f"evaluation/{dashboard}/"

    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=base_prefix)

    # Extract unique display names that match the pattern
    display_names = set()
    for page in pages:
        if "Contents" in page:
            for obj in page["Contents"]:
                key = obj["Key"]
                # Extract the display name from the key
                # Format: evaluation/{dashboard}/{display_name}/...
                parts = key.split("/")
                if len(parts) >= 3:
                    display_name = parts[2]  # The display name part
                    if display_name.startswith(full_group_name):
                        display_names.add(display_name)

    print(f"Found display names: {sorted(display_names)}")
    return display_names


def get_runs_from_api(
    api,
    workspace: str,
    groups: list[str],
    cache_path: Path,
    no_cache: bool,
    num_samples: int,
    eval_metric_group: GroupedWandbMetrics,
) -> list[RunInstance]:

    wandb_runs = []
    for group in groups:
        wandb_runs.extend(
            api.runs(
                path=workspace,
                # filters={"display_name": {"$regex": f"^(?!.*larger).*{group}.*$"}},
                filters={"display_name": {"$regex": f".*{group}.*"}},
            )
        )

    # NOTE: Last writer wins and should be sorted created_at desc
    memo = {}
    for run in wandb_runs:
        memo[run.display_name] = run

    all_runs = []
    for run in memo.values():
        if run.state == "crashed":
            logger.warning(f"Run {run.display_name} has crashed; still using its final result")

        if run.state == "running":
            logger.warning(f"Run {run.display_name} is still running; NOT skipping though")

        if run is not None:
            all_runs.append(mk_run_history(run, num_samples, eval_metric_group))

    all_runs = sorted(all_runs, key=lambda run: run.display_name.lower())
    if not no_cache:
        with open(cache_path, "w") as f:
            json.dump([run.as_dict() for run in all_runs], f)

    return all_runs


def mk_run_history(run: Run, samples: int, eval_metric_group: GroupedWandbMetrics) -> Any:
    if samples == 1:
        try:
            summary = [
                {
                    metric: run.summary[metric]
                    for metric in eval_metric_group.value
                    if metric in run.summary
                }
            ]
        except KeyError:
            print(run.id)
            print(run.summary.keys())

            breakpoint()
        return mk_run_instance(run, summary, samples)
    else:
        return mk_run_instance(run, run.scan_history(keys=eval_metric_group.value), samples)


def mk_run_from_json(run: dict) -> RunInstance:
    return RunInstance(
        id=run["id"],
        display_name=run["display_name"],
        config=run["config"],
        samples=pd.DataFrame(run["samples"]),
        state=run["state"],
    )


def mk_run_instance(run: Run, history: list[Any], n_samples: int) -> RunInstance:
    samples = pd.DataFrame.from_records(history).tail(n_samples)
    # logger.info(
    #    f"Collected RunInstance for {run.display_name}:{run.id} with samples: {samples.shape}"
    # )
    return RunInstance(
        id=run.id,
        display_name=run.display_name,
        config=run.config,
        samples=samples,
        state=run.state,
    )


def compute_mixture_neighborhood(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    ratios: pd.DataFrame,
    neighborhood: str,
    train_split: float,
) -> tuple[np.ndarray, np.ndarray]:
    neighborhood_size = int(train_split * len(X_train))
    logger.info(f"Computing neighborhood of size {neighborhood_size} for {neighborhood}")
    centroid = ratios[ratios.name == neighborhood][ratios.columns[3:]].values[0]
    distances = np.linalg.norm(X_train - centroid, axis=1)
    # Get indices of k smallest distances
    nearest_indices = np.argsort(distances)[:neighborhood_size]
    X_train = X_train[nearest_indices]
    Y_train = Y_train[nearest_indices]
    return X_train, Y_train


def plot_simulations(
    prior_distributions: np.ndarray,
    samples,
    columns: list[str],
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: float,
    output_dir: str = BASE_OUTPUT_DIR,
):
    plt.close()
    df = pd.DataFrame(
        data=np.concatenate([np.array([prior_distributions]), samples], axis=0),
        columns=columns,
    )
    df["sample"] = df.index
    melted_df = df.melt(id_vars=["sample"], var_name="Domain", value_name="Weight")
    g = sns.FacetGrid(melted_df, col="sample", col_wrap=4, aspect=2)
    g.map_dataframe(sns.barplot, x="Domain", y="Weight", palette="viridis", hue="Domain")
    g.set(ylim=(0, 0.75))
    g.set_axis_labels("Domain", "Weight")

    for ax in g.axes.flat:
        for label in ax.get_xticklabels():
            label.set_rotation(90)
        ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)

    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha)}_sim_grid.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()


def plot_correlation(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    index: int,
    predictors: list[Regressor],
    train_split: tuple[float],
    n_test: int,
    split_seed: int,
    n_samples: int,
    metric_name: str,
    regression_type: str,
    alpha: Optional[float] = None,
    output_dir: str = BASE_OUTPUT_DIR,
):
    plt.close()

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
        }
    )


    y_pred_train = predictors[index].predict(X_train)
    y_true_train = Y_train[:, index]

    corr_results = {}

    if train_split[0] == 1 and n_test == 0:
        # Only plot train if train and test are the same
        sns.regplot(
            x=y_pred_train,
            y=y_true_train,
            scatter_kws={"s": 64, "color": "#105257"},
            line_kws={"color": "#F0529C", "linewidth": 3, "linestyle": "dashed"},
            label="Train",
        )

        corr_train = np.corrcoef(y_pred_train, y_true_train)[0, 1]
        plt.legend(
            title=f"{metric_name.split('/')[-1]} correlation",
            labels=[f"Train: {np.round(corr_train * 100, 2)}"],
            fontsize=12,
            title_fontsize=12,
        )

        corr_results["train"] = corr_train
    else:
        # Predict test
        y_pred_test = predictors[index].predict(X_test)
        y_true_test = Y_test[:, index]

        # Plot test
        sns.regplot(
            x=y_pred_test,
            y=y_true_test,
            scatter_kws={"s": 64, "color": "#105257"},
            line_kws={"color": "#F0529C", "linewidth": 3, "linestyle": "dashed"},
            label="Test",
        )

        # Plot train
        sns.regplot(
            x=y_pred_train,
            y=y_true_train,
            scatter_kws={"s": 64, "color": "#B0C4DE"},
            line_kws={"color": "#8888FF", "linewidth": 3, "linestyle": "dotted"},
            label="Train",
        )

        corr_test = np.corrcoef(y_pred_test, y_true_test)[0, 1]
        corr_train = np.corrcoef(y_pred_train, y_true_train)[0, 1]

        import matplotlib.patches as mpatches

        test_dot = mpatches.Patch(color="#105257", label=f"Test: {np.round(corr_test * 100, 2)}")
        train_dot = mpatches.Patch(color="#B0C4DE", label=f"Train: {np.round(corr_train * 100, 2)}")

        plt.legend(
            handles=[test_dot, train_dot],
            title=f"{metric_name.split('/')[-1]} correlations",
            fontsize=12,
            title_fontsize=12,
        )

        corr_results["train"] = corr_train
        corr_results["test"] = corr_test

    # Common plot settings
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)
    plt.grid(True, linestyle="dashed")
    plt.tight_layout()

    # Save figure
    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha)}_fit.png"
    )
    plt.close()

    with open(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_correlations.json",
        "w",
    ) as f:
        f.write(json.dumps(corr_results))


def plot_interaction_matrix(
    output_dir: str,
    predictors: list[Regressor],
    regression_type: str,
    domain_names: list[str],
    metric_names: list[str],
    ratios: pd.DataFrame,
    metric_type: Optional[str] = None,
    interactions: Optional[List[str]] = None,
):
    metric_names = [metric.split("/")[-1].split(" ")[0] for metric in metric_names]

    interaction_pairs = []
    if interactions is not None:
        for interaction in interactions:
            interaction_pairs.append(tuple([int(var) for var in interaction.split(",")]))

    interaction_matrix = np.zeros((len(metric_names), len(domain_names) + len(interaction_pairs)))

    for i, predictor in enumerate(predictors):
        if regression_type == "lightgbm":
            interaction_matrix[i] = predictor.model.feature_importances_
        elif regression_type == "log_linear":
            interaction_matrix[i] = predictor.model[1:]
        elif regression_type == "linear":
            # normalize coefficients by the standard deviation of the corresponding domain
            #std = ratios[ratios.columns[3:]].std(ddof=0).values  # std for selected columns only
            interaction_matrix[i] = predictor.model.params #* std
        elif regression_type == "quadratic":
            interaction_matrix[i] = predictor.model.params

    domain_reordering = None 
    if "a3d4f82c" in output_dir:
        # hardcode the nice block structure for the superswarm
        new_order = ['code_fim', 'dolminomath', 'megamatt', 'openmathreasoning-fullthoughts-rewrite', 'swallowcode', 'swallowmath', 'tinymath-mind', 'tinymath-pot',  'flan', 'instruction-new-format',  'nemotron-synth-qa', 'rcqa', 'reddit-high', 'sponge', 'hqweb-pdf']
        domain_reordering = [domain_names.index(d) for d in new_order]
        domain_names = new_order
        interaction_matrix = interaction_matrix[:, domain_reordering]




    sorted_metric_indices = np.argsort(metric_names)
    metric_names = [metric_names[i] for i in sorted_metric_indices]
    interaction_matrix = interaction_matrix[sorted_metric_indices, :]

    if "olmo3:dev:7b:gen" in metric_names and "olmo3:dev:7b:math:v2" in metric_names:
        if metric_names.index("olmo3:dev:7b:gen") < metric_names.index("olmo3:dev:7b:math:v2"):
            # Swap the two if they are in the wrong order
            idx_gen = metric_names.index("olmo3:dev:7b:gen")
            idx_math = metric_names.index("olmo3:dev:7b:math:v2")
            metric_names[idx_gen], metric_names[idx_math] = metric_names[idx_math], metric_names[idx_gen]
            interaction_matrix[[idx_gen, idx_math]] = interaction_matrix[[idx_math, idx_gen]]


    plt.figure(figsize=(20, 16))
    plt.imshow(interaction_matrix, cmap="rainbow", aspect="auto")
    cmap = plt.cm.coolwarm
    vlim = np.abs(interaction_matrix).max()
    # Show color mesh
    c = plt.imshow(interaction_matrix, cmap=cmap, vmin=-vlim, vmax=+vlim, aspect='auto')

    bar_label = "Influence"
    if metric_type == "primary_score":
        bar_label += " (higher is better)"
    else:
        bar_label += " (lower is better)"

    plt.colorbar(label=bar_label)
    plt.xticks(ticks=np.arange(len(domain_names)), labels=domain_names, rotation=90)
    plt.yticks(ticks=np.arange(len(metric_names)), labels=metric_names)
    plt.title(f"Interaction matrix for {regression_type}")

    # Annotate each cell with its value
    for i in range(len(metric_names)):
        if regression_type in ["linear", "quadratic"]:
                p_values = predictors[i].model.pvalues
        for j in range(len(domain_names)):
            val = interaction_matrix[i, j]
            text_str = f"β={val:.2f}"
            if regression_type in ["linear", "quadratic"]:
                text_str += f"\np={p_values[j]:.2g}"
            plt.text(
                j, i, text_str,
                ha='center', va='center',
                color='black' if abs(val) < 0.5 * np.max(np.abs(interaction_matrix)) else 'white',
                fontsize=10
            )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/interaction_matrix.png",
        bbox_inches="tight",
    )
    plt.close()
    np.save(f"{output_dir}/interaction_matrix.npy", interaction_matrix)




# Optional: BH FDR (Benjamini–Hochberg) to convert p->q
def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    flat = pvals.ravel().astype(float)
    n = flat.size
    order = np.argsort(flat)
    ranks = np.empty_like(order); ranks[order] = np.arange(1, n+1)
    q = flat * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty_like(q_sorted); out[order] = q_sorted
    return out.reshape(pvals.shape)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm

# Define red-white-blue with a white "band"
colors = ["red", "white", "blue"]
cmap = LinearSegmentedColormap.from_list("red_white_blue", colors)

# Example boundaries: values between -0.1 and +0.1 stay white
bounds = [-1.0, -0.9, 0.9, 10]
norm = BoundaryNorm(bounds, cmap.N, extend="both")


def plot_interaction_matrix_signed_evidence(
    output_dir: str,
    predictors: List,
    regression_type: str,
    domain_names: List[str],
    metric_names: List[str],
    ratios: pd.DataFrame,
    metric_type: Optional[str] = None,
    use_fdr: bool = False,
    p_cap: float = 10.0,          # kept for API compatibility; unused in p-coloring
    sig_threshold: float = 0.05,
    gamma: float = 1.5            # >1 makes the color mapping less drastic
):
    # Normalize metric labels
    metric_names = [m.split("/")[-1].split(" ")[0] for m in metric_names]

    # Optional hardcoded domain block structure + build a column permutation
    domain_reordering = None
    if "a3d4f82c" in output_dir:
        new_order = [
            'code_fim', 'dolminomath', 'megamatt',
            'openmathreasoning-fullthoughts-rewrite',
            'swallowcode', 'swallowmath',
            'tinymath-mind', 'tinymath-pot',
            'flan', 'instruction-new-format',
            'nemotron-synth-qa', 'rcqa',
            'reddit-high', 'sponge', 'hqweb-pdf'
        ]
        domain_reordering = [domain_names.index(d) for d in new_order]
        domain_names = new_order

    # Collect coefficients (B) and p-values (P)
    B = np.zeros((len(metric_names), len(domain_names)))
    P = np.full_like(B, np.nan, dtype=float)

    for i, pred in enumerate(predictors):
        if regression_type == "linear":
            # Assumes pred.model.params and pred.model.pvalues are aligned to current domain_names order
            B[i] = pred.model.params
            P[i] = pred.model.pvalues
        elif regression_type == "lightgbm":
            # Feature importances only; no p-values
            B[i] = getattr(pred.model, "feature_importances_", np.zeros(len(domain_names)))
        elif regression_type == "log_linear":
            # e.g., first element intercept, then one per domain
            B[i] = pred.model[1:]

    # Row (metric) sorting
    order = np.argsort(metric_names)
    metric_names = [metric_names[k] for k in order]
    B = B[order]
    P = P[order]

    if "olmo3:dev:7b:gen" in metric_names and "olmo3:dev:7b:math:v2" in metric_names:
        if metric_names.index("olmo3:dev:7b:gen") < metric_names.index("olmo3:dev:7b:math:v2"):
            # Swap the two if they are in the wrong order
            idx_gen = metric_names.index("olmo3:dev:7b:gen")
            idx_math = metric_names.index("olmo3:dev:7b:math:v2")
            metric_names[idx_gen], metric_names[idx_math] = metric_names[idx_math], metric_names[idx_gen]
            B[[idx_gen, idx_math]] = B[[idx_math, idx_gen]]
            P[[idx_gen, idx_math]] = P[[idx_math, idx_gen]]

    # Column (domain) reordering to match hardcoded order, if requested
    if domain_reordering is not None:
        B = B[:, domain_reordering]
        P = P[:, domain_reordering]

    # Plotting
    if regression_type == "linear" and np.isfinite(P).any():
        # Optionally compute FDR q-values for dimming only
        Q = bh_adjust(P) if use_fdr else P

        # ---- COLOR BY p, gently (bounded), while TEXT shows raw p ----
        # Map p in [0,1] to a gentle strength via (1 - p)^(1/gamma), then apply sign(β)
        P_safe = np.clip(P, 0.0, 1.0)
        p_strength = (1.0 - P_safe) ** (1.0 / max(gamma, 1e-6))
        signed_score = np.sign(B) * p_strength  # in [-1, 1]
        v = 1.0

        plt.figure(figsize=(20, 16))
        #im = plt.imshow(
        #    signed_score,
        #    cmap="coolwarm",
        #    norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
        #    aspect="auto",
        #)

        im = plt.imshow(
            signed_score,
            cmap=cmap,
            norm=norm,
            aspect="auto"
        )


        # Dim non-significant cells (gray overlay)
        #mask = (Q if use_fdr else P_safe) > sig_threshold
        #overlay = np.zeros((*signed_score.shape, 4))
        #overlay[mask] = [0.7, 0.7, 0.7, 0.6]
        #plt.imshow(overlay, aspect="auto")

        # Annotate with β and RAW p (as requested)
        for i in range(len(metric_names)):
            for j in range(len(domain_names)):
                text_str = f"β={B[i, j]:.2f}\np={P[i, j]:.2g}"
                text_color = "black" #if abs(signed_score[i, j]) < 0.5 * v else "white"
                plt.text(j, i, text_str, ha="center", va="center",
                         fontsize=8, color=text_color)

        cbar = plt.colorbar(im)
        cbar.set_label(r"sign(β) × (1 − p)$^{1/\gamma}$")
        better = "higher is better" if metric_type == "primary_score" else "lower is better"
        plt.title(f"Signed evidence heatmap (color ∝ p, gentler gradient; text shows raw p)\n({better})")

    else:
        # No p-values available: show β heatmap only
        plt.figure(figsize=(25, 16))
        im = plt.imshow(B, cmap="coolwarm", aspect="auto")
        for i in range(len(metric_names)):
            for j in range(len(domain_names)):
                plt.text(j, i, f"{B[i, j]:.2f}", ha="center", va="center", fontsize=8)
        cbar = plt.colorbar(im)
        cbar.set_label("β value")
        plt.title(f"β heatmap ({regression_type})")

    plt.xticks(ticks=np.arange(len(domain_names)), labels=domain_names, rotation=90)
    plt.yticks(ticks=np.arange(len(metric_names)), labels=metric_names)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/interaction_matrix_signed_evidence.png", bbox_inches="tight")
    plt.close()



def mk_run_metrics(
    history,
    samples: int,
    metrics: Tuple[str, list[str]],
    display_name: str,
    average: bool = False,
    dashboard: list[str] = ["regmixer"],  # ["olmo-3-evals"]
    metric_type: Optional[str] = None,
    pull_from_dashboard: bool=False
) -> dict[str, float]:
    df = pd.DataFrame(history)
    results = {}
    group_name, group_metrics = metrics
    in_loop_tasks = [task for task in df.columns if task in group_metrics]
    offline_tasks = [task for task in group_metrics if task not in in_loop_tasks]
    if average:
        raise NotImplementedError("Averaging the task is implemented but out of date!")
        result = np.mean(
            [df.loc[:, metric_name].tail(samples).mean() for metric_name in group_metrics]
        )
        results[group_name] = result
    else:
        if pull_from_dashboard:
            assert metric_type=="primary_score", "Only primary_score metric type is supported for dashboard evaluation"
            for d in dashboard:
                offline_results = get_offline_evals_from_dashboard(display_name, offline_tasks, dashboard=d)
                results.update(offline_results)
        else:
            for metric_name in in_loop_tasks:
                results[metric_name] = df.loc[:, metric_name].tail(samples).mean()

            if len(offline_tasks) > 0:
                # need to obtain offline results
                for d in dashboard:
                    logger.info(f"Getting offline results for {display_name} in {d} dashboard")
                    offline_results = get_offline_evals(display_name, offline_tasks, group_name, dashboard=d, metric_type=metric_type)
                    results.update(offline_results)

    return results

def get_offline_evals_from_dashboard(display_name, tasks, dashboard):
    command = [
        "olmo-cookbook-eval", "results",
        "--dashboard", f"{dashboard}",        
    ]

    for task in tasks:
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

    df = df[df['name'].str.contains(display_name)]
    assert len(df) == 1 

    df = df[tasks]
    return df.to_dict()


def get_offline_evals(display_name, tasks, group_name, dashboard="regmixer", metric_type=None):#"olmo-3-evals"):# "regmixer"):
    bucket = "ai2-llm"
    prefix = f"evaluation/{dashboard}/{display_name}"

    s3 = boto3.client("s3")

    # Get list of all files under the prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    json_files = []
    jsonl_files = []

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/"):
                if key.endswith(".jsonl"):
                    jsonl_files.append(key)
                elif key.endswith(".json"):
                    json_files.append(key)

    all_jsonl_data = []

    for key in jsonl_files:
        if key.endswith("metrics-all.jsonl"):
            obj = s3.get_object(Bucket=bucket, Key=key)
            for line in obj["Body"].iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode("utf-8"))
                        all_jsonl_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON line in {key}: {e}")

    #all_available_tasks = [data['task_config'].get('metadata', {}).get('alias')  for data in all_jsonl_data]
    #logger.info(f"Available tasks in JSONL data for {display_name}:")
    #for task in sorted(all_available_tasks):
    #    print(f'"{task}",')
    #raise ValueError()

    offline_results = {}
    for i, task in enumerate(tasks):
        task_data = [
            data
            for data in all_jsonl_data
            if (
                data["task_config"].get("metadata", {}).get("alias") == task
                or data["task_name"] == task
            )
        ]
        if len(task_data) == 0:
            task = task.replace("bpb:", "")
            task_data = [
                data
                for data in all_jsonl_data
                if (
                    data["task_config"].get("metadata", {}).get("alias") == task
                    or data["task_name"] == task
                )
            ]
            if len(task_data) == 0:
                task = task + ":full"
                task_data = [
                    data
                    for data in all_jsonl_data
                    if (
                        data["task_config"].get("metadata", {}).get("alias") == task
                        or data["task_name"] == task
                    )
                ]
                if len(task_data) == 0:
                    all_available_tasks = [data['task_config'].get('metadata', {}).get('alias')  for data in all_jsonl_data]

                    if task.replace(":full", "") in ["ultrachat_masked_ppl", "wildchat_masked_ppl", "qasper_yesno:rc::olmes",
                        "sciriff_yesno:rc::olmes", "lab_bench_dbqa", "lab_bench_protocolqa", "medqa_en:rc::none"] and group_name == "pretraining_tasks_for_paper":
                        continue
                    else:
                        logger.warning(f"Task {task} not found in JSONL data for {display_name}. Available tasks: {all_available_tasks}")
                        continue
                else:
                    logger.info(
                        f"Task {task} found in JSONL data for {display_name} with alias {task_data[0]['task_config'].get('metadata', {}).get('alias')}"
                    )
                    
        data = task_data[0]
        if metric_type is not None:
            if metric_type not in data["metrics"]:
                logger.warning(
                    f"Metric type {metric_type} not found in task {data['task_name']} metrics. Available metrics: {data['metrics'].keys()}"
                )
                continue
            else:
                name = data["task_config"]["metadata"]["alias"].replace("bpb:", "").replace(":full", "")
                metric_value = data["metrics"][metric_type]
                offline_results[name] = metric_value
        else:
            if "bits_per_byte_corr" in data["metrics"]:
                name = data["task_config"]["metadata"]["alias"].replace("bpb:", "").replace(":full", "")
                offline_results[name] = data["metrics"]["bits_per_byte_corr"]
                #logger.info(
                #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte_corr {data['metrics']['bits_per_byte_corr']}"
                #)
            elif "bits_per_byte_corr_macro" in data["metrics"]:
                name = data["task_config"]["metadata"]["alias"].replace("bpb:", "").replace(":full", "")
                offline_results[name] = data["metrics"]["bits_per_byte_corr_macro"]
                #logger.info(
                #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte_corr_macro {data['metrics']['bits_per_byte_corr_macro']}"
                #)
            elif "bits_per_byte" in data["metrics"]:
                name = data["task_name"].replace("bpb:", "").replace(":full", "")
                offline_results[name] = data["metrics"]["bits_per_byte"]
                #logger.info(
                #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte {data['metrics']['bits_per_byte']}"
                #)
            else:
                logger.warning(
                    f"{data['task_name']} does not have bits_per_byte_corr or bits_per_byte_corr_macro in metrics {data['metrics'].keys()}"
                )

    return offline_results


def mk_weights_from_config(config: dict, priors: tuple, display_name: str, patched: bool = False) -> dict[str, float]:
    source_mixture_config = (
        config.get("dataset", {})
        .get("source_mixture_config", {})
    )

    sources = source_mixture_config.get("source_configs") or source_mixture_config.get("source_list").get("sources") or []

    source_configs = {
        source["source_name"]: source
        for source in sources
    }

    prefixes = ['dclm', 's2pdf', 'pes2o', 'stack-edu', 'finemath-3plus', 'arxiv', 'wikipedia']
    source_configs = {
        (
            name.replace("_", ":", 1)
            if any(name.startswith(prefix + "_") and not name.startswith("dclm_v2") for prefix in prefixes)
            else name
        ): value
        for name, value in source_configs.items()
    }

    if "62e7dc06" in display_name and patched:
        # need this when patching to align up all the domain names
        source_configs = {f"dclm:{k}" : v for k, v in source_configs.items()}

    weights = {}
    for domain in priors[0].keys():
        if domain not in source_configs:
            # two cases
            if ":" in domain and domain.split(":")[0] in source_configs:
                # 1) prior (i.e., swarm config) requests a leaf but mixes from the wandb (i.e. launched outside of the swarm, like natural distr) are specified at the source level
                source_name = domain.split(":")[0]
                weights[domain] = (
                    source_configs[source_name].get("target_ratio", 0.0) * priors[0][domain]
                )
            elif ":" in domain and domain.split(":")[-1] in source_configs:
                domain_name = domain.split(":")[-1]
                weights[domain] = source_configs[domain_name].get("target_ratio", 0.0)
            elif ":" not in domain:
                # 2) prior requests a source (i.e. when we condition on a topic-level p* mix) but wandb mixes are specified at the leaf level
                cfg = {
                    k: v.get("target_ratio", 0.0) for k, v in source_configs.items() if f"{domain}:" in k
                }
                weights[domain] = sum(cfg.values()) if cfg else 0.0
            else:
                # 3) prior's domain has 0 weight in the wandb config
                weights[domain] = 0.0
        else:
            weights[domain] = source_configs.get(domain, {}).get("target_ratio", 0.0)

    return weights


def solve_log_linear(
    predictor: list[Regressor],
    prior_distributions: np.ndarray,
    df_config: pd.DataFrame,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: float = 1.0,
    output_dir: str = BASE_OUTPUT_DIR,
    seed: int = 1337,
) -> np.ndarray:

    torch.manual_seed(seed)

    # Split params into biases (b) and t values
    t = [p[2:] for p in predictor]
    b = [p[1] for p in predictor]

    t = torch.tensor(t, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)

    # Initialize weights as a probability vector
    weights = torch.rand(len(t[0]) - 1, requires_grad=True)
    assert weights.sum() <= 1
    # weights = torch.nn.Parameter(raw_weights / raw_weights.sum())

    def objective(weights):
        return torch.sum(torch.exp(b + t @ weights))

    def project(w):
        """
        Projects a vector w (length n-1) so that:
        - Each entry is in [0, 1]
        - Sum of entries <= 1
        Returns the full probability vector (length n), where the last element is 1 - sum(w)
        """
        w.data.clamp_(0, 1)  # clamp each entry between 0 and 1

        total = w.sum()
        if total > 1:
            w.data.mul_(1 / total)  # rescale to make sum <= 1

        last = 1.0 - w.sum()
        last = torch.clamp(last, min=0.0, max=1.0)  # ensure numerical safety

        return torch.cat([w, last.unsqueeze(0)], dim=0)  # final full probability vector

    # Optimization
    optimizer = optim.Adam([weights], lr=0.001)
    n_iterations = 1000
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = objective(weights)
        loss.backward()
        optimizer.step()

        # Project onto the probability simplex
        with torch.no_grad():
            weights.data = project(weights.data)

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.4f}")

    best_weights = weights.detach().cpu().numpy()

    plot_and_log_weights(
        prior=prior_distributions,
        prediction=best_weights,
        metric_name=metric_name,
        regression_type=regression_type,
        train_split=train_split,
        n_test=n_test,
        split_seed=split_seed,
        n_samples=n_samples,
        alpha=alpha,
        df_config=df_config,
        output_dir=output_dir,
    )

    return best_weights



def expand_collapsed_weights(
    opt_weights: dict[str, float],
    original_prior: dict[str, float],
    collapsed_prior: dict[str, float],
) -> dict[str, float]:

    topics_to_expand = list(
        set(list(original_prior.keys())).difference(set(list(collapsed_prior.keys())))
    )
    collapsed_sources = list(
        set(list(collapsed_prior.keys())).difference(set(list(original_prior.keys())))
    )

    for source in collapsed_sources:
        topics_per_source = sorted([t for t in topics_to_expand if source in t])
        topic_weights = {
            t: original_prior[t] / collapsed_prior[source] * opt_weights[source]
            for t in topics_per_source
        }
        del opt_weights[source]  # remove the source key
        opt_weights.update(topic_weights)  # add the topic keys with their expanded weights

    return opt_weights


def add_back_in_fixed_source_weights(
    opt_weights: dict[str, float], original_prior: dict[str, float], fixed_weight: dict[str, float]
) -> dict[str, float]:

    domains_to_add_back_in = list(
        set(list(original_prior.keys())).difference(set(list(opt_weights.keys())))
    )

    final_weights = {}
    for source, weight in fixed_weight.items():
        if any([domain.startswith(source + ":") for domain in opt_weights]):
            # this source is already in the opt_weights, we just need to normalize it
            # get all the topics associated with this source, normalize the within-source distribution, and scale it by the fixed weights
            topics_per_source = {t: w for t, w in opt_weights.items() if t.startswith(source + ":")}
            total = sum(list(topics_per_source.values()))
            topics_per_source = {t: w / total * weight for t, w in topics_per_source.items()}
            final_weights.update(topics_per_source)

        elif source in domains_to_add_back_in:
            # this source is not in the opt_weights, but it is in the original prior and has a fixed weight
            # we can just add it back in with its fixed weight
            final_weights[source] = weight

        elif any([domain.startswith(source + ":") for domain in domains_to_add_back_in]):
            # this source is not in the opt_weights, but it has topics in the original prior
            # we need to expand the fixed weight according to the original prior distribution
            topics_per_source = {
                t: w for t, w in original_prior.items() if t.startswith(source + ":")
            }
            total = sum(list(topics_per_source.values()))
            topics_per_source = {t: w / total * weight for t, w in topics_per_source.items()}
            final_weights.update(topics_per_source)

    # normalize again just to fix any numerical issues
    total = sum(final_weights.values())
    final_weights = {k: v / total for k, v in final_weights.items()}

    return final_weights


def plot_and_log_weights(
    prior: dict[str, float],
    original_prior: dict[str, float],
    prediction: np.ndarray,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: float,
    df_config: pd.DataFrame,
    output_dir: str = BASE_OUTPUT_DIR,
    fixed_weight: Optional[dict[str, float]] = None,
):
    logger.info(f":::::::::{metric_name}:::::::::")
    logger.info("Predicted optimal weights:")

    if set(list(original_prior.keys())) != set(list(prior.keys())):
        # expand weights
        opt_weight_dict = {k: prediction[i] for i, (k, v) in enumerate(prior.items())}
        opt_weight_dict = expand_collapsed_weights(opt_weight_dict, original_prior, prior)
        out = [{"domain": domain, "weight": weight} for domain, weight in opt_weight_dict.items()]
        columns = list(prior.keys())
    else:
        columns = df_config.columns[3:].to_list()
        out = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)]

    if fixed_weight is not None:
        opt_weight_dict = add_back_in_fixed_source_weights(
            opt_weight_dict, original_prior, fixed_weight
        )
        out = [{"domain": domain, "weight": weight} for domain, weight in opt_weight_dict.items()]

    if len(out) != len(df_config.columns[3:]):
        logger.info("RAW WEIGHTS:")
        raw_weights = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)]
        logger.info(raw_weights)

    with open(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_optimal.json",
        "w",
    ) as f:
        logger.info(out)
        f.write(json.dumps(out))

    df = pd.DataFrame(
        data=np.concatenate(
            [
                np.array([list(prior.values())]),
                np.array([prediction]),
            ],
            axis=0,
        ),
        columns=columns,
    )
    df = pd.melt(df)
    df["type"] = (["Corpus"] + ["Optimal"]) * len(columns)

    plt.rc("axes", unicode_minus=False)
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
        }
    )

    _, ax = plt.subplots(figsize=(12, 10), layout="compressed")
    ax.ticklabel_format(useMathText=True)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.tick_params(axis="x", labelrotation=90)

    pallette = {
        "Corpus": "#105257",
        "Optimal": "#F0529C",
    }

    df_sorted = df[df["type"] == "Corpus"].sort_values(by="value", ascending=False)
    df["variable"] = pd.Categorical(df["variable"], categories=df_sorted["variable"], ordered=True)
    sns.barplot(data=df, x="variable", y="value", hue="type", palette=pallette, ax=ax)

    ax.legend(
        edgecolor="black",
        fancybox=False,
        prop={
            "size": 18,
        },
        handlelength=0.4,
        ncol=3,
    )

    ax.yaxis.grid(True, linestyle="--", which="both", color="gray", alpha=0.7)
    ax.set_ylim(0, 0.4)

    ax.set_xlabel(
        "Domain",
        fontdict={
            "size": 26,
        },
    )
    ax.set_ylabel(
        "Weight",
        fontdict={
            "size": 26,
        },
    )

    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed, n_samples, alpha=alpha)}_optimal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )


def mk_output_prefix(
    output_dir: str,
    metric: str,
    regression_type: str,
    train_split: tuple[float],
    n_test: int,
    split_seed: int,
    n_samples: int,
    alpha: Optional[float] = None,
) -> str:
    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", s)

    train_split_str = [str(t) for t in train_split]
    return (
        os.path.join(output_dir, sanitize(metric))
        + (f"_alpha_{str(alpha).replace('.', '_')}" if alpha and alpha != 1.0 else "")
        + (f"_{regression_type}_reg" if regression_type != "lightgbm" else "")
        + (f"_trainsplit_{'_'.join(train_split_str)}" if train_split[0] != 1.0 else "")
        + (f"_ntest_{n_test}" if n_test != 0 else "")
        + (f"_seed_{split_seed}" if split_seed != 0 else "")
        + (f"_{n_samples}_samples" if n_samples != 10 else "")
    )


def save_eval_config(eval_config: dict, output_dir: str, custom_name: Optional[str]= None) -> str:
    # Serialize dict in a stable way
    config_str = json.dumps(eval_config, sort_keys=True)

    # Hash it (short hash for readability)
    hash_str = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]


    if custom_name is not None:
        hash_str = hash_str + f"_{custom_name}"

    # Create directory
    folder_path = os.path.join(output_dir, hash_str)
    os.makedirs(folder_path, exist_ok=True)

    # Save config JSON inside
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(eval_config, f, indent=2)

    print(f"[INFO] Saved config to {config_path}")
    return folder_path


def filter_constrained_swarm(
    final_cookbook_path: Path, run_ratios: List, run_metrics: List
) -> Tuple[List, List]:
    assert (
        final_cookbook_path is not None
    ), "final_cookbook_path must be set to determine how to construct swarm to be unconstrained."

    with open(final_cookbook_path, "r") as f:
        data = yaml.safe_load(f)

    final_config = CookbookExperimentConfig(**data, path=final_cookbook_path)
    desired_tokens = final_config.max_tokens

    # logger.warning(f"Using hardcoded token counts!")
    # with open("cache/priors_cache_217af510306ab626a507634c64ca7ca8.json", "r") as f:
    #    available_tokens_per_source = json.load(f)['token_counts']

    token_universe = get_token_counts_and_ratios(
        final_config.dataset.sources, final_config.dataset.dtype, True
    )
    available_tokens_per_source = {
        path: relative_size * token_universe[1] for path, relative_size in token_universe[0].items()
    }

    original_swarm_size = len(run_ratios)

    valid_runs = [
        run["run"]
        for run in run_ratios
        if all(
            [
                run[source] * desired_tokens <= num_available_tokens
                for source, num_available_tokens in available_tokens_per_source.items()
            ]
        )
    ]

    run_ratios = [run for run in run_ratios if run["run"] in valid_runs]
    run_metrics = [run for run in run_metrics if run["run"] in valid_runs]

    logger.info(
        f"Removed {original_swarm_size - len(run_ratios)} swarm runs that would repeat tokens at the final run scale."
    )

    return run_ratios, run_metrics


def calculate_priors_with_manual(
    source_configs: list[SourceConfig],
    dtype,
    use_cache: bool,
    manual_prior: Optional[dict[str, float]] = None,
    fixed_source_weights: Optional[dict[str, float]] = None,
):
    priors = calculate_priors(
        source_configs=source_configs,
        dtype=dtype,
        use_cache=use_cache,
    )
    if manual_prior is not None or fixed_source_weights is not None:
        fixed_weights = manual_prior if manual_prior is not None else fixed_source_weights
        logger.info(f"Adjusting priors with manual prior weights: {fixed_weights}")
        for source_config in sorted(source_configs, key=lambda x: x.name):
            if source_config.topics:
                # adjust each topic weight by the manual prior
                try:
                    weights = np.array(
                        [
                            priors[0][f"{source_config.name}:{topic.name}"]
                            for topic in source_config.topics
                        ]
                    )
                except KeyError:
                    print(priors[0].keys())
                    print(f"Source config: {source_config.name}")
                    print(f"Topics: {[topic.name for topic in source_config.topics]}")

                    raise KeyError()

                normalized_weights = weights / weights.sum()
                if fixed_weights is not None and source_config.name in fixed_weights:
                    for i, topic in enumerate(source_config.topics):
                        priors[0][f"{source_config.name}:{topic.name}"] = (
                            fixed_weights[source_config.name] * normalized_weights[i]
                        )
            else:
                # directly overwrite source weight with manual prior
                if fixed_weights is not None and source_config.name in fixed_weights:
                    priors[0][source_config.name] = fixed_weights[source_config.name]

    # if we conditioned any of the topic-level weights, we collapse them into source-level weights
    for source_config in source_configs:
        if source_config.topics:
            if all([
                getattr(topic, 'weight', None) is not None or getattr(topic, 'target_ratio', None) is not None
                for topic in source_config.topics
            ]):
                # update prior with hardcoded topic weights or target_ratios
                source_weight = sum(
                    [
                        priors[0][f"{source_config.name}:{topic.name}"]
                        for topic in source_config.topics
                    ]
                )
                for topic in source_config.topics:
                    value = getattr(topic, 'weight', None)
                    if value is None:
                        value = getattr(topic, 'target_ratio', None)
                    priors[0][f"{source_config.name}:{topic.name}"] = value * source_weight

    original_prior = deepcopy(priors)

    for source_config in source_configs:
        if source_config.topics:
            if all([
                getattr(topic, 'weight', None) is not None or getattr(topic, 'target_ratio', None) is not None
                for topic in source_config.topics
            ]):
                source_weight = sum(
                    [
                        priors[0][f"{source_config.name}:{topic.name}"]
                        for topic in source_config.topics
                    ]
                )
                for topic in source_config.topics:
                    del priors[0][f"{source_config.name}:{topic.name}"]
                priors[0][source_config.name] = source_weight

    return priors, original_prior


def aggregate_mmlu(metrics: pd.DataFrame, metrics_to_index: list):
    logger.info("Aggregating MMLU metrics...")

    def add_weighted_dot_column(
        df: pd.DataFrame, weights: dict, output_col: str, metrics_to_index: list
    ):
        weight_series = pd.Series(weights)
        df[output_col] = df[weight_series.index].dot(weight_series)
        df.drop(columns=weight_series.index, inplace=True)
        columns_to_remove = set(weight_series.index)
        metrics_to_index = [col for col in metrics_to_index if col not in columns_to_remove]
        metrics_to_index.append(output_col)
        return metrics_to_index

    stem_weights = {
        "mmlu_abstract_algebra:rc::olmes": 0.03313452617627568,
        "mmlu_astronomy:rc::olmes": 0.05036447978793903,
        "mmlu_college_biology:rc::olmes": 0.04771371769383698,
        "mmlu_college_chemistry:rc::olmes": 0.03313452617627568,
        "mmlu_college_computer_science:rc::olmes": 0.03313452617627568,
        "mmlu_college_mathematics:rc::olmes": 0.03313452617627568,
        "mmlu_college_physics:rc::olmes": 0.033797216699801194,
        "mmlu_computer_security:rc::olmes": 0.03313452617627568,
        "mmlu_conceptual_physics:rc::olmes": 0.07786613651424784,
        "mmlu_electrical_engineering:rc::olmes": 0.04804506295559974,
        "mmlu_elementary_mathematics:rc::olmes": 0.12524850894632206,
        "mmlu_high_school_biology:rc::olmes": 0.10271703114645461,
        "mmlu_high_school_chemistry:rc::olmes": 0.06726308813783963,
        "mmlu_high_school_computer_science:rc::olmes": 0.03313452617627568,
        "mmlu_high_school_mathematics:rc::olmes": 0.08946322067594434,
        "mmlu_high_school_physics:rc::olmes": 0.050033134526176276,
        "mmlu_high_school_statistics:rc::olmes": 0.07157057654075547,
        "mmlu_machine_learning:rc::olmes": 0.03711066931742876,
    }
    other_weights = {
        "mmlu_anatomy:rc::olmes": 0.04164096236890808,
        "mmlu_business_ethics:rc::olmes": 0.030845157310302282,
        "mmlu_clinical_knowledge:rc::olmes": 0.08173966687230105,
        "mmlu_college_medicine:rc::olmes": 0.05336212214682295,
        "mmlu_global_facts:rc::olmes": 0.030845157310302282,
        "mmlu_human_aging:rc::olmes": 0.06878470080197409,
        "mmlu_management:rc::olmes": 0.03177051202961135,
        "mmlu_marketing:rc::olmes": 0.07217766810610735,
        "mmlu_medical_genetics:rc::olmes": 0.030845157310302282,
        "mmlu_miscellaneous:rc::olmes": 0.24151758173966686,
        "mmlu_nutrition:rc::olmes": 0.09438618136952498,
        "mmlu_professional_accounting:rc::olmes": 0.08698334361505243,
        "mmlu_professional_medicine:rc::olmes": 0.08389882788402221,
        "mmlu_virology:rc::olmes": 0.05120296113510179,
    }
    social_sciences_weights = {
        "mmlu_econometrics:rc::olmes": 0.03704907377315567,
        "mmlu_high_school_geography:rc::olmes": 0.06434839129021774,
        "mmlu_high_school_government_and_politics:rc::olmes": 0.06272343191420214,
        "mmlu_high_school_macroeconomics:rc::olmes": 0.12674683132921677,
        "mmlu_high_school_microeconomics:rc::olmes": 0.07734806629834254,
        "mmlu_high_school_psychology:rc::olmes": 0.17712057198570036,
        "mmlu_human_sexuality:rc::olmes": 0.04257393565160871,
        "mmlu_professional_psychology:rc::olmes": 0.19889502762430938,
        "mmlu_public_relations:rc::olmes": 0.03574910627234319,
        "mmlu_security_studies:rc::olmes": 0.07962300942476438,
        "mmlu_sociology:rc::olmes": 0.0653233669158271,
        "mmlu_us_foreign_policy:rc::olmes": 0.032499187520311994,
    }
    humanities_weights = {
        "mmlu_formal_logic:rc::olmes": 0.026780021253985122,
        "mmlu_high_school_european_history:rc::olmes": 0.03506907545164718,
        "mmlu_high_school_us_history:rc::olmes": 0.04335812964930925,
        "mmlu_high_school_world_history:rc::olmes": 0.050371944739638685,
        "mmlu_international_law:rc::olmes": 0.0257173219978746,
        "mmlu_jurisprudence:rc::olmes": 0.022954303931987247,
        "mmlu_logical_fallacies:rc::olmes": 0.034643995749202974,
        "mmlu_moral_disputes:rc::olmes": 0.07353878852284804,
        "mmlu_moral_scenarios:rc::olmes": 0.1902231668437832,
        "mmlu_philosophy:rc::olmes": 0.06609989373007438,
        "mmlu_prehistory:rc::olmes": 0.06886291179596174,
        "mmlu_professional_law:rc::olmes": 0.32603613177470775,
        "mmlu_world_religions:rc::olmes": 0.03634431455897981,
    }

    metrics_to_index = add_weighted_dot_column(metrics, stem_weights, "mmlu_stem", metrics_to_index)
    metrics_to_index = add_weighted_dot_column(
        metrics, other_weights, "mmlu_other", metrics_to_index
    )
    metrics_to_index = add_weighted_dot_column(
        metrics, social_sciences_weights, "mmlu_social_sciences", metrics_to_index
    )
    metrics_to_index = add_weighted_dot_column(
        metrics, humanities_weights, "mmlu_humanities", metrics_to_index
    )

    return metrics, metrics_to_index



def swarm_config_from_cookbook_or_regmixer_path(config: Path, use_cookbook: bool) -> Union[ExperimentConfig, CookbookExperimentConfig]:
    """
    Load configuration from a cookbook or regmixer path.
    """
    with open(config, "r") as f:
        data = yaml.safe_load(f)
    if use_cookbook:
        return CookbookExperimentConfig(**data, path=config)
    else:
        return ExperimentConfig(**data)
