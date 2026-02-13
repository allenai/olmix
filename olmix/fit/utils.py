import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import warnings
from copy import deepcopy
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import boto3
import cvxpy as cp
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from tqdm import tqdm
from wandb.apis.public import Run

from olmix.aliases import ExperimentConfig, SourceConfig, compute_max_tokens, get_model_num_params
from olmix.fit.law import ScalingLaw
from olmix.launch.synthesize_mixture import calculate_priors
from olmix.plots import BASE_OUTPUT_DIR


def get_target_tokens(
    target_tokens: int | None = None,
    target_chinchilla_multiple: float | None = None,
    target_model_id: str | None = None,
) -> int | None:
    """Compute target tokens for the final run.

    Uses target_tokens directly if set, otherwise computes from
    target_chinchilla_multiple and target_model_id.

    Returns:
        Target token count for constraint optimization, or None if not configured.

    Raises:
        ValueError: If target_chinchilla_multiple is set but target_model_id is not.
    """
    if target_tokens is not None:
        return target_tokens
    if target_chinchilla_multiple is not None:
        if target_model_id is None:
            raise ValueError("target_model_id required when using target_chinchilla_multiple")
        num_params = get_model_num_params(target_model_id)
        return compute_max_tokens(target_chinchilla_multiple, num_params)
    return None


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


if platform.system() == "Darwin":  # Darwin is the system name for macOS
    import multiprocessing as mp

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    mp.set_start_method("spawn", force=True)


def get_token_counts_and_ratios(
    source_configs: list[SourceConfig],
    dtype,
    use_cache: bool = True,
) -> tuple[dict[str, float], int]:
    """
    Get token counts and ratios for a list of source configurations.

    This is a compatibility wrapper around calculate_priors that provides
    the same interface as the old cookbook.utils.data.get_token_counts_and_ratios.

    Args:
        source_configs: List of SourceConfig objects
        dtype: Data type for numpy datasets
        use_cache: Whether to use cached results

    Returns:
        Tuple of (relative_sizes dict, total_tokens)
    """
    priors, total_tokens, _ = calculate_priors(source_configs, dtype, use_cache)
    return priors, total_tokens


def compute_constraints_from_config(
    config: ExperimentConfig,
    *,
    target_tokens: int,
    repetition_factor: float = 5.0,
    use_cache: bool = True,
) -> tuple[int, dict[str, float], float]:
    """Compute constraints from config's sources and target settings.

    Args:
        config: ExperimentConfig (used for sources and dtype)
        target_tokens: Target token count for the final run
        repetition_factor: Max repetition factor for constraint checking
        use_cache: Whether to use cached token counts

    Returns:
        Tuple of (target_tokens, available_tokens_per_source, repetition_factor)
    """
    token_universe = get_token_counts_and_ratios(config.data.sources, config.data.dtype, use_cache)
    available_tokens_per_source = {
        path: relative_size * token_universe[1] for path, relative_size in token_universe[0].items()
    }
    return target_tokens, available_tokens_per_source, repetition_factor


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
    model: Any

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
        return mixing_law(torch.tensor(x, dtype=torch.float), torch.tensor(self.model, dtype=torch.float)).numpy()


class GPRegressor(Regressor):
    def __init__(self, **kwargs):
        # Default hyperparameters (can be overridden via kwargs)
        length_scale = kwargs.get("length_scale", 1.0)
        length_scale_bounds = kwargs.get("length_scale_bounds", (1e-2, 1e2))
        constant_value = kwargs.get("constant_value", 1.0)
        constant_value_bounds = kwargs.get("constant_value_bounds", (1e-3, 1e3))
        noise_level = kwargs.get("noise_level", 0.1)
        noise_level_bounds = kwargs.get("noise_level_bounds", (1e-5, 1e1))
        n_restarts = kwargs.get("n_restarts_optimizer", 10)
        normalize_y = kwargs.get("normalize_y", True)

        # Build kernel: λ * RBF(σ) + WhiteKernel(σ_ε)
        kernel = ConstantKernel(constant_value, constant_value_bounds=constant_value_bounds) * RBF(
            length_scale=length_scale, length_scale_bounds=length_scale_bounds
        ) + WhiteKernel(noise_level=noise_level, noise_level_bounds=noise_level_bounds)

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=n_restarts,
            normalize_y=normalize_y,
            random_state=kwargs.get("random_state", None),
        )

    def fit(self, x, y, idx, **kwargs):
        target = y[:, idx]
        self.model.fit(x, target)

        # Optionally print optimized hyperparameters
        if kwargs.get("verbose", False):
            print(f"Task {idx} - Optimized kernel: {self.model.kernel_}")
            print(f"Task {idx} - Log marginal likelihood: {self.model.log_marginal_likelihood_value_:.3f}")


class AutoscaleRegressor(Regressor):
    """
    Fits y(p) = sum_i (N0_i + N * p_i)^(-gamma_i) + L. This equation is taken from the Autoscale paper.

    The autoscale paper is a bit unclear about how it actually fits to OOD domains; its notebook only shows fitting domain i's mix to an aggregated validation loss, for each i.
    But, we took the main equation of the paper and directly fit it, as a good-faith effort to implement their method for OOD evaluation.

    x: (n_samples, m) probability vectors p (rows typically sum to 1)
    y: (n_samples, n_targets) or (n_samples,)
    idx: column index of y to fit when y is 2D
    """

    def __init__(self, requested_tokens, max_nfev=50000, verbose=False, params=None, **kwargs):
        if requested_tokens is None:
            raise ValueError("requested_tokens must be provided for AutoscaleRegressor.")
        self.N = float(requested_tokens)
        self.max_nfev = int(max_nfev)
        self.verbose = bool(verbose)

        if params is not None:
            self.alpha_ = params.get("alpha", None)
            self.N0_ = self.alpha_ * self.N if self.alpha_ is not None else None
            self.gamma_ = params.get("gamma", None)
            self.L_ = params.get("L", None)
        else:
            self.alpha_ = None  # learned N0/N, shape (m,)
            self.N0_ = None  # learned N0, shape (m,)
            self.gamma_ = None  # learned gamma, shape (m,)
            self.L_ = None  # learned L, scalar
        self.result_ = None  # scipy result

    def _predict_given_params(self, X, alpha, gamma, L):
        # base = N0 + N p = N(alpha + p)
        base = self.N * (alpha[None, :] + X)
        base = np.maximum(base, 1e-30)
        return np.sum(base ** (-gamma[None, :]), axis=1) + L

    def fit(self, x, y, idx, **kwargs):
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, m). Got shape {X.shape}")

        y_target = Y if Y.ndim == 1 else Y[:, idx]

        n, m = X.shape
        if y_target.shape[0] != n:
            raise ValueError("x and y have inconsistent number of rows")

        # --- init: alpha ~ 1/m, gamma positive, L free ---
        alpha_init = np.full(m, 1.0 / m)
        gamma_init = np.full(m, 0.5)
        L_init = float(np.min(y_target) * 0.1)
        p0 = np.concatenate([alpha_init, gamma_init, [L_init]])

        # --- bounds: alpha>0, gamma>0, L unrestricted ---
        lb = np.concatenate([np.full(m, 1e-12), np.full(m, 1e-12), [-np.inf]])
        ub = np.concatenate([np.full(m, np.inf), np.full(m, np.inf), [np.inf]])

        L_ub = float(np.min(y_target))
        lb[-1] = 0.0
        ub[-1] = L_ub

        def resid(params):
            alpha = params[:m]
            gamma = params[m : 2 * m]
            L = params[-1]
            return self._predict_given_params(X, alpha, gamma, L) - y_target

        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=self.max_nfev)

        params_hat = res.x
        self.alpha_ = params_hat[:m]
        self.gamma_ = params_hat[m : 2 * m]
        self.L_ = float(params_hat[-1])
        self.N0_ = self.alpha_ * self.N
        self.result_ = res

        if True:  # self.verbose or kwargs.get("verbose", False):
            yhat = self._predict_given_params(X, self.alpha_, self.gamma_, self.L_)
            rmse = float(np.sqrt(np.mean((yhat - y_target) ** 2)))
            print(f"[AutoscaleRegressor] idx={idx} success={res.success} rmse={rmse:.6g}")
            print("  alpha (N0/N):", self.alpha_)
            print("  N0:", self.N0_)
            print("  gamma:", self.gamma_)
            print("  L:", self.L_)

        return self

    def predict(self, x, **kwargs):
        if self.alpha_ is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, m). Got shape {X.shape}")
        return self._predict_given_params(X, self.alpha_, self.gamma_, self.L_)

    def get_params(self):
        return {
            "alpha": self.alpha_,
            "gamma": self.gamma_,
            "L": self.L_,
        }


class BimixRegressor(Regressor):
    """
    Fits f(x) = sum_i F_i * x_i^(-alpha_i) --- this is derived from the BiMix paper.
    However, the bimix paper only fits validation loss of domain i vs mixture weight on domain i, so we add a summation to extend it to OOD evaluation.
    Also, bimix models the number of steps. For our setup, we don't need to model this, so the equation collapses into a classical power law.

    x: (n_samples, m) probability vectors (rows should sum to 1; all entries should be > 0)
    y: (n_samples, n_targets) or (n_samples,)
    idx: which column of y to fit when y is 2D
    """

    def __init__(self, eps=1e-12, max_nfev=50000, verbose=False, **kwargs):
        self.eps = float(eps)
        self.max_nfev = int(max_nfev)
        self.verbose = bool(verbose)

        self.F_ = None  # shape (m,)
        self.alpha_ = None  # shape (m,)
        self.result_ = None  # scipy result

    def _predict_given_params(self, X, F, alpha):
        X = np.asarray(X, dtype=float)
        Xc = np.clip(X, self.eps, 1.0)  # avoid 0^(-alpha)
        return np.sum(F[None, :] * (Xc ** (-alpha[None, :])), axis=1)

    def fit(self, x, y, idx, **kwargs):
        X = np.asarray(x, dtype=float)
        Y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, m). Got shape {X.shape}")

        y_target = Y if Y.ndim == 1 else Y[:, idx]

        n, m = X.shape
        if y_target.shape[0] != n:
            raise ValueError("x and y have inconsistent number of rows")

        # ---- init ----
        # Simple, robust-ish defaults: F around mean(y)/m, alpha around 1
        F_init = np.full(m, max(np.mean(y_target), self.eps) / m)
        alpha_init = np.full(m, 1.0)
        p0 = np.concatenate([F_init, alpha_init])

        # ---- bounds ----
        # F_i >= 0, alpha_i >= 0
        lb = np.concatenate([np.zeros(m), np.zeros(m)])
        ub = np.concatenate([np.full(m, np.inf), np.full(m, np.inf)])

        def resid(params):
            F = params[:m]
            alpha = params[m : 2 * m]
            yhat = self._predict_given_params(X, F, alpha)
            return yhat - y_target

        res = least_squares(resid, p0, bounds=(lb, ub), max_nfev=self.max_nfev)

        params_hat = res.x
        self.F_ = params_hat[:m]
        self.alpha_ = params_hat[m : 2 * m]
        self.result_ = res

        if self.verbose or kwargs.get("verbose", False):
            yhat = self._predict_given_params(X, self.F_, self.alpha_)
            rmse = float(np.sqrt(np.mean((yhat - y_target) ** 2)))
            print(f"[BimixRegressor] idx={idx} success={res.success} rmse={rmse:.6g}")
            print("  F:", self.F_)
            print("  alpha:", self.alpha_)

        return self

    def predict(self, x, **kwargs):
        if self.F_ is None or self.alpha_ is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(x, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"x must be 2D (n_samples, m). Got shape {X.shape}")
        return self._predict_given_params(X, self.F_, self.alpha_)


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


def mixing_law(x, param, **kwargs):
    log_c_i = param[0]
    t_i = param[1:]
    result = torch.exp(log_c_i) + torch.exp(torch.matmul(x, t_i))
    return result


def init_params_log_linear_law(idx, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10):  # originally (-2, 1.5, 10)
        for _ in range(30):
            ts = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            yield [log_c_i, *ts]


def init_params_log_nonlinear_law(idx, B_mask, num_domains=3):
    for log_c_i in np.linspace(-2, 1.5, 10):  # originally (-2, 1.5, 10)
        for _ in range(30):
            lin_params = [-np.random.rand() if i == idx else np.random.rand() * 0.1 for i in range(num_domains)]
            quadratic_params = [np.random.rand() * 0.1 for i in range(len(B_mask))]
            yield [log_c_i, *lin_params, *quadratic_params]


REGRESSION_TYPES = {
    "lightgbm": LightGBMRegressor,
    "log_linear": LogLinearRegressor,
    "search": SearchRegressor,
    "gp": GPRegressor,
    "autoscale": AutoscaleRegressor,
    "bimix": BimixRegressor,
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
        ratios: pd.DataFrame,
        num_samples: int = 1_000_000,
        seed: int = 1337,
        search_iterations: int = 10,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        swarm_config: ExperimentConfig | None = None,
        obj_weights: list | None = None,
        temperature: float | None = None,
        make_worst_mix: bool = False,
        requested_tokens: int | None = None,
        target_tokens: int | None = None,
        repetition_factor: float = 5.0,
        **kwargs,
    ) -> np.ndarray:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)

        min_weight = 1e-5
        min_dirichlet = 1
        max_dirichlet = 100
        search_dirichlet_factor = 2.0

        search_prior = np.array(list(prior_distributions.values()))

        if temperature is not None:
            search_prior = search_prior**temperature
            search_prior = search_prior / np.sum(search_prior)

        logger.info(f"Prior for sampling is {search_prior}")

        best_weights = np.zeros(len(prior_distributions))

        if constrain_objective:
            if swarm_config is None:
                raise ValueError("swarm_config required for constrain_objective")
            if target_tokens is None:
                raise ValueError("target_tokens required for constrain_objective")
            desired_tokens, available_tokens_per_source, rep_factor = compute_constraints_from_config(
                swarm_config, target_tokens=target_tokens, repetition_factor=repetition_factor
            )
            if requested_tokens is not None:
                desired_tokens = requested_tokens
            # Ensure order of sources in simulations matches constraint dictionary
            available_tokens_per_source = {
                source: available_tokens_per_source[source] for source, _ in prior_distributions.items()
            }

        # Multi-step search leveraging iterative prior results
        for search_step in tqdm(range(search_iterations), desc=f"Searching in {num_samples} candidate samples"):
            offset = np.log(search_dirichlet_factor * (search_step + 1))
            alphas = np.exp(
                np.random.uniform(
                    low=np.log(min_dirichlet) + offset,
                    high=np.log(max_dirichlet) + offset,
                    size=num_samples,
                )
            )

            # generate simulations by sampling from dirichlet distribution with parameter prior * alpha
            simulations = (
                torch.distributions.Dirichlet(torch.from_numpy(alphas[:, None] * search_prior)).sample().numpy()
            )

            if constrain_objective:
                original_simulation_size = len(simulations)

                # Simple elementwise constraint checking: weight[source] * target_tokens <= available[source] * repetition_factor
                token_usage = simulations * desired_tokens
                token_limits = np.array(list(available_tokens_per_source.values())) * rep_factor
                valid_mask = (token_usage <= token_limits).all(axis=1)

                filtered_count = np.sum(~valid_mask)
                if filtered_count > 0:
                    logger.info(f"Filtering out {filtered_count} simulations that exceed token constraints")

                simulations = simulations[valid_mask]

                logger.info(
                    f"Removed {original_simulation_size - len(simulations)} out of {original_simulation_size} simulations that would repeat tokens at the final run scale."
                )

                if len(simulations) == 0:
                    continue

            predictions = np.array([reg.predict(simulations) for reg in predictor])
            if opt_avg_metric:
                if obj_weights is not None:
                    objs = np.average(predictions, axis=0, weights=obj_weights)
                    logger.info(f"Computing weighted average of predictions using {obj_weights}")
                else:
                    objs = predictions.mean(axis=0)
            else:
                objs = predictor[index].predict(simulations)

            if make_worst_mix:
                best_mask = (objs.max() - objs) < 1e-3
                print(objs.max())
            else:
                best_mask = (objs - objs.min()) < 1e-3
                print(objs.min())
            best_weights = simulations[best_mask].mean(0)

            # Zero out weights below min_weight threshold and normalize
            best_weights[best_weights < min_weight] = 0.0
            best_weights /= best_weights.sum()

            search_prior = (best_weights + search_prior) / 2

            predicted_best_performance = sum([pred.predict([best_weights]) for pred in predictor])
            logger.info(
                f"Current best weights is: {best_weights} with predicted performance {predicted_best_performance}, and search prior is: {search_prior}"
            )

        if constrain_objective:
            # Verify best weights satisfy constraints
            token_usage = best_weights * desired_tokens
            token_limits = np.array(list(available_tokens_per_source.values())) * rep_factor
            if not all(token_usage <= token_limits):
                raise ValueError("Best weights are out of bounds!")

        return best_weights


class SearchProposer(Proposer):
    def propose(
        self,
        index: int,
        predictor: list[SearchRegressor],
        prior_distributions: dict,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        swarm_config: ExperimentConfig | None = None,
        requested_tokens: int | None = None,
        target_tokens: int | None = None,
        repetition_factor: float = 5.0,
        **kwargs,
    ):
        if constrain_objective:
            if swarm_config is None:
                raise ValueError("swarm_config required for constrain_objective")
            if target_tokens is None:
                raise ValueError("target_tokens required for constrain_objective")
            desired_tokens, available_tokens_per_source, rep_factor = compute_constraints_from_config(
                swarm_config, target_tokens=target_tokens, repetition_factor=repetition_factor
            )
            if requested_tokens is not None:
                desired_tokens = requested_tokens
            # Ensure order matches prior_distributions
            available_tokens_per_source = {
                source: available_tokens_per_source[source] for source, _ in prior_distributions.items()
            }

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
                token_limits = np.array(list(available_tokens_per_source.values())) * rep_factor

                if (token_usage <= token_limits).all() and pred < best_performance:
                    best_performance = pred
                    best_weights = weight
            else:
                if pred < best_performance:
                    best_performance = pred
                    best_weights = weight

        return best_weights


class LogLinearExactProposer(Proposer):
    def propose(
        self,
        predictor: list[SearchRegressor],
        prior_distributions: dict,
        opt_avg_metric: bool = False,
        constrain_objective: bool = False,
        swarm_config: ExperimentConfig | None = None,
        kl_reg: float | None = 0.1,
        obj_weights: list | None = None,
        requested_tokens: int | None = None,
        manual_kl: dict | None = None,
        target_tokens: int | None = None,
        repetition_factor: float = 5.0,
        **kwargs,
    ):
        assert opt_avg_metric, "LogLinearExactProposer only supports opt_avg_metric=True"
        if kl_reg is None:
            raise ValueError("kl_reg must be provided for LogLinearExactProposer")

        caps = None
        if constrain_objective:
            if swarm_config is None:
                raise ValueError("swarm_config required for constrain_objective")
            if target_tokens is None:
                raise ValueError("target_tokens required for constrain_objective")
            desired_tokens, available_tokens_per_source, rep_factor = compute_constraints_from_config(
                swarm_config, target_tokens=target_tokens, repetition_factor=repetition_factor
            )
            if requested_tokens is not None:
                desired_tokens = requested_tokens
            # Ensure order matches prior_distributions
            available_tokens_per_source = {
                source: available_tokens_per_source[source] for source, _ in prior_distributions.items()
            }
            caps = np.array(list(available_tokens_per_source.values())) * rep_factor / desired_tokens

        np.array([p.model[0] for p in predictor])  # (n,)
        A = np.array([p.model[1:] for p in predictor])  # (n, d)
        n, d = A.shape
        weights = np.ones(n) / n if obj_weights is None else np.array(obj_weights)

        x = cp.Variable(d)

        if manual_kl is not None:
            logger.info(f"Using manual KL prior distribution: {manual_kl}")
            q = np.array(list(manual_kl.values()))
        else:
            logger.info(f"Using prior distribution for KL: {prior_distributions}")
            q = np.array(list(prior_distributions.values()))

        q = np.asarray(q, dtype=float)
        eps = 1e-12
        q = np.maximum(q, eps)  # ensure strictly positive
        q = q / q.sum()

        # c_i doesn’t affect the argmin, but harmless to include if you want the value
        # obj = cp.sum(cp.multiply(weights, C) + cp.exp(A @ x))
        # obj = cp.sum(cp.multiply(weights, cp.exp(A @ x)))       # identical argmin
        loss = cp.sum(cp.multiply(weights, cp.exp(A @ x)))

        # KL(x || q) = sum x*log(x/q) = sum rel_entr(x, q)
        kl = cp.sum(cp.rel_entr(x, q))

        obj = loss + kl_reg * kl

        constraints = [x >= 0, cp.sum(x) == 1]
        if constrain_objective:
            constraints.append(x <= caps)

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(solver="ECOS", verbose=True)  # ECOS or SCS are good

        print(prob.value, prob.status)

        return x.value


PROPOSER_TYPES = {"simulation": SimulationProposer, "search": SearchProposer, "exact": LogLinearExactProposer}


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
    requested_tokens: int | None = None,
) -> Regressor:
    logger.info(f"Building regression model, index: {idx}")
    reg = REGRESSION_TYPES[regression_type](requested_tokens=requested_tokens)
    reg.fit(X_train, Y_train, idx, early_stopping=early_stopping)
    return reg


def get_runs_without_wandb(full_group_name, dashboard="olmo-3-evals"):  # "regmixer"):
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
    eval_metrics: list[str],
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
            all_runs.append(mk_run_history(run, eval_metrics))

    all_runs = sorted(all_runs, key=lambda run: run.display_name.lower())
    if not no_cache:
        with open(cache_path, "w") as f:
            json.dump([run.as_dict() for run in all_runs], f)

    return all_runs


def mk_run_history(run: Run, eval_metrics: list[str]) -> Any:
    summary = [{metric: run.summary[metric] for metric in eval_metrics if metric in run.summary}]
    return mk_run_instance(run, summary)


def mk_run_from_json(run: dict) -> RunInstance:
    return RunInstance(
        id=run["id"],
        display_name=run["display_name"],
        config=run["config"],
        samples=pd.DataFrame(run["samples"]),
        state=run["state"],
    )


def mk_run_instance(run: Run, history: list[Any]) -> RunInstance:
    samples = pd.DataFrame.from_records(history).tail(1)
    return RunInstance(
        id=run.id,
        display_name=run.display_name,
        config=run.config,
        samples=samples,
        state=run.state,
    )


def mk_run_metrics(
    history,
    metrics: tuple[str, list[str]],
    display_name: str,
    average: bool = False,
    dashboard: list[str] | None = None,  # ["olmo-3-evals"]
) -> dict[str, float]:
    if dashboard is None:
        dashboard = ["regmixer"]
    df = pd.DataFrame(history)
    results = {}
    group_name, group_metrics = metrics
    in_loop_tasks = [task for task in df.columns if task in group_metrics]
    offline_tasks = [task for task in group_metrics if task not in in_loop_tasks]
    if average:
        raise NotImplementedError("Averaging the task is implemented but out of date!")
        result = np.mean([df.loc[:, metric_name].tail(1).mean() for metric_name in group_metrics])
        results[group_name] = result
    else:
        for metric_name in in_loop_tasks:
            results[metric_name] = df.loc[:, metric_name].tail(1).mean()

        if len(offline_tasks) > 0:
            # need to obtain offline results
            for d in dashboard:
                logger.info(f"Getting offline results for {display_name} in {d} dashboard")
                offline_results = get_offline_evals(
                    display_name,
                    offline_tasks,
                    group_name,
                    dashboard=d,
                )
                results.update(offline_results)

    return results


def get_offline_evals_from_dashboard(display_name, tasks, dashboard):
    command = [
        "olmo-cookbook-eval",
        "results",
        "--dashboard",
        f"{dashboard}",
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

    df = df[df["name"].str.contains(display_name)]
    assert len(df) == 1

    df = df[tasks]
    return df.to_dict()


def get_offline_evals(
    display_name,
    tasks,
    group_name,
    dashboard="regmixer",
):  # "olmo-3-evals"):# "regmixer"):
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

    # all_available_tasks = [data['task_config'].get('metadata', {}).get('alias')  for data in all_jsonl_data]
    # logger.info(f"Available tasks in JSONL data for {display_name}:")
    # for task in sorted(all_available_tasks):
    #    print(f'"{task}",')
    # raise ValueError()

    offline_results = {}
    for _i, task in enumerate(tasks):
        task_data = [
            data
            for data in all_jsonl_data
            if (data["task_config"].get("metadata", {}).get("alias") == task or data["task_name"] == task)
        ]
        if len(task_data) == 0:
            task = task.replace("bpb:", "")
            task_data = [
                data
                for data in all_jsonl_data
                if (data["task_config"].get("metadata", {}).get("alias") == task or data["task_name"] == task)
            ]
            if len(task_data) == 0:
                task = task + ":full"
                task_data = [
                    data
                    for data in all_jsonl_data
                    if (data["task_config"].get("metadata", {}).get("alias") == task or data["task_name"] == task)
                ]
                if len(task_data) == 0:
                    all_available_tasks = [
                        data["task_config"].get("metadata", {}).get("alias") for data in all_jsonl_data
                    ]

                    if (
                        task.replace(":full", "")
                        in [
                            "ultrachat_masked_ppl",
                            "wildchat_masked_ppl",
                            "qasper_yesno:rc::olmes",
                            "sciriff_yesno:rc::olmes",
                            "lab_bench_dbqa",
                            "lab_bench_protocolqa",
                            "medqa_en:rc::none",
                        ]
                        and group_name == "pretraining_tasks_for_paper"
                    ):
                        continue
                    else:
                        logger.warning(
                            f"Task {task} not found in JSONL data for {display_name}. Available tasks: {all_available_tasks}"
                        )
                        continue
                else:
                    logger.info(
                        f"Task {task} found in JSONL data for {display_name} with alias {task_data[0]['task_config'].get('metadata', {}).get('alias')}"
                    )

        data = task_data[0]
        if "bits_per_byte_corr" in data["metrics"]:
            name = data["task_config"]["metadata"]["alias"].replace("bpb:", "").replace(":full", "")
            offline_results[name] = data["metrics"]["bits_per_byte_corr"]
            # logger.info(
            #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte_corr {data['metrics']['bits_per_byte_corr']}"
            # )
        elif "bits_per_byte_corr_macro" in data["metrics"]:
            name = data["task_config"]["metadata"]["alias"].replace("bpb:", "").replace(":full", "")
            offline_results[name] = data["metrics"]["bits_per_byte_corr_macro"]
            # logger.info(
            #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte_corr_macro {data['metrics']['bits_per_byte_corr_macro']}"
            # )
        elif "bits_per_byte" in data["metrics"]:
            name = data["task_name"].replace("bpb:", "").replace(":full", "")
            offline_results[name] = data["metrics"]["bits_per_byte"]
            # logger.info(
            #    f"Task {name} found in JSONL data for {display_name} with bits_per_byte {data['metrics']['bits_per_byte']}"
            # )
        else:
            logger.warning(
                f"{data['task_name']} does not have bits_per_byte_corr or bits_per_byte_corr_macro in metrics {data['metrics'].keys()}"
            )

    return offline_results


def mk_weights_from_config(config: dict, priors: tuple, display_name: str, patched: bool = False) -> dict[str, float]:
    source_mixture_config = config.get("dataset", {}).get("source_mixture_config", {})

    sources = (
        source_mixture_config.get("source_configs") or source_mixture_config.get("source_list").get("sources") or []
    )

    source_configs = {source["source_name"]: source for source in sources}

    prefixes = ["dclm", "s2pdf", "pes2o", "stack-edu", "finemath-3plus", "arxiv", "wikipedia"]
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
        source_configs = {f"dclm:{k}": v for k, v in source_configs.items()}

    weights = {}
    for domain in priors[0].keys():
        if domain not in source_configs:
            # two cases
            if ":" in domain and domain.split(":")[0] in source_configs:
                # 1) prior (i.e., swarm config) requests a leaf but mixes from the wandb (i.e. launched outside of the swarm, like natural distr) are specified at the source level
                source_name = domain.split(":")[0]
                weights[domain] = source_configs[source_name].get("target_ratio", 0.0) * priors[0][domain]
            elif ":" in domain and domain.split(":")[-1] in source_configs:
                domain_name = domain.split(":")[-1]
                weights[domain] = source_configs[domain_name].get("target_ratio", 0.0)
            elif ":" not in domain:
                # 2) prior requests a source (i.e. when we condition on a topic-level p* mix) but wandb mixes are specified at the leaf level
                cfg = {k: v.get("target_ratio", 0.0) for k, v in source_configs.items() if f"{domain}:" in k}
                weights[domain] = sum(cfg.values()) if cfg else 0.0
            else:
                # 3) prior's domain has 0 weight in the wandb config
                weights[domain] = 0.0
        else:
            weights[domain] = source_configs.get(domain, {}).get("target_ratio", 0.0)

    return weights


def expand_collapsed_weights(
    opt_weights: dict[str, float],
    original_prior: dict[str, float],
    collapsed_prior: dict[str, float],
) -> dict[str, float]:
    topics_to_expand = list(set(list(original_prior.keys())).difference(set(list(collapsed_prior.keys()))))
    collapsed_sources = list(set(list(collapsed_prior.keys())).difference(set(list(original_prior.keys()))))

    for source in collapsed_sources:
        topics_per_source = sorted([t for t in topics_to_expand if source in t])
        topic_weights = {
            t: original_prior[t] / collapsed_prior[source] * opt_weights[source] for t in topics_per_source
        }
        del opt_weights[source]  # remove the source key
        opt_weights.update(topic_weights)  # add the topic keys with their expanded weights

    return opt_weights


def add_back_in_fixed_source_weights(
    opt_weights: dict[str, float], original_prior: dict[str, float], fixed_weight: dict[str, float]
) -> dict[str, float]:
    domains_to_add_back_in = list(set(list(original_prior.keys())).difference(set(list(opt_weights.keys()))))

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
            topics_per_source = {t: w for t, w in original_prior.items() if t.startswith(source + ":")}
            total = sum(list(topics_per_source.values()))
            topics_per_source = {t: w / total * weight for t, w in topics_per_source.items()}
            final_weights.update(topics_per_source)

    # normalize again just to fix any numerical issues
    total = sum(final_weights.values())
    final_weights = {k: v / total for k, v in final_weights.items()}

    return final_weights


def save_fit_config(fit_config: dict, output_dir: str, custom_name: str | None = None) -> str:
    # Serialize dict in a stable way
    config_str = json.dumps(fit_config, sort_keys=True)

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
        json.dump(fit_config, f, indent=2)

    print(f"[INFO] Saved config to {config_path}")
    return folder_path


def filter_constrained_swarm(final_cookbook_path: Path, run_ratios: list, run_metrics: list) -> tuple[list, list]:
    assert final_cookbook_path is not None, (
        "final_cookbook_path must be set to determine how to construct swarm to be unconstrained."
    )

    with open(final_cookbook_path) as f:
        data = yaml.safe_load(f)

    final_config = ExperimentConfig(**data)
    desired_tokens = final_config.training.get_max_tokens()

    token_universe = get_token_counts_and_ratios(final_config.data.sources, final_config.data.dtype, True)
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
    manual_prior: dict[str, float] | None = None,
    fixed_source_weights: dict[str, float] | None = None,
) -> tuple[tuple[dict[str, float], int, dict[str, int]], tuple[dict[str, float], int, dict[str, int]]]:
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
                        [priors[0][f"{source_config.name}:{topic.name}"] for topic in source_config.topics]
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
            if all(
                [
                    getattr(topic, "weight", None) is not None or getattr(topic, "target_ratio", None) is not None
                    for topic in source_config.topics
                ]
            ):
                # update prior with hardcoded topic weights or target_ratios
                source_weight = sum([priors[0][f"{source_config.name}:{topic.name}"] for topic in source_config.topics])
                for topic in source_config.topics:
                    value = getattr(topic, "weight", None)
                    if value is None:
                        value = getattr(topic, "target_ratio", None)
                    assert value is not None
                    priors[0][f"{source_config.name}:{topic.name}"] = value * source_weight

    original_prior = deepcopy(priors)

    for source_config in source_configs:
        if source_config.topics:
            if all(
                [
                    getattr(topic, "weight", None) is not None or getattr(topic, "target_ratio", None) is not None
                    for topic in source_config.topics
                ]
            ):
                source_weight = sum([priors[0][f"{source_config.name}:{topic.name}"] for topic in source_config.topics])
                for topic in source_config.topics:
                    del priors[0][f"{source_config.name}:{topic.name}"]
                priors[0][source_config.name] = source_weight

    return priors, original_prior


def aggregate_mmlu(metrics: pd.DataFrame, metrics_to_index: list):
    logger.info("Aggregating MMLU metrics...")

    def add_weighted_dot_column(df: pd.DataFrame, weights: dict, output_col: str, metrics_to_index: list):
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
    metrics_to_index = add_weighted_dot_column(metrics, other_weights, "mmlu_other", metrics_to_index)
    metrics_to_index = add_weighted_dot_column(
        metrics, social_sciences_weights, "mmlu_social_sciences", metrics_to_index
    )
    metrics_to_index = add_weighted_dot_column(metrics, humanities_weights, "mmlu_humanities", metrics_to_index)

    return metrics, metrics_to_index


def swarm_config_from_path(config: str) -> ExperimentConfig:
    """
    Load configuration from a config file path.

    Args:
        config: Path to the YAML config file
    Returns:
        ExperimentConfig loaded from the file
    """
    with open(config) as f:
        data = yaml.safe_load(f)
    return ExperimentConfig(**data)
