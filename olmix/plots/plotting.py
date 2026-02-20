"""Plotting functions for olmix experiments.

This module contains visualization functions for:
- Simulation weight distributions
- Regression correlation plots
- Interaction matrix heatmaps
- Optimal weight comparisons
"""

import json
import logging
import os
import re
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

if TYPE_CHECKING:
    from olmix.fit.utils import Regressor

logger = logging.getLogger(__name__)

BASE_OUTPUT_DIR = "output/"

# Define red-white-blue colormap with a white "band"
_colors = ["red", "white", "blue"]
_cmap = LinearSegmentedColormap.from_list("red_white_blue", _colors)

# Boundaries: values between -0.1 and +0.1 stay white
_bounds = [-1.0, -0.9, 0.9, 10]
_norm = BoundaryNorm(_bounds, _cmap.N, extend="both")


def mk_output_prefix(
    output_dir: str,
    metric: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
) -> str:
    """Generate a standardized output file prefix for plots and results."""

    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\\-]", "_", s)

    train_split_str = str(train_split)
    return (
        os.path.join(output_dir, sanitize(metric))
        + (f"_{regression_type}_reg" if regression_type != "lightgbm" else "")
        + (f"_trainsplit_{train_split_str}" if train_split != 1.0 else "")
        + (f"_ntest_{n_test}" if n_test != 0 else "")
        + (f"_seed_{split_seed}" if split_seed != 0 else "")
    )


def plot_correlation(
    Y_test: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    index: int,
    predictors: list["Regressor"],
    train_split: float,
    n_test: int,
    split_seed: int,
    metric_name: str,
    regression_type: str,
    output_dir: str = BASE_OUTPUT_DIR,
    average_bpb: bool = False,
    test_ratios_path: tuple[str, ...] = (),
):
    """Create a regression plot showing predicted vs actual values."""
    plt.close()

    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.labelsize": 16,
        }
    )

    if average_bpb:
        num_tasks = len(predictors)
        y_pred_train = np.mean([predictors[i].predict(X_train) for i in range(num_tasks)], axis=0)
        y_true_train = Y_train.mean(axis=1)
        metric_name = "average_bpb"
    else:
        y_pred_train = predictors[index].predict(X_train)
        y_true_train = Y_train[:, index]

    corr_results = {}

    if n_test == 0 and len(test_ratios_path) == 0:
        # Only plot train if no test set is defined
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
        if average_bpb:
            y_pred_test = np.mean([predictors[i].predict(X_test) for i in range(num_tasks)], axis=0)
            y_true_test = Y_test.mean(axis=1)
            metric_name = "average_bpb"
        else:
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

    output_prefix = mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed)

    # Save figure
    plt.savefig(f"{output_prefix}_fit.png")
    plt.close()

    with open(f"{output_prefix}_correlations.json", "w") as f:
        f.write(json.dumps(corr_results))


def plot_interaction_matrix(
    output_dir: str,
    predictors: list["Regressor"],
    regression_type: str,
    domain_names: list[str],
    metric_names: list[str],
    ratios: pd.DataFrame,
):
    """Create a heatmap showing domain-metric interaction coefficients."""
    metric_names = [metric.split("/")[-1].split(" ")[0] for metric in metric_names]

    interaction_matrix = np.zeros((len(metric_names), len(domain_names)))

    for i, predictor in enumerate(predictors):
        if regression_type == "lightgbm":
            interaction_matrix[i] = predictor.model.feature_importances_
        elif regression_type == "log_linear":
            interaction_matrix[i] = predictor.model[1:]
        else:
            logger.info(f"Unknown regression type: {regression_type}, skipping...")
            return

    sorted_metric_indices = np.argsort(metric_names)
    metric_names = [metric_names[i] for i in sorted_metric_indices]
    interaction_matrix = interaction_matrix[sorted_metric_indices, :]

    plt.figure(figsize=(20, 16))
    plt.imshow(interaction_matrix, cmap="rainbow", aspect="auto")
    cmap = plt.cm.coolwarm
    vlim = np.abs(interaction_matrix).max()
    # Show color mesh
    plt.imshow(interaction_matrix, cmap=cmap, vmin=-vlim, vmax=+vlim, aspect="auto")

    bar_label = "Influence"
    bar_label += " (lower is better)"

    plt.colorbar(label=bar_label)
    plt.xticks(ticks=np.arange(len(domain_names)), labels=domain_names, rotation=90)
    plt.yticks(ticks=np.arange(len(metric_names)), labels=metric_names)
    plt.title(f"Interaction matrix for {regression_type}")

    # Annotate each cell with its value
    for i in range(len(metric_names)):
        for j in range(len(domain_names)):
            val = interaction_matrix[i, j]
            text_str = f"β={val:.2f}"
            plt.text(
                j,
                i,
                text_str,
                ha="center",
                va="center",
                color="black" if abs(val) < 0.5 * np.max(np.abs(interaction_matrix)) else "white",
                fontsize=10,
            )

    plt.tight_layout()

    plt.savefig(
        f"{output_dir}/interaction_matrix.png",
        bbox_inches="tight",
    )
    plt.close()
    np.save(f"{output_dir}/interaction_matrix.npy", interaction_matrix)


def plot_and_log_weights(
    prior: dict[str, float],
    original_prior: dict[str, float],
    prediction: np.ndarray,
    metric_name: str,
    regression_type: str,
    train_split: float,
    n_test: int,
    split_seed: int,
    domain_cols: list[str],
    output_dir: str = BASE_OUTPUT_DIR,
    expand_collapsed_weights_fn=None,
):
    """Create a bar chart comparing corpus weights vs optimal weights."""
    logger.info(f":::::::::{metric_name}:::::::::")
    logger.info("Predicted optimal weights:")

    if set(list(original_prior.keys())) != set(list(prior.keys())):
        # expand weights
        opt_weight_dict = {k: prediction[i] for i, (k, v) in enumerate(prior.items())}
        if expand_collapsed_weights_fn is not None:
            opt_weight_dict = expand_collapsed_weights_fn(opt_weight_dict, original_prior, prior)
        out = [{"domain": domain, "weight": weight} for domain, weight in opt_weight_dict.items()]
        columns = list(prior.keys())
    else:
        columns = domain_cols
        out = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)]

    if len(out) != len(domain_cols):
        logger.info("RAW WEIGHTS:")
        raw_weights = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)]
        logger.info(raw_weights)

    with open(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed)}_optimal.json",
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
    df["type"] = (["Corpus", "Optimal"]) * len(columns)

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
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed)}_optimal.png",
        bbox_inches="tight",
        pad_inches=0.1,
    )
