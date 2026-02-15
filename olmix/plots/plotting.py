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


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment to convert p-values to q-values."""
    flat = pvals.ravel().astype(float)
    n = flat.size
    order = np.argsort(flat)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    q = flat * n / ranks
    # enforce monotonicity
    q_sorted = np.minimum.accumulate(q[order][::-1])[::-1]
    out = np.empty_like(q_sorted)
    out[order] = q_sorted
    return out.reshape(pvals.shape)


def mk_output_prefix(
    output_dir: str,
    metric: str,
    regression_type: str,
    train_split: tuple[float, ...],
    n_test: int,
    split_seed: int,
) -> str:
    """Generate a standardized output file prefix for plots and results."""

    def sanitize(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_\\-]", "_", s)

    train_split_str = [str(t) for t in train_split]
    return (
        os.path.join(output_dir, sanitize(metric))
        + (f"_{regression_type}_reg" if regression_type != "lightgbm" else "")
        + (f"_trainsplit_{'_'.join(train_split_str)}" if train_split[0] != 1.0 else "")
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
    train_split: tuple[float, ...],
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

    if train_split[0] == 1 and n_test == 0 and len(test_ratios_path) == 0:
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

    # Save figure
    plt.savefig(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed)}_fit.png"
    )
    plt.close()

    with open(
        f"{mk_output_prefix(output_dir, metric_name, regression_type, train_split, n_test, split_seed)}_correlations.json",
        "w",
    ) as f:
        f.write(json.dumps(corr_results))


def plot_interaction_matrix(
    output_dir: str,
    predictors: list["Regressor"],
    regression_type: str,
    domain_names: list[str],
    metric_names: list[str],
    ratios: pd.DataFrame,
    interactions: list[str] | None = None,
):
    """Create a heatmap showing domain-metric interaction coefficients."""
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
            # std = ratios[ratios.columns[3:]].std(ddof=0).values  # std for selected columns only
            interaction_matrix[i] = predictor.model.params  # * std
        elif regression_type == "quadratic":
            interaction_matrix[i] = predictor.model.params

    domain_reordering = None
    if "a3d4f82c" in output_dir:
        # hardcode the nice block structure for the superswarm
        new_order = [
            "code_fim",
            "dolminomath",
            "megamatt",
            "openmathreasoning-fullthoughts-rewrite",
            "swallowcode",
            "swallowmath",
            "tinymath-mind",
            "tinymath-pot",
            "flan",
            "instruction-new-format",
            "nemotron-synth-qa",
            "rcqa",
            "reddit-high",
            "sponge",
            "hqweb-pdf",
        ]
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
    plt.imshow(interaction_matrix, cmap=cmap, vmin=-vlim, vmax=+vlim, aspect="auto")

    bar_label = "Influence"
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


def plot_interaction_matrix_signed_evidence(
    output_dir: str,
    predictors: list,
    regression_type: str,
    domain_names: list[str],
    metric_names: list[str],
    ratios: pd.DataFrame,
    use_fdr: bool = False,
    p_cap: float = 10.0,  # kept for API compatibility; unused in p-coloring
    sig_threshold: float = 0.05,
    gamma: float = 1.5,  # >1 makes the color mapping less drastic
):
    """Create a signed evidence heatmap with p-value coloring."""
    # Normalize metric labels
    metric_names = [m.split("/")[-1].split(" ")[0] for m in metric_names]

    # Optional hardcoded domain block structure + build a column permutation
    domain_reordering = None
    if "a3d4f82c" in output_dir:
        new_order = [
            "code_fim",
            "dolminomath",
            "megamatt",
            "openmathreasoning-fullthoughts-rewrite",
            "swallowcode",
            "swallowmath",
            "tinymath-mind",
            "tinymath-pot",
            "flan",
            "instruction-new-format",
            "nemotron-synth-qa",
            "rcqa",
            "reddit-high",
            "sponge",
            "hqweb-pdf",
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
        bh_adjust(P) if use_fdr else P

        # ---- COLOR BY p, gently (bounded), while TEXT shows raw p ----
        # Map p in [0,1] to a gentle strength via (1 - p)^(1/gamma), then apply sign(β)
        P_safe = np.clip(P, 0.0, 1.0)
        p_strength = (1.0 - P_safe) ** (1.0 / max(gamma, 1e-6))
        signed_score = np.sign(B) * p_strength  # in [-1, 1]

        plt.figure(figsize=(20, 16))
        # im = plt.imshow(
        #    signed_score,
        #    cmap="coolwarm",
        #    norm=TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v),
        #    aspect="auto",
        # )

        im = plt.imshow(signed_score, cmap=_cmap, norm=_norm, aspect="auto")

        # Dim non-significant cells (gray overlay)
        # mask = (Q if use_fdr else P_safe) > sig_threshold
        # overlay = np.zeros((*signed_score.shape, 4))
        # overlay[mask] = [0.7, 0.7, 0.7, 0.6]
        # plt.imshow(overlay, aspect="auto")

        # Annotate with β and RAW p (as requested)
        for i in range(len(metric_names)):
            for j in range(len(domain_names)):
                text_str = f"β={B[i, j]:.2f}\np={P[i, j]:.2g}"
                text_color = "black"  # if abs(signed_score[i, j]) < 0.5 * v else "white"
                plt.text(j, i, text_str, ha="center", va="center", fontsize=8, color=text_color)

        cbar = plt.colorbar(im)
        cbar.set_label(r"sign(β) × (1 − p)$^{1/\gamma}$")
        better = "lower is better"
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


def plot_and_log_weights(
    prior: dict[str, float],
    original_prior: dict[str, float],
    prediction: np.ndarray,
    metric_name: str,
    regression_type: str,
    train_split: tuple[float, ...],
    n_test: int,
    split_seed: int,
    df_config: pd.DataFrame,
    output_dir: str = BASE_OUTPUT_DIR,
    fixed_weight: dict[str, float] | None = None,
    expand_collapsed_weights_fn=None,
    add_back_in_fixed_source_weights_fn=None,
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
        columns = df_config.columns[3:].to_list()
        out = [{"domain": columns[idx], "weight": weight} for idx, weight in enumerate(prediction)]

    if fixed_weight is not None and add_back_in_fixed_source_weights_fn is not None:
        opt_weight_dict = add_back_in_fixed_source_weights_fn(opt_weight_dict, original_prior, fixed_weight)
        out = [{"domain": domain, "weight": weight} for domain, weight in opt_weight_dict.items()]

    if len(out) != len(df_config.columns[3:]):
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
