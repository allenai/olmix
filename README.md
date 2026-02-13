# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)

> [!WARNING]
> This project is under active development. We are migrating from our internal infrastructure to open source — expect rough edges, missing docs, and breaking changes.

Toolkit for optimizing pretraining data mixtures. Learns from small-scale proxy experiments ("swarms") to predict how data mixing ratios affect downstream performance, then proposes optimized mixtures for full-scale training.

## Installation

```bash
git clone https://github.com/allenai/olmix.git
cd olmix
uv pip install -e ".[dev]"
```

## Part 1: Fitting from CSV data

The fastest way to use olmix is to bring your own swarm results as CSV files. This section explains how to run regression fitting and mixture optimization.

### Input format

Prepare a directory with two CSV files. Each row is one training run from your swarm, and the two files are joined on `run_id`.

**`ratios.csv`** — the mixing weights used in each run. Domain columns must sum to ~1.0 per row:

| run_id  | dclm  | wikipedia | arxiv |
|---------|-------|-----------|-------|
| run_001 | 0.45  | 0.30      | 0.25  |
| run_002 | 0.60  | 0.20      | 0.20  |
| run_003 | 0.33  | 0.33      | 0.34  |

**`metrics.csv`** — the evaluation metrics measured for each run (lower is better for BPB metrics):

| run_id  | arc_challenge_bpb | hellaswag_bpb | mmlu_stem_bpb |
|---------|-------------------|---------------|---------------|
| run_001 | 1.23              | 0.87          | 1.45          |
| run_002 | 1.15              | 0.91          | 1.38          |
| run_003 | 1.20              | 0.89          | 1.42          |

The domain column names in `ratios.csv` and the metric column names in `metrics.csv` can be anything — olmix derives them from the CSV headers. An optional `name` column in either file provides human-readable run labels.

### Running a fit

```bash
olmix fit \
  --from-csv path/to/swarm_data/ \
  --config configs/experiments/data_proportions/mix_baseline.yaml \
  --regression-type log_linear \
  --opt-avg-metric
```

`--from-csv` points to the directory containing `ratios.csv` and `metrics.csv`. `--config` is required — it tells olmix how to compute **priors** (the natural token distribution across your domains), which serve as the baseline the optimizer improves upon. See [How the config affects fitting](#how-the-config-affects-fitting) below.

#### Key flags

| Flag | What it does |
|------|-------------|
| `--regression-type` | Model type: `log_linear` (default, parametric scaling law), `lightgbm` (gradient-boosted trees), `linear` (OLS), `quadratic` (OLS with interaction terms) |
| `--opt-avg-metric` | Optimize the average across all metrics jointly. Without this, each metric is optimized independently and results are averaged post-hoc. |
| `--fit-only` | Only fit the regression models, skip the mixture proposal step. Useful for inspecting model quality before committing to a proposal. |
| `--proposer-type` | How to search for optimal weights: `exact` (default, convex optimization for log-linear), `simulation` (Dirichlet Monte Carlo), `search` (grid over observed points) |
| `--constrain-objective` | Constrain proposed weights so no source exceeds its available tokens (requires `target_tokens` or `target_chinchilla_multiple` in config) |
| `--kl-reg FLOAT` | KL divergence regularization strength (exact proposer only). Penalizes the proposed mix for diverging from the prior. Higher values stay closer to the natural distribution. |
| `--train-split FLOAT` | Fraction of runs used for training (rest held out). Default `1.0` uses all runs for both training and evaluation. |
| `--keep-sources A B` | Only use runs where sources A and B have nonzero weight (and all others are zero). Useful for fitting a subset of domains. |
| `--drop-metrics M1 M2` | Exclude specific metrics from fitting. |

### How the config affects fitting

The `--config` YAML file defines your data sources and their S3/local paths. During fitting, olmix uses this to compute **priors** — the natural token distribution across domains:

```yaml
sources:
  - name: dclm
    topics:
      - name: science_math_and_technology
        paths: ["s3://bucket/dclm/science/**/*.npy"]
      - name: software_development
        paths: ["s3://bucket/dclm/code/**/*.npy"]
  - name: wikipedia
    paths: ["s3://bucket/wikipedia/*.npy"]
  - name: arxiv
    paths: ["s3://bucket/arxiv/*.npy"]
```

Olmix counts tokens at each path to produce a prior like `{dclm:science_math_and_technology: 0.35, dclm:software_development: 0.25, wikipedia: 0.22, arxiv: 0.18}`. This prior serves two roles:

1. **Regression baseline.** The fitted model learns how deviations from this distribution affect metrics.
2. **Proposal anchor.** The optimizer uses the prior as a starting point (Dirichlet center for simulation, KL reference for exact).

Config fields that affect the fit:

| Field | Effect |
|-------|--------|
| `sources` | Defines the domain names and paths used to count tokens and compute priors |
| `sources[].topics` | Hierarchical sources (e.g. `dclm` with topics) create domain names like `dclm:science_math_and_technology` |
| `dtype` | Token counting uses `dtype` byte width (default: `uint32` = 4 bytes per token) |
| `fixed_source_weights` | Pins specific sources to fixed weights — they are excluded from optimization |
| `manual_prior` | Overrides the calculated prior for specific sources |
| `target_tokens` / `target_chinchilla_multiple` | Token budget for `--constrain-objective` — ensures proposed weights don't require more tokens than available per source |
| `repetition_factor` | Maximum times a source's tokens can be repeated (default: 5.0). Used with `--constrain-objective`. |

### Output

All results are written to a subdirectory under `output/`. The directory name is derived from a hash of the eval config, so different configurations produce separate output folders.

| File | Description |
|------|-------------|
| `config.json` | Full configuration used for this fit (for reproducibility) |
| `interaction_matrix.png` | Heatmap of regression coefficients: rows are domains, columns are metrics. Shows which domains help or hurt each metric. |
| `interaction_matrix_signed_evidence.png` | Same matrix colored by statistical significance. Green = significant positive effect, red = significant negative effect. |
| `{metric}_*_fit.png` | Per-metric regression plot: predicted vs. actual values. Tight clustering along the diagonal means the model fits well. |
| `{metric}_*_optimal.json` | Proposed optimal weights for this metric (list of `{"domain": ..., "weight": ...}`). |
| `{metric}_*_optimal.png` | Bar chart comparing the natural prior (corpus distribution) to the proposed optimal weights. |
| `predicted_performance.json` | Predicted average metric value at the proposed optimal weights. |
| `path_to_regression_model.txt` | Path to the cached regression model (pickle). Reused on subsequent fits with the same regression config. |

When `--opt-avg-metric` is set, the key output is `opt_avg_all_metrics_*_optimal.json` — the single set of weights that optimizes the average across all metrics.

---

## Part 2: Launching swarms and fitting from W&B

Once you're comfortable with the fitting workflow above, you can use olmix end-to-end: generate candidate mixtures, launch proxy training runs on Beaker, and fit directly from the W&B results.

### Experiment configs

Start from one of the configs in [`configs/experiments/`](configs/experiments/):

| Suite | What it tests |
|-------|--------------|
| [`data_proportions/`](configs/experiments/data_proportions/) | Varying topic weights across sources |
| [`quality_thresholds/`](configs/experiments/quality_thresholds/) | Including/excluding quality vigintiles |
| [`quality_upsampling/`](configs/experiments/quality_upsampling/) | Weighting quality buckets within topics |
| [`training_duration/`](configs/experiments/training_duration/) | Effect of training length on mixture quality |

### Step 1: Sample candidate mixtures

The input config defines sources hierarchically (source → topic → quality bucket) with S3 paths. This step scans those paths to count available tokens, uses the token counts as Dirichlet priors, and samples `variants` mixture configurations. Each mix flattens the hierarchy into fully-qualified leaf keys (e.g. `"dclm:software_development"`) with a `[weight, repetition_factor]` pair. Repetition factors are computed from how many tokens are needed vs. available. Output is written to `output/mixes/`.

```bash
olmix mix generate --config configs/experiments/data_proportions/mix_baseline.yaml
```

### Step 2: Preview training commands

Takes the sampled mixtures from step 1 and renders the full OLMo training command for each variant — model config, data paths, mixing weights, eval settings. Prints to stdout without launching anything.

```bash
olmix launch preview --config configs/experiments/data_proportions/mix_baseline.yaml
```

### Step 3: Launch a swarm

Submits one Beaker job per variant (steps 1-2 happen automatically if no pre-generated mix file is provided). Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Launch metadata (Beaker experiment IDs, W&B group link, git commit) is saved alongside the mix JSON in `output/mixes/`.

```bash
olmix launch run --config configs/experiments/data_proportions/mix_baseline.yaml
```

### Step 4: Fit from W&B results

Point `--from-wandb` at the launch output directory. The group ID and experiment config are auto-resolved from the metadata JSON written by step 3 — no `--config` or `--experiment-groups` flags needed:

```bash
olmix fit \
  --from-wandb output/mixes/data_proportions/mix_baseline/<LAUNCH_OUTPUT_DIR>/ \
  --regression-type log_linear \
  --opt-avg-metric
```

This pulls eval metrics from the completed W&B runs, pairs them with the mixing weights, and produces the same output described in [Output](#output) above.

## Development

```bash
make run-checks   # format + lint + typecheck + test
```

## Citation

```bibtex
@article{chen2026olmix,
  title={Olmix: A Framework for Data Mixing Throughout LM Development},
  author={Chen, Mayee F and Murray, Tyler and Heineman, David and Jordan, Matt and Hajishirzi, Hannaneh and Re, Christopher and Soldaini, Luca and Lo, Kyle},
  year={2026},
  month={February}
}
```

## License

[Apache 2.0](LICENSE)
