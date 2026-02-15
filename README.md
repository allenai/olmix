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

The easiest way to use Olmix is to bring your own swarm results as CSV files. Our swarms are available on [Huggingface](https://huggingface.co/datasets/allenai/olmix). This section explains how to run regression fitting and mixture optimization.

### Input format

Prepare two CSV files. Each row is one training run from your swarm, and the two files are joined on the ID column (`run` or `run_id`).

**`ratios.csv`** — the mixing weights used in each run. Domain columns must sum to ~1.0 per row:

| run     | dclm  | wikipedia | arxiv |
|---------|-------|-----------|-------|
| run_001 | 0.45  | 0.30      | 0.25  |
| run_002 | 0.60  | 0.20      | 0.20  |
| run_003 | 0.33  | 0.33      | 0.34  |

**`metrics.csv`** — the evaluation metrics measured for each run (lower is better for BPB metrics):

| run     | arc_challenge_bpb | hellaswag_bpb | mmlu_stem_bpb |
|---------|-------------------|---------------|---------------|
| run_001 | 1.23              | 0.87          | 1.45          |
| run_002 | 1.15              | 0.91          | 1.38          |
| run_003 | 1.20              | 0.89          | 1.42          |

The domain column names in `ratios.csv` and the metric column names in `metrics.csv` can be anything — olmix derives them from the CSV headers. Both `run` and `run_id` are accepted as the ID column. An optional `name` column in either file provides human-readable run labels.

### Fit config

`olmix fit` is configured entirely via a YAML file. See [`configs/fits/dclm_baseline.yaml`](configs/fits/dclm_baseline.yaml) for a full example. The config has these sections:

```yaml
swarm:
  ratios: path/to/ratios.csv        # Required — CSV with domain mixture ratios per run
  metrics: path/to/metrics.csv      # Required — CSV with eval metrics per run

priors:                              # Required — token distribution across domains
  relative_sizes:
    domain_a: 0.6
    domain_b: 0.4
  total_tokens: 1000000000
  token_counts:
    domain_a: 600000000
    domain_b: 400000000

eval:                                 # Required — evaluation task definitions
  type: offline                       # offline | inloop
  tasks:
    math:
      - "minerva_math_algebra::olmes"
    code:
      - "codex_humaneval:3shot::none"
    qa:
      - "arc_challenge:rc::olmes"

regression:
  type: log_linear                   # log_linear | lightgbm | search | gp | autoscale | bimix
  alpha: 1.0
  seed: 0
  n_test: 0
  train_split: [1.0]
  simulation_samples: 100000
  opt_avg_metric: true
  aggregate_task_families: false

proposer:
  type: exact                        # exact | simulation | search
  temperature: null
  kl_reg: 0.1
  use_natural_kl: false
  fit_only: false
  make_worst_mix: false

constraints:
  enabled: false
  target_tokens: null                # Total token budget for the final training run
  repetition_factor: 5.0

filtering:
  keep_sources: []
  support_domains: []
  drop_metrics: []
  fixed_weight: {}
  obj_weights: {}
```

The **priors** section defines the natural token distribution across your domains inline — no S3 access or external JSON files needed.

The **eval** section defines which evaluation tasks to use, grouped by family. Two types are supported:

- **`offline`** — for cookbook-eval metrics (used by `olmix fit` with CSV data). Tasks are metric names matching CSV column headers.
- **`inloop`** — for WandB in-loop metrics (used by `olmix launch` and `olmix fit`). Tasks map olmo-core task IDs to WandB metric names: `{task_id: "eval/downstream/task_id (BPB v2)"}`.

Task families are defined by the nesting structure (e.g., `math`, `code`, `qa`) and are used by `aggregate_task_families`.

### Running a fit

```bash
olmix fit --config configs/fits/dclm_baseline.yaml --output-dir output/my_fit
```

That's it. All settings come from the YAML config. The two required CLI flags are:

| Flag | Description |
|------|-------------|
| `--config` | Path to the YAML fit configuration file |
| `--output-dir` | Directory for saving fit outputs |

### Config reference

#### `eval` section

| Field | What it does |
|-------|-------------|
| `type` | Eval type: `offline` (cookbook-eval metrics for CSV-based fitting) or `inloop` (WandB in-loop metrics for launch + fitting) |
| `tasks` | Tasks grouped by family. For `offline`: `{family: [metric_name, ...]}`. For `inloop`: `{family: {task_id: wandb_metric_name}}`. |

#### `regression` section

| Field | What it does |
|-------|-------------|
| `type` | Model type: `log_linear` (default, parametric scaling law), `lightgbm` (gradient-boosted trees), `gp` (Gaussian process), `autoscale` (power-law autoscaling), `bimix` (BiMix-style power law) |
| `opt_avg_metric` | Optimize the average across all metrics jointly. Without this, each metric is optimized independently. |
| `aggregate_task_families` | Fit one model per task family (math, code, QA) instead of per individual task. Much faster with many metrics. |
| `train_split` | Fraction of runs used for training. Default `[1.0]` uses all runs for both training and evaluation. |
| `simulation_samples` | Number of Monte Carlo samples for the simulation proposer. |
| `alpha` | Alpha to apply to simulated distributions. |
| `n_test` | Number of held-out test samples for evaluating the regression model. |
| `seed` | Random state for train-test split. |

#### `proposer` section

| Field | What it does |
|-------|-------------|
| `type` | How to search for optimal weights: `exact` (convex optimization for log-linear), `simulation` (Dirichlet Monte Carlo), `search` (grid over observed points) |
| `kl_reg` | KL divergence regularization strength (exact proposer only). Penalizes the proposed mix for diverging from the prior. |
| `use_natural_kl` | Use the natural (token-count-based) distribution as the KL reference, even when a manual prior is set. |
| `temperature` | Temperature for adjusting the Dirichlet prior in simulation. Closer to 0 = more uniform. |
| `fit_only` | Only fit the regression models, skip the mixture proposal step. Useful for inspecting model quality. |
| `make_worst_mix` | Invert the objective function and produce a bad mix (for counterfactual analysis). |

#### `constraints` section

| Field | What it does |
|-------|-------------|
| `enabled` | Enable token budget constraints on the proposed mixture. |
| `target_tokens` | Total token budget for the final training run. Required when `enabled: true`. |
| `repetition_factor` | Maximum times a source's tokens can be repeated (default: 5.0). |

#### `filtering` section

| Field | What it does |
|-------|-------------|
| `keep_sources` | Only use runs where these sources have nonzero weight (and all others are zero). |
| `support_domains` | Only use runs where these domains' ratios sum to 1. |
| `drop_metrics` | Exclude specific metrics from fitting. |
| `fixed_weight` | Pin specific domains to fixed weights — they are excluded from optimization. Native dict syntax (not JSON string). |
| `obj_weights` | Non-uniform weights for averaging BPB across tasks. Default is uniform. |

### Output

All results are written to a hashed subdirectory under the `--output-dir` you specify. The subdirectory name is derived from a hash of the config, so different configurations produce separate output folders.

| File | Description |
|------|-------------|
| `config.json` | Full configuration used for this fit (for reproducibility) |
| `interaction_matrix.png` | Heatmap of regression coefficients: rows are domains, columns are metrics. Shows which domains help or hurt each metric. |
| `interaction_matrix_signed_evidence.png` | Same matrix colored by statistical significance. Green = significant positive effect, red = significant negative effect. |
| `interaction_matrix.npy` | Raw interaction matrix as a NumPy array (for downstream analysis). |
| `{metric}_*_fit.png` | Per-metric regression plot: predicted vs. actual values. Tight clustering along the diagonal means the model fits well. |
| `{metric}_*_correlations.json` | Correlation metrics (e.g. R²) for each regression fit. |
| `path_to_regression_model.txt` | Path to the cached regression model (pickle). Reused on subsequent fits with the same regression config. |

When `fit_only: false`, the proposer step also produces:

| File | Description |
|------|-------------|
| `{metric}_*_optimal.json` | Proposed optimal weights for this metric (list of `{"domain": ..., "weight": ...}`). |
| `{metric}_*_optimal.png` | Bar chart comparing the natural prior (corpus distribution) to the proposed optimal weights. |
| `predicted_performance.json` | Predicted average metric value at the proposed optimal weights. |

When `opt_avg_metric: true` is set, the key output is `opt_avg_all_metrics_*_optimal.json` — the single set of weights that optimizes the average across all metrics.

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

Submits one Beaker job per variant (steps 1-2 happen automatically if no pre-generated mix file is provided). Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Launch metadata (Beaker experiment IDs, W&B group link, git commit, **priors**) is saved alongside the mix JSON in `output/mixes/`.

```bash
olmix launch run --config configs/experiments/data_proportions/mix_baseline.yaml
```

Use `--dry-run` to generate the metadata JSON (including priors) without launching any jobs. The priors from the metadata can be copied into a fit YAML config's `priors` section.

### Step 4: Export to CSV and fit

Once the swarm runs complete, export the ratios and metrics to CSV files (e.g. from W&B), then fit using the YAML config workflow described in [Part 1](#part-1-fitting-from-csv-data):

```bash
olmix fit --config configs/fits/my_config.yaml --output-dir output/my_fit
```

The priors saved in the launch metadata JSON (from step 3) can be copied directly into the YAML config's `priors` section.

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
