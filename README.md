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
  seed: 0
  n_test: 0
  train_split: [1.0]
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

The **priors** section defines the natural token distribution across your domains via `token_counts`. Relative sizes and total tokens are computed automatically. Use `olmix priors compute` to scan S3 sources and generate the token counts for a config.

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
| `aggregate_task_families` | Fit one model per task family (math, code, QA) instead of per individual task. Much faster with many metrics. |
| `train_split` | Fraction of runs used for training. Default `[1.0]` uses all runs for both training and evaluation. |
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

The key output is `opt_avg_all_metrics_*_optimal.json` — the single set of weights that optimizes the average across all metrics.

---

## Part 2: Launching swarms and fitting from W&B

Once you're comfortable with the fitting workflow above, you can use olmix end-to-end: generate candidate mixtures, launch proxy training runs on Beaker, and fit directly from the W&B results.

The workflow uses two separate configs:

- **`GenerationConfig`** — controls how mixes are sampled (data sources, priors, swarm parameters, token budget). See [`configs/generations/example.yaml`](configs/generations/example.yaml).
- **`LaunchConfig`** — controls how training runs are launched (infra, training hyperparams, eval). See configs in [`configs/experiments/`](configs/experiments/).

### Step 0: Compute priors (token counts)

Before generating mixes, compute the token counts for your data sources. This scans S3 paths and outputs the `priors` block for your generation config:

```bash
olmix priors compute --config configs/generations/example.yaml
```

Copy the output into your generation config's `priors:` section. Use `--output priors.yaml` to write to a file instead. Results are cached in `cache/` for subsequent runs; use `--no-cache` to force a fresh scan.

### Step 1: Generate candidate mixtures

Use `olmix generate` to sample mixture variants from a generation config. The `--base` flag provides a launch config template, and each variant is written as a self-contained launch config YAML file — ready to submit directly.

```bash
olmix generate \
  --config configs/generations/example.yaml \
  --base configs/experiments/data_proportions/mix_baseline.yaml \
  --output output/my_variants/
```

This produces one YAML file per variant in the output directory:

```
output/my_variants/
  example-swarm-a1b2c3d4-0000.yaml
  example-swarm-a1b2c3d4-0001.yaml
  example-swarm-a1b2c3d4-0002.yaml
  example-swarm-a1b2c3d4-0003.yaml
```

Each variant file is a complete launch config with infra, training, data, eval, and the sampled mix:

```yaml
name: example-swarm-a1b2c3d4-0000
group_id: a1b2c3d4
infra:
  budget: ai2/oe-base
  cluster: ai2/jupiter
  # ...
training:
  proxy_model_id: olmo3_14m
  # ...
data:
  sources:
    - name: dclm
      topics:
        - name: science_math_and_technology
          paths: [s3://...]
    - name: wikipedia
      paths: [s3://...]
eval:
  type: inloop
  tasks: { ... }
mix:
  dclm:science_math_and_technology:
    weight: 0.55
    repetition_factor: 1.0
  wikipedia:
    weight: 0.10
    repetition_factor: 1.5
```

Inspect and edit these files before launching — this is the point where you have full control over what gets trained.

### Step 2: Preview training commands

Takes the generated variants directory and renders the full OLMo training command for each variant. Prints to stdout without launching anything.

```bash
olmix launch preview --variants output/my_variants/
```

### Step 3: Launch a swarm

Submits one Beaker job per variant. Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Launch metadata is saved in the variants directory.

```bash
olmix launch run --variants output/my_variants/
```

Use `--dry-run` to generate the metadata JSON without launching any jobs.

### Step 4: Export to CSV and fit

Once the swarm runs complete, export the ratios and metrics to CSV files (e.g. from W&B), then fit using the YAML config workflow described in [Part 1](#part-1-fitting-from-csv-data):

```bash
olmix fit --config configs/fits/my_config.yaml --output-dir output/my_fit
```

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
