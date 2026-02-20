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

## Part 1: Mixture optimization from CSV data

The easiest way to use Olmix is to bring your own swarm results as CSV files. Our swarms (30M models trained on 3B tokens) are available on [Huggingface](https://huggingface.co/datasets/allenai/olmix). This section explains how to run regression fitting on the swarm to output an optimized data mixture.

### Input format

Prepare two CSV files. Each row is one proxy run from your swarm, and the two files are joined on the ID column (`run` or `run_id`).

**`ratios.csv`** — the mixing weights used in each run. Domain columns must sum to ~1.0 per row:

| run      | name          | index | dclm  | wikipedia | arxiv |
|----------|---------------|-------|-------|-----------|-------|
| hz0dfydj | my-swarm-0000 | 0     | 0.45  | 0.30      | 0.25  |
| pj0hxxl7 | my-swarm-0001 | 1     | 0.60  | 0.20      | 0.20  |
| sqleanmq | my-swarm-0002 | 2     | 0.33  | 0.33      | 0.34  |

**`metrics.csv`** — the evaluation metrics measured for each run (lower is better for BPB metrics):

| run      | name          | index | arc_challenge_bpb | hellaswag_bpb | mmlu_stem_bpb |
|----------|---------------|-------|-------------------|---------------|---------------|
| hz0dfydj | my-swarm-0000 | 0     | 1.23              | 0.87          | 1.45          |
| pj0hxxl7 | my-swarm-0001 | 1     | 1.15              | 0.91          | 1.38          |
| sqleanmq | my-swarm-0002 | 2     | 1.20              | 0.89          | 1.42          |

The domain column names in `ratios.csv` and the metric column names in `metrics.csv` can be anything — Olmix derives them automatically from the CSV headers. The following columns are treated as metadata and skipped during fitting: `run` (or `run_id`) — the required ID column used to join the two files; `name` — an optional human-readable label; `index` — an optional sequential index; and any unnamed row-index columns (e.g., added by pandas on export). Only `run` or `run_id` is required.

### Fit config

`olmix fit` is configured via a YAML file. Run it with:

```bash
olmix fit --config configs/fits/dclm_baseline.yaml --output-dir output/my_fit
```

| Flag | Description |
|------|-------------|
| `--config` | Path to the YAML fit configuration file |
| `--output-dir` | Directory for saving fit outputs |

See [`configs/fits/dclm_baseline.yaml`](configs/fits/dclm_baseline.yaml) for a full example. The config has these sections:

```yaml
swarm:
  ratios: path/to/ratios.csv        # Required — CSV with domain mixture ratios per run
  metrics: path/to/metrics.csv      # Required — CSV with eval metrics per run

priors:
  relative_sizes:
    domain_a: 0.6
    domain_b: 0.4
  token_counts:
    domain_a: 600000000
    domain_b: 400000000

eval:                                # optional — omit to use all metrics without grouping
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
  train_split: 1.0
  aggregate_task_families: false

proposer:
  type: exact                        # exact | simulation | search
  temperature: null
  kl_reg: 0.1
  fit_only: false
  make_worst_mix: false

constraints:
  enabled: false
  target_tokens: null                # Total token budget for the final training run
  repetition_factor: 4.0

filtering:
  drop_metrics: []
  obj_weights: {}
```

### Config reference

Only `swarm` and `priors` are required. All other sections are optional and fall back to the defaults shown above.

#### `priors` section

| Field | What it does |
|-------|-------------|
| `relative_sizes` | Fractional weight of each domain in the natural corpus (should sum to ~1.0). Defines the prior distribution used as the KL regularization target in the proposer. |
| `token_counts` | Absolute token count per domain. Used for repetition constraint. |
| `total_tokens` | (Optional) Total token budget across all domains, equal to the sum across `token_counts`. |

#### `eval` section

| Field | What it does |
|-------|-------------|
| `tasks` | Metrics to include, grouped by task family. Each family maps to a list of metric names matching CSV column headers. Task families are used by `aggregate_task_families`. The entire `eval` section is optional — if omitted, all metrics in the CSV are used. |

#### `regression` section

| Field | What it does |
|-------|-------------|
| `type` | Model type: `log_linear` (default, parametric scaling law), `lightgbm` (gradient-boosted trees), `gp` (Gaussian process), `autoscale` (power-law autoscaling), `bimix` (BiMix-style power law) |
| `aggregate_task_families` | Fit one model per task family (e.g., math, code, QA) instead of per individual task. Requires task family to be defined in `eval.tasks`. |
| `train_split` | Fraction/number of runs used for fitting the regression model. Default `1.0` uses all runs. |
| `n_test` | Number of held-out test samples for evaluating the regression model. If nonzero, only reports fit quality and does not propose a mix. |
| `seed` | Random state for train-test split. |

#### `proposer` section

| Field | What it does |
|-------|-------------|
| `type` | How to search for optimal weights: `exact` (convex optimization for log-linear), `simulation` (Dirichlet Monte Carlo), `search` (over swarm mixes) |
| `kl_reg` | KL divergence regularization strength (exact proposer only). Penalizes the proposed mix for diverging from the prior as specified in `relative_sizes`. |
| `temperature` | Temperature for adjusting the Dirichlet prior in simulation. Closer to 0 = more uniform. |
| `fit_only` | Only fit the regression models, skip the mixture proposal step. |
| `make_worst_mix` | Invert the objective function and produce a bad mix (for counterfactual analysis). |

#### `constraints` section

| Field | What it does |
|-------|-------------|
| `enabled` | Enable repetition constraints in the optimization step. |
| `target_tokens` | Total token budget for the final training run. Required when `enabled: true`. |
| `repetition_factor` | Maximum times a source's tokens can be repeated (default: 4.0). |

#### `filtering` section

| Field | What it does |
|-------|-------------|
| `drop_metrics` | Exclude specific metrics from the objective. |
| `obj_weights` | Non-uniform weights for averaging BPB across tasks. Default is uniform. |

### Output

All results are written to a hashed subdirectory under the `--output-dir` you specify. The subdirectory name is derived from a hash of the config, so different configurations produce separate output folders.

| File | Description |
|------|-------------|
| `config.json` | Full configuration used for this fit (for reproducibility) |
| `interaction_matrix.png` | Heatmap of regression coefficients: rows are domains, columns are metrics. Shows which domains help or hurt each metric. |
| `interaction_matrix.npy` | Raw interaction matrix as a NumPy array (for downstream analysis). |
| `{metric}_*_fit.png` | Per-metric regression plot: predicted vs. actual values. Tight clustering along the diagonal means the model fits well. |
| `{metric}_*_correlations.json` | Correlation metrics (e.g. R²) for each regression fit. |
| `path_to_regression_model.txt` | Path to the cached regression model (pickle). Reused on subsequent fits with the same regression config. |

By default (unless `fit_only: true`), the proposer step also produces:

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
- **`LaunchConfig`** — controls how training runs are launched (infra, training hyperparams, eval, **mix**). See configs in [`configs/experiments/`](configs/experiments/).

Every `LaunchConfig` requires an explicit top-level `mix` field that maps domain keys to weights and repetition factors. The `data.sources` section describes *what data exists*; the `mix` section describes *how much of each domain to use*.

The `mix` supports two formats — **nested** (recommended for hand-written configs) and **flat** (used by generated configs). Both are equivalent; nested mixes are auto-flattened on load.

**Nested format** — mirrors the source/topic/quality hierarchy. Weights at each level are multiplied to get the final leaf weight. `repetition_factor` is inherited from the nearest ancestor that sets it:

```yaml
mix:
  dclm:
    weight: 0.8
    repetition_factor: 1.0
    science_math_and_technology:
      weight: 0.25
      repetition_factor: 1.0
    software_development:
      weight: 0.625
      repetition_factor: 1.0
    education_and_jobs:
      weight: 0.125
      repetition_factor: 1.0
  wikipedia:
    weight: 0.1
    repetition_factor: 2.0
  arxiv:
    weight: 0.1
    repetition_factor: 1.5
```

For quality-level nesting:

```yaml
mix:
  all_dressed:
    weight: 0.98
    repetition_factor: 1.0
    science:
      weight: 0.20
      high: { weight: 0.70, repetition_factor: 1.0 }
      med: { weight: 0.30, repetition_factor: 1.0 }
    code:
      weight: 0.50
      high: { weight: 0.70, repetition_factor: 1.0 }
      med: { weight: 0.30, repetition_factor: 1.0 }
  arxiv:
    weight: 0.02
    repetition_factor: 1.5
```

**Flat format** — colon-separated domain keys, each with `weight` and `repetition_factor`. This is what `olmix generate` produces:

```yaml
mix:
  dclm:science_math_and_technology:
    weight: 0.2
    repetition_factor: 1.0
  dclm:software_development:
    weight: 0.5
    repetition_factor: 1.0
  wikipedia:
    weight: 0.1
    repetition_factor: 2.0
```

### Step 0: Compute priors (token counts)

Before generating mixes, compute the token counts for your data sources. This scans S3 paths and outputs the `priors` block for your generation config:

```bash
olmix priors compute --config configs/generations/example.yaml
```

This outputs a YAML block you can paste directly into your generation config:

```yaml
priors:
  token_counts:
    arxiv: 21377485731
    dclm:education_and_jobs: 20771836713
    dclm:science_math_and_technology: 84526121193
    dclm:software_development: 23878302458
    wikipedia: 3692487830
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
description: Data proportions experiment - balanced baseline mix
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
      paths:
      - s3://...
  - name: wikipedia
    paths:
    - s3://...
eval:
  tasks: { ... }
mix:
  dclm:science_math_and_technology:
    weight: 0.55
  wikipedia:
    weight: 0.10
group_id: a1b2c3d4
```

Inspect and edit these files before launching — this is the point where you have full control over what gets trained.

### Step 2: Preview training commands

Renders the full OLMo training command for each variant. The `--variants` flag accepts a directory of configs or a single config file. Prints to stdout without launching anything.

```bash
olmix launch preview --variants output/my_variants/          # directory
olmix launch preview --variants configs/experiments/data_proportions/mix_heavy_code.yaml  # single file
```

### Step 3: Launch a swarm

Submits one Beaker job per variant. Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Launch metadata is saved in the variants directory.

```bash
olmix launch run --variants output/my_variants/
olmix launch run --variants configs/experiments/data_proportions/mix_heavy_code.yaml  # single file
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
