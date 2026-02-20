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

### How to run

`olmix fit` is configured via a YAML file containing `ratios.csv` and `metrics.csv`. Run it with:

```bash
olmix fit --config configs/examples/fit/example.yaml --output-dir output/my_fit
```

| Flag | Description |
|------|-------------|
| `--config` | Path to the YAML fit configuration file |
| `--output-dir` | Directory for saving fit outputs |

See [`configs/examples/fit/example.yaml`](configs/examples/fit/example.yaml) for a full example. The config has these sections:

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

## Part 2: Generating swarm mixtures

`olmix generate` samples a swarm of mixtures from a `GenerationConfig` YAML and writes each one as a `LaunchConfig` file used for training the proxy models. A key capability supported by `olmix generate` is **mixture reuse**: freeze the relative topic weights within the swarm. See [`configs/examples/generate/example.yaml`](configs/examples/generate/example.yaml) for a basic `GenerationConfig` and [`configs/examples/generate/partial_mixture_reuse.yaml`](configs/examples/generate/partial_mixture_reuse.yaml) for a partial mixture reuse example.

### Step 0: Compute priors (token counts)

Before generating mixes, set the priors for the data paths in your config. There are two fields: 1. **relative_sizes**, which is used as the Dirichlet prior and 2. **token_counts**, which is used to enforce repetition constraints on the swarm (by default, we ensure no data is repeated at the proxy model scale). These priors can be set manually or computed automatically using `olmix priors compute` to be the natural distribution and the actual sizes of the data paths:

```bash
olmix priors compute --config configs/examples/generate/example.yaml
```

This scans S3 paths and outputs a `priors:` block to paste into your generation config:

```yaml
priors:
  relative_sizes:
    arxiv: 0.13859324268101414
    dclm:education_and_jobs: 0.13466673502770904
    dclm:science_math_and_technology: 0.5479947162541395
    dclm:software_development: 0.1548063887874921
    wikipedia: 0.023938917249645256
  token_counts:
    arxiv: 21377485731
    dclm:education_and_jobs: 20771836713
    dclm:science_math_and_technology: 84526121193
    dclm:software_development: 23878302458
    wikipedia: 3692487830
```

### Step 1: Generate candidate mixtures

Use `olmix generate` to sample mixture variants from a generation config. The `--base` flag provides a `LaunchConfig` template (infra, training, eval settings); each variant inherits from it and gets a unique sampled `mix` written into it.

```bash
olmix generate \
  --config configs/examples/generate/example.yaml \
  --base configs/examples/launch/data_proportions/mix_baseline.yaml \
  --output output/my_variants/
```

This produces one self-contained `LaunchConfig` YAML per variant:

```
output/my_variants/
  example-swarm-a1b2c3d4-0000.yaml
  example-swarm-a1b2c3d4-0001.yaml
  ...
```

Inspect and edit these files before launching — this is where you have full control over what gets trained.

### Step 2: Launch a swarm

```bash
olmix launch run --variants output/my_variants/
```

Submits one training job per variant. Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Use `--dry-run` to generate metadata without launching.

### Step 3: Export to CSV and fit

Once runs complete, export ratios and metrics to CSV files (e.g. from W&B), then fit using the workflow in [Part 1](#part-1-mixture-optimization-from-csv-data).

### GenerationConfig reference

```yaml
name: my-swarm

data:            # What data sources exist and how they're organized
priors:          # Natural token distribution at the leaf level (from olmix priors compute)
swarm:           # Sampling parameters
max_tokens:      # Token budget per proxy run
```

#### `data`

`data.sources` lists data pools in a hierarchy: **source → topic → quality**. Each source specifies exactly one of `paths` (flat source), `topics`, or `quality`.

An optional `weight` field can appear on any topic, which pins its share within that source's allocation (values within a source should sum to ~1.0). Anything without a `weight` is sampled from the Dirichlet and varies freely across runs. This is the **mixture reuse** pattern: freeze the existing ratios, and only recompute on affected domains. Note that topics and sources are used here relatively; for example, the aggregated virtual domain `existing` is a source while `wikipedia` is a topic within it:

```yaml
data:
  sources:
  - name: existing
    topics:
    - name: dclm:science_math_and_technology
      paths: [...]
      weight: 0.55   # frozen from prior optimization
    - name: dclm:software_development
      paths: [...]
      weight: 0.30   # frozen
    - name: dclm:entertainment
      paths: [...]   # no weight → sampled freely in each variant
      weight: 0.1
    - name: wikipedia
      paths: [...]
      weight: 0.05
  - name: stack-edu
    topics:
    - name: Python
      paths: [...]   # free to vary
    - name: Java
      paths: [...]   # free to vary
```

For this example, the domains to recompute are `existing`, `stack-edu:Python`, and `stack-edu:Java`.

#### `priors`

Must be at the **leaf level** (e.g. `dclm:science_math_and_technology`, not `dclm`). `relative_sizes` defines the Dirichlet prior center for free domains; `token_counts` enforces the repetition constraint (no domain sampled past `repetition_factor` × its available data).

#### `swarm`

| Field | Description | Default |
|-------|-------------|---------|
| `variants` | Number of mixture variants to generate | `1` |
| `seed` | Random seed | `42` |
| `min_strength` / `max_strength` | Dirichlet concentration range. Low = diverse/extreme mixes; high = mixes near the prior | `0.1` / `5.0` |
| `min_source_strength` / `max_source_strength` | Override strength for source-level sampling | — |
| `min_topic_strength` / `max_topic_strength` | Override strength for topic-level sampling | — |
| `minimum_weight` | Domains below this are zeroed out | `0.002` |
| `minimum_source_weight` / `minimum_topic_weight` | Override `minimum_weight` at source or topic level | — |
| `nonzero_weight` | Domain keys that must be nonzero in every variant | — |
| `manual_prior` | Override source-level Dirichlet prior, e.g. `{dclm: 0.75, stack-edu: 0.25}` | — |
| `manual_topic_prior` | Override topic-level Dirichlet prior for specific keys (topic still sampled) | — |
| `repetition_factor` | Max allowed data repetition per domain | `1.0` |
| `enable_bound` | Enforce the repetition bound when sampling | `true` |
| `existing_mix_file` | Pickle of prior swarm ratios; new samples too close are rejected | — |

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
