# `olmix fit` Reference

## Purpose

Fit scaling/regression models from swarm CSV outputs and optionally propose
optimized mixture weights.

Canonical command:

```bash
olmix fit --config <fit-config.yaml> --output-dir <dir>
```

Legacy compatibility alias:

```bash
olmix-fit --config <fit-config.yaml> --output-dir <dir>
```

## Command Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to YAML fit config file |
| `--output-dir` | Yes | Output root directory for fit artifacts |

## Minimal Example

```bash
olmix fit --config configs/examples/fit/example.yaml --output-dir output/my_fit
```

## Input Data Expectations

`swarm.ratios` and `swarm.metrics` are CSV files joined by `run` or `run_id`.

- `ratios.csv`: metadata columns plus domain-weight columns (weights sum to ~1.0).
- `metrics.csv`: metadata columns plus metric columns.
- Metadata columns skipped during fitting: `run`/`run_id`, `name`, `index`, unnamed index columns.

## Fit Config Schema

Only `swarm` and `priors` are required. Other sections are optional.

```yaml
swarm:
  ratios: path/to/ratios.csv
  metrics: path/to/metrics.csv

priors:
  relative_sizes:
    domain_a: 0.6
    domain_b: 0.4
  token_counts:
    domain_a: 600000000
    domain_b: 400000000

eval:
  tasks:
    math: ["metric_a"]
    code: ["metric_b"]

regression:
  type: log_linear
  seed: 0
  n_test: 0
  train_split: 1.0
  aggregate_task_families: false

proposer:
  type: exact
  temperature: null
  kl_reg: 0.1
  fit_only: false
  make_worst_mix: false

constraints:
  enabled: false
  target_tokens: null
  repetition_factor: 4.0

filtering:
  drop_metrics: []
  obj_weights: {}
```

### `priors`

| Field | Description |
|------|-------------|
| `relative_sizes` | Prior distribution across domains (typically sums to ~1.0) |
| `token_counts` | Absolute available tokens per domain for repetition constraints |

### `eval`

| Field | Description |
|------|-------------|
| `tasks` | Task-family mapping. If omitted, all metric columns are used |

### `regression`

| Field | Description |
|------|-------------|
| `type` | Regression model: `log_linear`, `lightgbm`, `search`, `gp`, `autoscale`, `bimix` |
| `aggregate_task_families` | Fit by task family instead of per metric |
| `train_split` | Fraction/count of runs used for training |
| `n_test` | Held-out test sample count |
| `seed` | Train/test split seed |

### `proposer`

| Field | Description |
|------|-------------|
| `type` | Optimization search type: `exact`, `simulation`, `search` |
| `kl_reg` | KL regularization strength (`exact` proposer only) |
| `temperature` | Dirichlet temperature for `simulation` proposer |
| `fit_only` | Skip proposal and only fit regressors |
| `make_worst_mix` | Invert objective to propose a bad/counterfactual mix |

### `constraints`

| Field | Description |
|------|-------------|
| `enabled` | Enable repetition constraints |
| `target_tokens` | Required if `enabled: true` |
| `repetition_factor` | Max allowed repetition multiplier |

### `filtering`

| Field | Description |
|------|-------------|
| `drop_metrics` | Metrics to exclude from objective |
| `obj_weights` | Per-metric weighting for aggregate objective |

## Validation Rules And Common Errors

- `constraints.enabled: true` requires `constraints.target_tokens`.
- `regression.aggregate_task_families: true` requires `eval.tasks`.
- `proposer.type: search` requires `regression.type: search`.
- `proposer.kl_reg` is valid only with `proposer.type: exact`.

## Output Artifacts

Fit outputs are written to a hash-named subdirectory under `--output-dir`.

Typical files:

- `config.json`
- `interaction_matrix.png`
- `interaction_matrix.npy`
- `{metric}_*_fit.png`
- `{metric}_*_correlations.json`
- `path_to_regression_model.txt`

If proposal runs (`fit_only: false`), also includes:

- `{metric}_*_optimal.json`
- `{metric}_*_optimal.png`
- `predicted_performance.json`

Primary aggregate output:

- `opt_avg_all_metrics_*_optimal.json`
