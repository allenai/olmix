# Add YAML config for `olmix fit`

## Context

Follow-up to plan 017 (ExperimentConfig decomposition). Currently `olmix fit` is configured entirely via CLI flags (~35 of them), making complex invocations hard to reproduce. The user wants a YAML config for `olmix fit` that is **completely independent** of the launch ExperimentConfig YAMLs, mirroring how `olmix launch` takes a YAML config.

The immediate use case: fitting against DCLM swarm data in `dclm_ratios.csv` (127 runs, 24 domains) and `dclm_metrics.csv` (127 runs, ~110 eval metrics).

## Design

### YAML schema (nested sections)

```yaml
swarm:
  ratios: path/to/ratios.csv          # Required — CSV with domain mixture ratios per run
  metrics: path/to/metrics.csv        # Required — CSV with eval metrics per run

priors:                                # Required — token distribution across domains
  relative_sizes: {domain: weight}
  total_tokens: int
  token_counts: {domain: count}

regression:
  type: log_linear                     # log_linear | lightgbm | search | gp | autoscale | bimix
  alpha: 1.0
  seed: 0
  n_test: 0
  train_split: [1.0]
  simulation_samples: 100000
  opt_avg_metric: false
  aggregate_task_families: false

proposer:
  type: exact                          # exact | simulation | search
  temperature: null
  kl_reg: null
  use_natural_kl: false
  fit_only: false
  make_worst_mix: false

constraints:
  enabled: false                       # maps to constrain_objective
  target_tokens: null
  target_chinchilla_multiple: null
  target_model_id: null
  repetition_factor: 5.0
  requested_tokens: null

filtering:
  keep_sources: []
  support_domains: []
  drop_metrics: []
  fixed_weight: {}                     # native dict, not JSON string
  obj_weights: {}
```

### Key decisions

- **CSV-only**: `from_wandb` is removed from `olmix fit` entirely. A separate script will handle WandB-to-CSV conversion in the future.
- **`swarm` section**: named after the swarm runs that produced the CSVs; direct file paths for flexibility
- **Inline priors only**: priors fully specified in YAML (no `priors_path` indirection)
- **`fixed_weight` as native dict**: YAML uses dict syntax instead of JSON string
- **No `output`/`evaluation` sections**: these were low-value sections; output dir is auto-generated
- **YAML-only interface**: `olmix fit --config <yaml>` is the complete interface, no CLI override flags

### CSV loader changes

Rewrite `load_from_csv()` to take `ratios_path` and `metrics_path` as direct file paths (no `csv_dir` pattern). Also accept both `run_id` and `run` as the ID column since the DCLM CSVs use `run`.

## Implementation steps

### Step 1: Create example config at `configs/fits/dclm_baseline.yaml`

Write the example YAML config using realistic values from the DCLM data with inline priors.

### Step 2: Create Pydantic models in `olmix/fit/config.py`

New file with Pydantic `BaseModel` sub-configs following the same convention as `ExperimentConfig` (`InfraConfig`, `TrainingConfig`, etc.):

- `SwarmDataConfig` — swarm CSV paths
- `PriorsConfig` — inline token distribution
- `RegressionConfig` — regression model settings
- `ProposerConfig` — mixture proposer settings
- `ConstraintsConfig` — token budget constraints
- `FilteringConfig` — domain/metric filtering
- `FitConfig` — top-level, composes all sub-configs

`FitConfig` has a `from_yaml(path)` classmethod (same pattern as loading `ExperimentConfig` from YAML in `cli.py`). `PriorsConfig` has `to_tuple()` returning `(relative_sizes, total_tokens, token_counts)` for passing to `run_fit()`.

Note: the existing `FitConfig` dataclass (used to snapshot CLI params to JSON) will be renamed or removed since the new `FitConfig` Pydantic model replaces it.

### Step 3: Update `olmix/fit/loaders.py`

- Rewrite `load_from_csv()` to take `ratios_path: str, metrics_path: str` (direct file paths, no `csv_dir`)
- Accept both `run_id` and `run` as the ID column

### Step 4: Rewrite `olmix/fit/cli.py`

- Remove all existing CLI flags (all ~35 of them)
- Single interface: `olmix fit --config <yaml>` (required)
- Load `FitConfig` from YAML, resolve priors, load CSVs, call `run_fit()`
- `FitConfig` dataclass stays (for saving config snapshot to output dir) but is built from `FitConfig`

### Step 5: Update tests

- Add `tests/test_fit_config.py`: YAML loading, validation, priors resolution, round-trip
- Update `tests/test_config.py` if any existing fit-related tests need adjustment

## Key files

| File | Change |
|---|---|
| `configs/fits/dclm_baseline.yaml` | **New** — example fit config |
| `olmix/fit/config.py` | **New** — Pydantic models for FitConfig |
| `olmix/fit/cli.py` | Rewrite: remove all ~35 CLI flags; single `--config <yaml>` interface; build `fit()` around `FitConfig` |
| `olmix/fit/loaders.py` | Remove `load_from_wandb()`. Rewrite `load_from_csv(ratios_path, metrics_path)` to take direct file paths; accept `run` column |
| `olmix/fit/core.py` | No changes (interface stays the same) |
| `tests/test_fit_config.py` | **New** — tests for YAML config loading |

## Verification

1. `python -c "from olmix.fit.config import FitConfig; c = FitConfig.from_yaml('configs/fits/dclm_baseline.yaml'); print(c)"` — YAML loads and validates
2. `olmix fit --config configs/fits/dclm_baseline.yaml --fit-only` — end-to-end regression fit using YAML config with the DCLM data (no mixture proposal, just validates the pipeline)
3. `olmix fit --config configs/fits/dclm_baseline.yaml` — full fit + propose optimal mixture
4. `python -m pytest tests/ -x -q` — all tests pass
5. `ruff check olmix/` — lint clean
