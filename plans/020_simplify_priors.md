# Simplify and unify PriorsConfig across fit and launch (Issue #15)

## Context

`PriorsConfig` has three redundant fields (`relative_sizes`, `total_tokens`, `token_counts`) — any one can derive the others. Additionally, `olmix launch` auto-computes priors from S3 instead of requiring them in the config, making the system "magic" and inconsistent with `olmix fit` which requires explicit priors.

Goals:
1. Keep only `token_counts` in `PriorsConfig`; derive the rest as computed properties
2. Add `PriorsConfig` to `ExperimentConfig` so launch uses explicit priors too
3. Add `olmix priors compute` CLI to scan S3 and output token_counts for a config
4. Replace the opaque `priors: tuple` pattern in `run_fit()` with explicit `relative_sizes: dict`

## Implementation

### Step 1: Rewrite `PriorsConfig` in `olmix/fit/config.py`

- Single stored field: `token_counts: dict[str, int]`
- `@property relative_sizes` — normalize to sum to 1
- `@property total_tokens` — `sum(token_counts.values())`
- `model_validator(mode="before")` — strip `relative_sizes` and `total_tokens` from input (backward compat)
- Remove `to_tuple()` method

### Step 2: Add `priors` to `ExperimentConfig` in `olmix/aliases.py`

- Add `priors: PriorsConfig` as required field
- Import `PriorsConfig` from `olmix.fit.config`

### Step 3: Add `olmix priors compute` CLI in `olmix/cli.py`

New command group + subcommand:
```
olmix priors compute --config <experiment.yaml> [--no-cache] [--output <file>]
```
- Reads `data.sources` from experiment config
- Calls existing `calculate_priors()` from `synthesize_mixture.py`
- Outputs YAML-formatted `priors: token_counts: {...}` to stdout or file

### Step 4: Update launch flow in `olmix/launch/synthesize_mixture.py`

In `mk_mixtures()` (line 494): use `config.priors` (required) instead of calling `calculate_priors()`. Remove the auto-compute call.

### Step 5: Replace `priors: tuple` in `olmix/fit/core.py`

Change `run_fit()` signature:
- `priors: tuple` → `relative_sizes: dict[str, float]`
- `original_priors: tuple` → `original_relative_sizes: dict[str, float]`

Update all `priors[0]` → `relative_sizes` and `original_priors[0]` → `original_relative_sizes` references (lines ~149, 454, 468-469).

Note: `original_relative_sizes` is the untouched copy used only for plotting (showing original distribution vs proposed optimal). `relative_sizes` is the working copy that gets domains removed via `fixed_weight` and `support_domains` filtering.

### Step 6: Update `olmix/fit/cli.py`

- Remove `to_tuple()` usage
- Build `relative_sizes` and `original_relative_sizes` as plain dicts from `cfg.priors.relative_sizes`
- Pass to `run_fit()` with new parameter names

### Step 7: Simplify `configs/fits/dclm_baseline.yaml`

Remove `relative_sizes:` and `total_tokens:` from `priors:`, keep only `token_counts:`.

### Step 8: Add priors to all 31 experiment configs

Run a script that calls `calculate_priors()` for each config (uses existing cache in `cache/priors_cache_*.json`) and inserts `priors: token_counts: {...}` into each YAML. Two known cache entries cover the data_proportions/training_duration configs (5 domains) and quality_thresholds/quality_upsampling configs (14 domains).

### Step 9: Update tests

- `tests/test_fit_config.py` — new PriorsConfig tests (computed properties, backward compat, model_dump)
- `tests/test_config.py` — add required priors to all ExperimentConfig test fixtures
- `tests/test_cli.py` — test `priors compute --help`

### Step 10: Update README

- Simplify `priors` YAML example to show only `token_counts`
- Document `olmix priors compute` command
- Update workflow description

## Key files

| File | Change |
|---|---|
| `olmix/fit/config.py` | Rewrite PriorsConfig: single field + computed properties |
| `olmix/aliases.py` | Add `priors: PriorsConfig` (required) to ExperimentConfig |
| `olmix/cli.py` | Add `olmix priors compute` command; update launch to prefer config.priors |
| `olmix/launch/synthesize_mixture.py` | `mk_mixtures()` uses config.priors instead of auto-computing |
| `olmix/fit/core.py` | `priors: tuple` → `relative_sizes: dict[str, float]` |
| `olmix/fit/cli.py` | Remove `to_tuple()`, pass dicts directly |
| `configs/fits/dclm_baseline.yaml` | Remove redundant fields |
| `configs/experiments/**/*.yaml` (31 files) | Add `priors: token_counts: {...}` from cached S3 scans |
| `tests/test_fit_config.py` | Update PriorsConfig tests |
| `README.md` | Simplify priors docs, add priors compute docs |

## Files NOT changed

| File | Reason |
|---|---|
| `olmix/fit/utils.py` | Proposers already receive `prior_distributions` as plain dict |

## Verification

1. `make run-checks` — format, lint, pyright, pytest all pass
2. `olmix fit --config configs/fits/dclm_baseline.yaml --output-dir /tmp/test` — end-to-end fit
3. `olmix launch preview --config configs/experiments/data_proportions/mix_baseline.yaml` — uses explicit priors from config
4. `olmix priors compute --help` — new CLI command exists
