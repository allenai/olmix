# Plan: Refactor `olmix fit` to support local CSV input

## Context

**Current state:**
- `olmix fit` (~1138 lines in `olmix/fit/cli.py`) always pulls swarm results from WandB
- Users must provide `--experiment-groups` (a UUID) and `--config` (YAML path) to run fits
- The group ID comes from `output/{name}-{group_id}/metadata.json` after `olmix launch`

**User request:**
- Support running fits from pre-assembled local CSV data (for reproducibility, offline use, custom pipelines)
- Make data source explicit: `--from-csv` or `--from-wandb` (no default)
- For `--from-wandb`, accept the launch output directory path instead of a raw group ID

**Intended outcome:**
- Two clear data source paths with clean separation
- `--from-wandb` auto-resolves group ID and config from `metadata.json`
- `--from-csv` reads from a directory of CSV files
- Monolithic 1138-line function decomposed into loader + core + CLI orchestrator

## Design decisions

- **CLI**: Explicit `--from-csv DIR_PATH` or `--from-wandb LAUNCH_OUTPUT_DIR` (mutually exclusive, one required)
- **`--from-wandb PATH`**: Takes the launch output directory (e.g., `output/my-swarm-abc12345/`). Reads `metadata.json` to auto-resolve `group_id` and experiment config (stored inline). No `--config` or `--experiment-groups` needed.
- **`--from-csv DIR`**: Takes a directory containing `ratios.csv` and `metrics.csv`. Requires `--config` for priors computation.
- **CSV format**: Two files joined on `run_id`:
  - `ratios.csv` — columns: `run_id, <domain columns...>`
  - `metrics.csv` — columns: `run_id, <metric columns...>`
- **Export**: Not in scope for now.

## Implementation plan

### Step 1: Create `olmix/fit/loaders.py`

Extract WandB data loading (currently `cli.py` lines 471–638) into a new file with two loader functions.

**`load_from_wandb()`** — returns `(ratios, metrics, launch_configs)`:
- Reads `metadata.json` from the provided launch output directory
- Extracts `group_id` from `metadata["metadata"]["group_id"]`
- Reconstructs `ExperimentConfig` from `metadata["config"]` (inline config)
- Initializes WandB API, fetches runs, builds ratios/metrics DataFrames
- Includes existing caching logic (JSON run cache, pickle DataFrame cache)

**`load_from_csv(csv_dir)`** — returns `(ratios, metrics)`:
- Reads `ratios.csv` and `metrics.csv` from the directory
- Validates both files exist and have a `run_id` column
- Validates ratios sum to ~1.0 per row
- Returns DataFrames in the canonical format: `[run, name, index, ...data columns]`
  - `run` = `run_id`, `name` = `run_id` (or `name` column if present), `index` = auto-generated

### Step 2: Create `olmix/fit/core.py`

Extract the fitting/proposing logic (currently `cli.py` lines 641–1138) into `run_fit()`.

```python
def run_fit(
    ratios: pd.DataFrame,
    metrics: pd.DataFrame,
    priors: tuple,
    original_priors: tuple,
    output_dir: str,
    eval_metrics: list[str] | None,  # None → derived from metrics.columns
    experiment_groups: tuple[str, ...] | None,  # None in CSV mode
    launch_configs: list | None,  # None in CSV mode (needed for constraints)
    # ... all fitting/proposing options
) -> None:
```

Key adjustments:
- `eval_metrics` defaults to `metrics.columns[3:]` when `None` (CSV path derives metrics from column names)
- Hardcoded experiment-group outlier checks (lines 673–684) guarded with `if experiment_groups:`
- `full_group_names` references guarded similarly
- `constrain_objective` raises error if `launch_configs is None`

### Step 3: Refactor `olmix/fit/cli.py`

The `fit` command becomes a thin orchestrator.

**CLI option changes:**
- Add `--from-csv` (`type=click.Path(exists=True, file_okay=False)`) — directory with CSVs
- Add `--from-wandb` (`type=click.Path(exists=True, file_okay=False)`) — path to launch output dir
- Keep `--config` but not required at decorator level (required only with `--from-csv`, validated in body)
- Remove `required=True` from `--experiment-groups` (auto-resolved from metadata.json for `--from-wandb`, unused for `--from-csv`)
- WandB-specific options (`--workspace`, `--num-samples`, `--no-cache`, `--pull-from-dashboard`, `--dashboard`, `--use-cookbook`, `--patched`) remain but are only meaningful with `--from-wandb`

**Validation:**
```python
if from_csv and from_wandb:
    raise click.UsageError("--from-csv and --from-wandb are mutually exclusive")
if not from_csv and not from_wandb:
    raise click.UsageError("Must specify either --from-csv or --from-wandb")
if from_csv and not config:
    raise click.UsageError("--config is required when using --from-csv")
```

**Orchestration flow:**
```python
if from_wandb:
    ratios, metrics, launch_configs = load_from_wandb(from_wandb, workspace, ...)
    priors, original_priors = calculate_priors_with_manual(launch_configs[0].sources, ...)
    eval_metrics = ALL_WANDB_METRICS
else:
    ratios, metrics = load_from_csv(from_csv)
    launch_configs = [swarm_config_from_path(c) for c in config]
    priors, original_priors = calculate_priors_with_manual(launch_configs[0].sources, ...)
    eval_metrics = None  # derived from CSV columns in run_fit

run_fit(ratios, metrics, priors, original_priors, eval_metrics=eval_metrics, ...)
```

### Step 4: Update `olmix/fit/__init__.py`

Add new public exports: `load_from_csv`, `load_from_wandb`, `run_fit`.

## Files to modify

| File | Action |
|------|--------|
| `olmix/fit/loaders.py` | **Create** — `load_from_wandb()`, `load_from_csv()` |
| `olmix/fit/core.py` | **Create** — `run_fit()` |
| `olmix/fit/cli.py` | **Refactor** — thin orchestrator with new flags |
| `olmix/fit/__init__.py` | **Update** — add new exports |

## CSV directory format

```
swarm_data/
├── ratios.csv    # run_id, domain1, domain2, ...
└── metrics.csv   # run_id, metric1, metric2, ...
```

Example `ratios.csv`:
```
run_id,dclm,wikipedia,arxiv,s2pdf
run_001,0.45,0.30,0.15,0.10
run_002,0.60,0.20,0.10,0.10
```

Example `metrics.csv`:
```
run_id,arc_challenge_bpb,hellaswag_bpb,mmlu_stem_bpb
run_001,1.23,0.87,1.45
run_002,1.15,0.91,1.38
```

## Verification

1. **Existing behavior preserved**: `olmix fit --from-wandb output/my-swarm-abc12345/ -r log_linear` produces identical results to current `olmix fit -g abc12345 -c config.yaml -r log_linear`
2. **CSV path works**: Create a test directory with synthetic CSVs, run `olmix fit --from-csv ./test_data/ -c config.yaml -r log_linear --fit-only`
3. **Mutual exclusion**: `olmix fit --from-csv X --from-wandb Y` errors
4. **Neither specified**: `olmix fit -r log_linear` errors asking for `--from-csv` or `--from-wandb`
5. **Missing config for CSV**: `olmix fit --from-csv ./data/` errors asking for `--config`
6. **Run existing tests**: `pytest tests/` passes
