# Decompose `ExperimentConfig` into typed sub-configs

## Context

`ExperimentConfig` in `olmix/aliases.py` is a 54-field god object that mixes unrelated concerns: Beaker infrastructure, OLMo Core training, data sources, mixture sampling, and fit constraints. The user wants clean abstractions so each config type is self-contained and its purpose is immediately obvious.

### The config types after this refactor

1. **`ExperimentConfig`** (YAML) = `InfraConfig` + `TrainingConfig` + `DataConfig` + `SwarmConfig` — governs `olmix launch` and `olmix mix generate`
2. **`FitConfig`** (CLI dataclass, already exists in `olmix/fit/cli.py`) — governs `olmix fit`. Absorbs constraint parameters that were previously in ExperimentConfig. Rename `config` field to `experiment_config_path`.

Key principle: **`DataConfig` + `TrainingConfig` = everything needed to train an OLMo Core model.** `SwarmConfig` is the outer-loop that samples candidate mixtures. `InfraConfig` is where to run it.

## Design

### `ExperimentConfig` sub-configs (all in `olmix/aliases.py`)

**`InfraConfig`** — Beaker launch infrastructure (10 fields):
```
budget, workspace, cluster, priority, preemptible, nodes, gpus, weka, shared_filesystem, wandb_debug
```

**`TrainingConfig`** — OLMo Core model + training + eval (12 fields):
```
proxy_model_id, tokenizer, chinchilla_multiple, seed, device_batch_size, global_batch_size,
checkpoint_path, train_type, eval_interval, eval_tasks, no_eval, instance_filter
```
Includes `get_max_tokens()` method (uses `chinchilla_multiple` + `proxy_model_id`).
- `seed`: OLMo Core training seed (passed as `-S` to training script). Separate from swarm seed.
- `instance_filter`: training-time repetitive sequence filtering (OLMo Core behavior).

**`DataConfig`** — data landscape, what data exists and how it's stored (2 fields):
```
sources, dtype
```
- `sources: list[SourceConfig]` — data sources with paths, topics, quality buckets, max_repetition_factor
- `dtype: NumpyDatasetDType` — token data format (needed by both training to read data and by priors calculation to count tokens)

Combined with `TrainingConfig`, this is everything OLMo Core needs to train a model.

**`SwarmConfig`** — outer-loop mixture sampling parameters (20 fields):
```
seed, variants, mix_temperature, source_mix_temperature, topic_mix_temperature,
min_strength, max_strength, min_source_strength, max_source_strength,
min_topic_strength, max_topic_strength, minimum_weight, minimum_source_weight,
minimum_topic_weight, nonzero_weight, fixed_source_weights, manual_prior,
manual_topic_prior, allow_repetition, sample_multiplier, existing_mix_file
```
- `seed`: Dirichlet sampling seed for `synthesize_mixture.py`. Independent from training seed.

### Composed `ExperimentConfig`

```python
class ExperimentConfig(BaseModel):
    name: str
    description: str = ""
    infra: InfraConfig
    training: TrainingConfig
    data: DataConfig
    swarm: SwarmConfig
```

No `model_validator`. No constraint fields — those move to `FitConfig`.

### Constraint parameters move to `FitConfig`

The 4 constraint fields (`target_tokens`, `target_chinchilla_multiple`, `target_model_id`, `repetition_factor`) are exclusively used by `olmix fit`. They become CLI flags on `olmix fit` and fields in `FitConfig`:

| Old (ExperimentConfig YAML) | New (olmix fit CLI flag / FitConfig field) |
|---|---|
| `target_tokens` | `--target-tokens` (replaces `--requested-tokens`) |
| `target_chinchilla_multiple` | `--target-chinchilla-multiple` (new) |
| `target_model_id` | `--target-model-id` (new) |
| `repetition_factor` | `--repetition-factor` (new, default 5.0) |

`get_target_tokens()` moves to a standalone helper in `olmix/fit/utils.py`. `compute_constraints_from_config()` takes constraint params directly.

### YAML format (nested)

```yaml
name: mix-baseline
description: Data proportions experiment - balanced baseline mix

infra:
  budget: ai2/oe-base
  workspace: ai2/oe-data
  cluster: ai2/jupiter
  priority: high
  nodes: 1
  gpus: 1

training:
  proxy_model_id: olmo3_14m
  tokenizer: dolma2
  chinchilla_multiple: 0.5
  seed: 42              # OLMo Core training seed
  global_batch_size: 16

data:
  dtype: uint32
  sources:
    - name: dclm
      max_repetition_factor: 1.0
      topics:
        - name: science_math_and_technology
          paths: [...]
    - name: wikipedia
      paths: [...]
      max_repetition_factor: 2.0

swarm:
  seed: 42              # Dirichlet sampling seed
  variants: 1
```

### Complete field mapping (all 50 ExperimentConfig fields)

| Field | New location |
|---|---|
| `name` | ExperimentConfig (top-level) |
| `description` | ExperimentConfig (top-level) |
| `budget` | InfraConfig |
| `workspace` | InfraConfig |
| `cluster` | InfraConfig |
| `priority` | InfraConfig |
| `preemptible` | InfraConfig |
| `nodes` | InfraConfig |
| `gpus` | InfraConfig |
| `weka` | InfraConfig |
| `shared_filesystem` | InfraConfig |
| `wandb_debug` | InfraConfig |
| `proxy_model_id` | TrainingConfig |
| `tokenizer` | TrainingConfig |
| `chinchilla_multiple` | TrainingConfig |
| `seed` | TrainingConfig (training seed) + SwarmConfig (sampling seed) |
| `device_batch_size` | TrainingConfig |
| `global_batch_size` | TrainingConfig |
| `checkpoint_path` | TrainingConfig |
| `train_type` | TrainingConfig |
| `eval_interval` | TrainingConfig |
| `eval_tasks` | TrainingConfig |
| `no_eval` | TrainingConfig |
| `instance_filter` | TrainingConfig |
| `dtype` | DataConfig |
| `sources` | DataConfig |
| `variants` | SwarmConfig |
| `mix_temperature` | SwarmConfig |
| `source_mix_temperature` | SwarmConfig |
| `topic_mix_temperature` | SwarmConfig |
| `min_strength` | SwarmConfig |
| `max_strength` | SwarmConfig |
| `min_source_strength` | SwarmConfig |
| `max_source_strength` | SwarmConfig |
| `min_topic_strength` | SwarmConfig |
| `max_topic_strength` | SwarmConfig |
| `minimum_weight` | SwarmConfig |
| `minimum_source_weight` | SwarmConfig |
| `minimum_topic_weight` | SwarmConfig |
| `nonzero_weight` | SwarmConfig |
| `fixed_source_weights` | SwarmConfig |
| `manual_prior` | SwarmConfig |
| `manual_topic_prior` | SwarmConfig |
| `allow_repetition` | SwarmConfig |
| `sample_multiplier` | SwarmConfig |
| `existing_mix_file` | SwarmConfig |
| `target_tokens` | **Removed** → FitConfig CLI `--target-tokens` |
| `target_chinchilla_multiple` | **Removed** → FitConfig CLI `--target-chinchilla-multiple` |
| `target_model_id` | **Removed** → FitConfig CLI `--target-model-id` |
| `repetition_factor` | **Removed** → FitConfig CLI `--repetition-factor` |

### Source/Topic/Quality sub-model fields (unchanged)

These stay on their Pydantic models, nested under `DataConfig.sources`:

| Model | Fields |
|---|---|
| `SourceConfig` | `name`, `paths`, `topics`, `quality`, `max_repetition_factor` |
| `TopicConfig` | `name`, `paths`, `quality`, `max_repetition_factor`, `max_topic_ratio`, `weight` |
| `QualityConfig` | `name`, `paths`, `weight`, `max_repetition_factor` |

### File changes

| File | Changes |
|---|---|
| `olmix/aliases.py` | Define `InfraConfig`, `TrainingConfig`, `DataConfig`, `SwarmConfig`. Remove constraint fields + `get_target_tokens()`. Compose `ExperimentConfig` from 4 sub-configs. |
| `olmix/launch/beaker.py` | `config.cluster` → `config.infra.cluster`, `config.seed` → `config.training.seed`, `config.dtype` → `config.data.dtype`, etc. (~25 field accesses) |
| `olmix/launch/synthesize_mixture.py` | `config.sources` → `config.data.sources`, `config.dtype` → `config.data.dtype`, `config.seed` → `config.swarm.seed`, `config.variants` → `config.swarm.variants`, etc. (~22 field accesses) |
| `olmix/launch/launch_utils.py` | No changes (already takes `list[SourceConfig]` directly) |
| `olmix/cli.py` | `config.gpus` → `config.infra.gpus`, etc. (~5 field accesses) |
| `olmix/fit/cli.py` | Add new CLI flags for constraints. Rename `FitConfig.config` → `FitConfig.experiment_config_path`. Update sub-config field accesses. |
| `olmix/fit/utils.py` | Refactor `compute_constraints_from_config()` to take constraint params directly. Update sub-config field accesses. |
| `olmix/fit/loaders.py` | Update sub-config field accesses (~6 changes) |
| `olmix/fit/core.py` | Update sub-config field accesses (~4 changes) |
| `config/examples/launch/**/*.yaml` (31 files) | Convert flat YAML to nested format with `infra:`, `training:`, `data:`, `swarm:` sections |
| `tests/test_config.py` | Update test fixtures to use nested format |

## Implementation steps

### Step 1: Define sub-config classes in `olmix/aliases.py`
- Add `InfraConfig`, `TrainingConfig`, `DataConfig`, `SwarmConfig` as Pydantic BaseModel classes
- Move `get_max_tokens()` to `TrainingConfig`
- Remove constraint fields + `get_target_tokens()` from ExperimentConfig
- Compose ExperimentConfig from 4 sub-configs

### Step 2: Convert all 31 YAML configs to nested format
- Group flat keys under `infra:`, `training:`, `data:`, `swarm:` sections
- Split `seed` into `training.seed` and `swarm.seed`
- Move `dtype` and `sources` under `data:`
- Move sampling params under `swarm:`
- Remove any constraint fields

### Step 3: Update `olmix/launch/beaker.py`
- Training fields via `config.training.*`, infra fields via `config.infra.*`, data fields via `config.data.*`

### Step 4: Update `olmix/launch/synthesize_mixture.py`
- Data fields via `config.data.*`, swarm fields via `config.swarm.*`, `get_max_tokens()` via `config.training.get_max_tokens()`

### Step 5: Update `olmix/cli.py`
- `config.gpus` → `config.infra.gpus`, etc.

### Step 6: Update `olmix/fit/cli.py`
- Add CLI flags: `--target-tokens`, `--target-chinchilla-multiple`, `--target-model-id`, `--repetition-factor`
- Remove `--requested-tokens`
- Rename `FitConfig.config` → `FitConfig.experiment_config_path`
- Update sub-config field accesses

### Step 7: Update `olmix/fit/utils.py`
- Refactor `compute_constraints_from_config()` to take constraint params directly
- Move `get_target_tokens()` logic to standalone helper
- Update sub-config field accesses

### Step 8: Update `olmix/fit/loaders.py` and `olmix/fit/core.py`
- Update sub-config field accesses

### Step 9: Update tests
- Update `tests/test_config.py` fixtures to use nested format
- Run full test suite

## Verification

1. `python -c "from olmix.aliases import ExperimentConfig, InfraConfig, TrainingConfig, DataConfig, SwarmConfig"` — imports work
2. `python -c "import yaml; from olmix.aliases import ExperimentConfig; ExperimentConfig(**yaml.safe_load(open('config/examples/launch/data_proportions/mix_baseline.yaml')))"` — nested YAML loading works
3. `python -m pytest tests/ -x -q` — all existing tests pass
4. `ruff check olmix/ && pyright olmix/` — lint + typecheck pass
5. `olmix launch run --config config/examples/launch/data_proportions/mix_baseline.yaml --dry-run` — dry-run successfully outputs metadata files to `output/mixes/`
6. Prepare an `olmix launch run --config config/examples/launch/data_proportions/mix_baseline.yaml` command for user review (do NOT execute — user will review and launch manually)

## Key files

| File | Role |
|---|---|
| `olmix/aliases.py:139-231` | Current ExperimentConfig (primary target) |
| `olmix/launch/beaker.py:62-183` | Heaviest user of training + infra fields |
| `olmix/launch/synthesize_mixture.py:486-559` | Heaviest user of data + swarm fields |
| `olmix/fit/utils.py:67-91` | `compute_constraints_from_config` — constraint logic to refactor |
| `olmix/fit/cli.py:30-98` | FitConfig dataclass — absorbs constraint fields, rename `config` |
| `olmix/fit/cli.py:350-593` | `fit()` — loads ExperimentConfig, uses data/swarm fields |
| `olmix/fit/loaders.py` | `load_from_wandb` — reconstructs ExperimentConfig from JSON |
| `config/examples/launch/**/*.yaml` | 31 YAML files to convert to nested format |
