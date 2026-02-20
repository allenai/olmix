# Split ExperimentConfig into LaunchConfig + GenerationConfig (Issue #19)

## Context

The current CLI is too magical: `olmix launch run` can dynamically generate mixes on the fly, and a single `ExperimentConfig` mixes generation concerns (priors, swarm) with launch concerns (infra, eval). The desired workflow is explicit and sequential:

1. `olmix priors compute` — scan S3, output token counts
2. `olmix generate` — sample mixes, output per-variant mix files
3. User inspects generated mixes
4. `olmix launch run` — launch from a base config + generated variants (no dynamic mix generation)

## Config Design

**`GenerationConfig`** — input to `olmix generate`. Only what's needed to sample mixes:
```python
class GenerationConfig(BaseModel):
    name: str = ""
    data: DataConfig            # source hierarchy (for Dirichlet structure)
    priors: PriorsConfig        # token counts → prior distribution + feasibility
    swarm: SwarmConfig = SwarmConfig()  # all sampling parameters
    max_tokens: int             # training token budget (for repetition factors)
```

`priors` provides:
- `relative_sizes` → Dirichlet prior distribution
- `total_tokens` / `token_counts` → repetition factor bounds, feasibility checks

`swarm` controls sampling: `variants`, `seed`, `mix_temperature`, `min_strength`/`max_strength`, `minimum_weight`, `nonzero_weight`, `fixed_source_weights`, `manual_prior`, `allow_repetition`, `sample_multiplier`, etc.

`max_tokens` = total training token budget per variant (replaces `20 * model_params * chinchilla_multiple`).

**`LaunchConfig`** — base config for `olmix launch`. Everything needed to run training:
```python
class LaunchConfig(BaseModel):
    name: str
    description: str = ""
    infra: InfraConfig
    training: TrainingConfig
    data: DataConfig
    eval: InLoopEvalConfig
```

**`VariantConfig`** — output of `olmix generate`. One per variant:
```python
class VariantConfig(BaseModel):
    name: str                                       # e.g., "example-a1b2c3d4-0000"
    mix: dict[str, tuple[float, float]]             # domain → (weight, repetition_factor)
```

Rename `ExperimentConfig` → `GenerationConfig` (in generation contexts) or `LaunchConfig` (in launch contexts). No backward compat alias.

## CLI Design

| Command | Input | Output |
|---------|-------|--------|
| `olmix priors compute -c <config>` | Any config with `data.sources` | Token counts YAML to stdout/file |
| `olmix generate -c <gen_config> -o <dir>` | GenerationConfig | N variant YAML files in `<dir>/` |
| `olmix launch run -c <base_config> --variants <dir>` | LaunchConfig + directory of VariantConfigs | Beaker jobs |
| `olmix launch preview -c <base_config> --variants <dir>` | LaunchConfig + directory of VariantConfigs | Training commands to stdout |
| `olmix launch status -g <id> -c <config>` | LaunchConfig (for cluster info) | Job statuses |
| `olmix launch cancel -g <id> -c <config>` | LaunchConfig (for cluster info) | Cancel jobs |

Remove `olmix mix generate` (replaced by top-level `olmix generate`).

## Output format for `olmix generate`

Given `olmix generate -c configs/generations/example.yaml -o output/example/`:

```
output/example/
  example-a1b2c3d4-0000.yaml
  example-a1b2c3d4-0001.yaml
  example-a1b2c3d4-0002.yaml
  example-a1b2c3d4-0003.yaml
```

Each variant file:
```yaml
name: example-a1b2c3d4-0000
mix:
  dclm:science_math_and_technology: [0.55, 1.0]
  dclm:software_development: [0.15, 1.0]
  arxiv: [0.20, 1.0]
  wikipedia: [0.10, 1.5]
```

## Module Reorganization

Move mix generation code out of `olmix/launch/` into `olmix/generate/`:

| From | To | Contents |
|------|----|----------|
| `olmix/launch/synthesize_mixture.py` | `olmix/generate/synthesize_mixture.py` | `mk_mixtures()`, `generate_weights_dirichlet()`, `calculate_priors()` |
| `olmix/launch/launch_utils.py` (generation parts) | `olmix/generate/utils.py` | `mk_mixes()`, `prettify_mixes()` |
| `olmix/launch/launch_utils.py` (launch parts) | `olmix/launch/utils.py` | `mk_source_instances()` |

`olmix/launch/` retains only launch-specific code: `beaker.py`, `utils.py` (`mk_source_instances`), `train.py`.

## Implementation Steps

### Step 1: Define new config types in `olmix/aliases.py`

- Create `GenerationConfig` — name, data, priors, swarm, max_tokens
- Create `LaunchConfig` — name, description, infra, training, data, eval
- Create `VariantConfig` — name, mix
- Add `from_yaml` classmethod to each
- Rename `ExperimentConfig` references throughout
- `ExperimentGroup.config` → `LaunchConfig`, instances carry their mix from VariantConfig

### Step 2: Create `olmix/generate/` package

- `olmix/generate/__init__.py` — exports
- `olmix/generate/synthesize_mixture.py` — move from `olmix/launch/`, update `ExperimentConfig` → `GenerationConfig`, replace `config.training.get_max_tokens()` with `config.max_tokens`
- `olmix/generate/utils.py` — move `mk_mixes()`, `prettify_mixes()` from `olmix/launch/launch_utils.py`

### Step 3: Slim down `olmix/launch/`

- `olmix/launch/launch_utils.py` → `olmix/launch/utils.py` — keep only `mk_source_instances()`
- Update `olmix/launch/__init__.py` — remove generation exports
- Update `olmix/launch/beaker.py` — `ExperimentConfig` → `LaunchConfig`, rework `mk_experiment_group()` to take LaunchConfig + list of VariantConfigs

### Step 4: Add `olmix generate` CLI command in `olmix/cli.py`

```
olmix generate -c <generation_config.yaml> -o <output_dir>
```
- Loads `GenerationConfig.from_yaml(config)`
- Calls `mk_mixes()` from `olmix.generate` to sample N mixes
- Writes each as a VariantConfig YAML: `{name}-{group_uuid}-{idx:04}.yaml`
- Prints summary

### Step 5: Refactor `olmix launch run` in `olmix/cli.py`

- Takes `--config` (LaunchConfig YAML) and `--variants` (directory of VariantConfig YAMLs)
- Loads LaunchConfig + all variant files
- Builds ExperimentGroup by merging LaunchConfig + each VariantConfig's mix
- No more `--mixture-file`, no dynamic mix generation
- Remove `priors` from `_save_launch_metadata`

### Step 6: Refactor `olmix launch preview` in `olmix/cli.py`

- Same as launch run: `--config` + `--variants`
- Preview training commands for all variants

### Step 7: Simplify `olmix priors compute` in `olmix/cli.py`

- Only needs `data.sources` — load with `LaunchConfig` (extra="ignore") or a minimal config
- Remove the placeholder hack

### Step 8: Remove `olmix mix generate` in `olmix/cli.py`

- Remove the `mix` command group (replaced by top-level `olmix generate`)

### Step 9: Update `olmix/fit/utils.py` and `olmix/fit/core.py`

- Rename `ExperimentConfig` → `GenerationConfig` or `LaunchConfig` as appropriate
- Update imports from `olmix.launch.synthesize_mixture` → `olmix.generate.synthesize_mixture`

### Step 10: Update `olmix/__init__.py`

- Export `LaunchConfig`, `GenerationConfig`, `VariantConfig` (remove `ExperimentConfig`)

### Step 11: Update tests

- `tests/test_config.py` — `TestLaunchConfig`, `TestGenerationConfig`, `TestVariantConfig`, update `ExperimentGroup` tests
- `tests/test_cli.py` — `test_generate_help`, update launch tests, remove `test_mix_*`
- `tests/test_imports.py` — update for new module structure
- Rename all `ExperimentConfig` references

### Step 12: Create example generation config

Create `configs/generations/example.yaml`:
```yaml
name: example-swarm
data:
  sources:
    - name: arxiv
      paths: ["s3://ai2-llm/preprocessed/arxiv/.../*.npy"]
    - name: dclm
      topics:
        - name: science_math_and_technology
          paths: ["s3://ai2-llm/preprocessed/dclm/.../science_math_and_technology/**/*.npy"]
        - name: education_and_jobs
          paths: ["s3://ai2-llm/preprocessed/dclm/.../education_and_jobs/**/*.npy"]
        - name: software_development
          paths: ["s3://ai2-llm/preprocessed/dclm/.../software_development/**/*.npy"]
    - name: wikipedia
      paths: ["s3://ai2-llm/preprocessed/wiki/.../*.npy"]
priors:
  token_counts:
    arxiv: 21377485731
    dclm:education_and_jobs: 20771836713
    dclm:science_math_and_technology: 84526121193
    dclm:software_development: 23878302458
    wikipedia: 3692487830
swarm:
  seed: 42
  variants: 4
  mix_temperature: 1.0
  min_strength: 0.1
  max_strength: 5.0
  minimum_weight: 0.002
max_tokens: 140000000  # 20 * 14M * 0.5
```

### Step 13: Update README

- Document the new workflow: priors → generate → launch
- Show example GenerationConfig, LaunchConfig, and variant output
- Update CLI reference

## Key files

| File | Change |
|------|--------|
| `olmix/aliases.py` | Replace ExperimentConfig with GenerationConfig + LaunchConfig + VariantConfig |
| `olmix/cli.py` | Add `olmix generate`, refactor `launch run`/`preview` to take base config + variants dir, remove `mix generate`, simplify `priors compute` |
| `olmix/generate/__init__.py` | New package for mix generation |
| `olmix/generate/synthesize_mixture.py` | Moved from `olmix/launch/`; use `GenerationConfig`, use `config.max_tokens` |
| `olmix/generate/utils.py` | `mk_mixes()`, `prettify_mixes()` moved from `olmix/launch/launch_utils.py` |
| `olmix/launch/beaker.py` | Rework to take LaunchConfig + VariantConfigs |
| `olmix/launch/utils.py` | Slimmed: only `mk_source_instances()` |
| `olmix/launch/__init__.py` | Remove generation exports |
| `olmix/fit/utils.py` | `ExperimentConfig` → appropriate types; update imports |
| `olmix/fit/core.py` | `ExperimentConfig` → appropriate types |
| `olmix/__init__.py` | Export new types |
| `tests/test_config.py` | New config type tests |
| `tests/test_cli.py` | Test `olmix generate`, update launch tests |
| `tests/test_imports.py` | Update for new module structure |
| `configs/generations/example.yaml` | New example GenerationConfig |
| `README.md` | Document new workflow |

## Files deleted

| File | Reason |
|------|--------|
| `olmix/launch/synthesize_mixture.py` | Moved to `olmix/generate/synthesize_mixture.py` |
| `olmix/launch/launch_utils.py` | Split into `olmix/generate/utils.py` + `olmix/launch/utils.py` |

## Files NOT changed

| File | Reason |
|------|--------|
| `olmix/fit/config.py` | PriorsConfig unchanged |
| `olmix/fit/cli.py` | No changes needed |
| `config/examples/launch/**/*.yaml` | Left as-is (old workflow) |

## Verification

1. `make run-checks` — format, lint, pyright, pytest all pass
2. `olmix generate -c configs/generations/example.yaml -o /tmp/test_gen` — produces 4 variant YAML files with name + mix
3. Create a test LaunchConfig, then: `olmix launch preview -c <launch_config> --variants /tmp/test_gen/` — previews training commands
4. `olmix launch run --help` — shows `--config` + `--variants`, no `--mixture-file`
5. `olmix fit --config configs/fits/dclm_baseline.yaml --output-dir /tmp/test_fit` — still works
6. `olmix priors compute -c configs/generations/example.yaml` — works without placeholder hack
