# Plan: Simplify olmix to Use Single Config for Constraints

## Goal

All constraint settings come from `ExperimentConfig`. Remove `--final-cookbook-path` and `--manual-token-constraint-path` options.

## Problem

Current constraint system has multiple input paths (cookbook, manual YAML) that are confusing and hard to track. The data paths are always the same - only target tokens and repetition factor differ.

## Solution

Single source of truth: `ExperimentConfig` contains everything needed for constraints.

New fields:
- `target_tokens: int | None` - explicit target tokens for final run
- `target_chinchilla_multiple: float | None` - compute target tokens from formula
- `target_model_id: str | None` - model for final run
- `repetition_factor: float = 5.0` - allowed data repetition

## Changes Required

### 0. Update Model Registry in `olmix/aliases.py`

Add OLMo3 models and verify 1M model exists for fast testing:

```python
MODEL_NUM_PARAMS: dict[str, int] = {
    # OLMo2 models
    "olmo2_1m": 1_000_000,
    "olmo2_30m": 30_000_000,
    "olmo2_60m": 60_000_000,
    "olmo2_190m": 190_000_000,
    "olmo2_1b": 1_000_000_000,
    "olmo2_7b": 7_000_000_000,
    # OLMo3 models (from OLMo-core)
    "olmo3_1m": 1_000_000,
    "olmo3_14m": 14_000_000,
    "olmo3_30m": 30_000_000,
    "olmo3_60m": 60_000_000,
    "olmo3_100m": 100_000_000,
    "olmo3_190m": 190_000_000,
    "olmo3_370m": 370_000_000,
    "olmo3_600m": 600_000_000,
    "olmo3_760m": 760_000_000,
    "olmo3_1b": 1_000_000_000,
    "olmo3_3b": 3_000_000_000,
    "olmo3_7b": 7_000_000_000,
    "olmo3_13b": 13_000_000_000,
    "olmo3_32b": 32_000_000_000,
}
```

### 1. Add Fields to `ExperimentConfig` in `olmix/aliases.py`

```python
# Constraint optimization settings (for olmix fit)
target_tokens: int | None = None
target_chinchilla_multiple: float | None = None
target_model_id: str | None = None
repetition_factor: float = 5.0

def get_target_tokens(self) -> int | None:
    """Compute target tokens for the final run."""
    if self.target_tokens is not None:
        return self.target_tokens
    if self.target_chinchilla_multiple is not None:
        if self.target_model_id is None:
            raise ValueError("target_model_id required when using target_chinchilla_multiple")
        num_params = get_model_num_params(self.target_model_id)
        return compute_max_tokens(self.target_chinchilla_multiple, num_params)
    return None
```

### 2. Add Helper in `olmix/fit/utils.py`

```python
def compute_constraints_from_config(
    config: ExperimentConfig,
    use_cache: bool = True,
) -> tuple[int, dict[str, float], float]:
    """Compute constraints from config's sources and target settings."""
    target_tokens = config.get_target_tokens()
    if target_tokens is None:
        raise ValueError("Config must have target_tokens or target_chinchilla_multiple")

    token_universe = get_token_counts_and_ratios(config.sources, config.dtype, use_cache)
    available_tokens_per_source = {
        path: relative_size * token_universe[1]
        for path, relative_size in token_universe[0].items()
    }
    return target_tokens, available_tokens_per_source, config.repetition_factor
```

### 3. Simplify `olmix/fit/cli.py`

Remove these CLI options:
- `--final-cookbook-path`
- `--manual-token-constraint-path`
- `--repetition-factor`
- `--constrain-swarm`

When `--constrain-objective` is set, use config directly:

```python
if constrain_objective:
    config = launch_configs[0]
    target_tokens = config.get_target_tokens()
    if target_tokens is None:
        raise click.UsageError(
            "For --constrain-objective, set target_tokens or target_chinchilla_multiple in config"
        )
    eval_config["constrain_objective"] = True
    eval_config["target_tokens"] = target_tokens
    eval_config["repetition_factor"] = config.repetition_factor
```

### 4. Simplify Proposers in `olmix/fit/utils.py`

Remove all cookbook/manual path handling from proposers. Update signatures:

```python
def propose(
    self,
    predictor: list,
    prior_distributions: dict[str, float],
    original_prior: dict[str, float],
    opt_avg_metric: bool = False,
    constrain_objective: bool = False,
    swarm_config: ExperimentConfig | None = None,  # Only this for constraints
    ...
):
    if constrain_objective:
        if swarm_config is None:
            raise ValueError("swarm_config required for constrain_objective")
        desired_tokens, available_tokens_per_source, repetition_factor = \
            compute_constraints_from_config(swarm_config)
        ...
```

### 5. Update Example Config

```yaml
# configs/examples/test_olmo2_30m.yaml
name: olmix-test-30m
proxy_model_id: olmo2_30m
chinchilla_multiple: 0.5
sources:
  - name: dclm
    paths: [...]

# Final run constraints
target_model_id: olmo2_7b
target_chinchilla_multiple: 20.0
repetition_factor: 5.0
```

## Files to Modify

| File | Changes |
|------|---------|
| `olmix/aliases.py` | Add constraint fields and `get_target_tokens()` |
| `olmix/fit/cli.py` | Remove cookbook/manual options, use config for constraints |
| `olmix/fit/utils.py` | Add `compute_constraints_from_config()`, simplify proposers |
| `configs/examples/test_olmo2_30m.yaml` | Add constraint settings |

## Usage

```yaml
# experiment.yaml
name: my-swarm
proxy_model_id: olmo2_30m
chinchilla_multiple: 0.5
sources:
  - name: dclm
    paths: ["s3://..."]

target_model_id: olmo2_7b
target_chinchilla_multiple: 20.0
repetition_factor: 5.0
```

```bash
olmix fit fit -c experiment.yaml --constrain-objective
```

## Verification

### 1. Verify Token Counting from Paths

Run `olmix launch preview` to verify total available tokens are calculated correctly:

```bash
olmix launch preview --config configs/examples/test_olmo2_30m.yaml
```

Expected output should show:
- Total tokens per source (calculated from paths)
- Prior distribution (proportional to token counts)

Compare against manual check:
```python
# Quick script to verify token counts
from olmix.launch.synthesize_mixture import calculate_priors
from olmix.aliases import ExperimentConfig

config = ExperimentConfig.from_yaml("configs/examples/test_olmo2_30m.yaml")
priors, total_tokens, token_counts = calculate_priors(config.sources, config.dtype, use_cache=False)
print(f"Total tokens: {total_tokens:,}")
for source, count in token_counts.items():
    print(f"  {source}: {count:,} tokens")
```

### 2. Verify Swarm Trains Correct Token Count

Launch a test swarm and verify training tokens:

```bash
olmix launch run --config configs/examples/test_olmo2_30m.yaml --dry-run
```

Check the output shows:
- `chinchilla_multiple: 0.5` (from config)
- Training tokens = 0.5 × 20 × 30M = 300M tokens
- This should be << total available tokens

After actual launch, verify in WandB:
- `throughput/total tokens` at end of run matches expected
- Run completes without "insufficient tokens" errors

### 3. Verify Constraint Optimization Uses Correct Tokens

Run fit with constraints:

```bash
olmix fit fit -g <GROUP_ID> -c configs/examples/test_olmo2_30m.yaml -w ai2-llm/olmix -G training_loss --constrain-objective
```

Verify in output:
- `target_tokens` matches: 20 × 20 × 7B = 2.8T tokens (from config's `target_chinchilla_multiple` and `target_model_id`)
- `available_tokens_per_source` matches the counts from step 1
- Proposed mixture respects constraint: `weight[source] × target_tokens ≤ available[source] × repetition_factor`

### 4. Pre-check: Verify Available Tokens Support Test Sizes

Before creating test configs, verify the data paths have enough tokens:

```bash
# Check available tokens from sources
olmix launch preview --config configs/examples/test_olmo2_30m.yaml
```

The output will show total available tokens per source. Ensure:
- Total tokens > 1.4B (largest test config needs 1.4B tokens)
- If not enough tokens, either:
  - Use smaller chinchilla_multiples
  - Add more data paths to sources

Example check (run this Python snippet):
```python
from olmix.launch.synthesize_mixture import calculate_priors
from olmix.aliases import ExperimentConfig

config = ExperimentConfig.from_yaml("configs/examples/test_olmo2_30m.yaml")
priors, total_tokens, _ = calculate_priors(config.sources, config.dtype, use_cache=True)
print(f"Total available tokens: {total_tokens:,}")

# Check if we have enough for the largest test
required_tokens = 5.0 * 20 * 14_000_000  # 1.4B
if total_tokens < required_tokens:
    print(f"WARNING: Not enough tokens! Need {required_tokens:,} but only have {total_tokens:,}")
    print(f"Maximum safe chinchilla_multiple: {total_tokens / (20 * 14_000_000):.2f}")
else:
    print(f"OK: Have {total_tokens / required_tokens:.1f}x the required tokens")
```

### 5. Create Verification Configs with Varying Token Amounts

Use the **14M model** for fast verification. **Note:** Only create configs with token amounts ≤ available tokens from step 4.

**Config A: `configs/test/verify_tokens_140m.yaml`** (fastest)
```yaml
name: verify-tokens-140m
proxy_model_id: olmo3_14m    # Use 14M model for fast tests
chinchilla_multiple: 0.5     # 0.5 × 20 × 14M = 140M tokens
nodes: 1
gpus: 1
global_batch_size: 16
variants: 1
seed: 42
# ... same sources as test_olmo2_30m.yaml
```

**Config B: `configs/test/verify_tokens_700m.yaml`**
```yaml
name: verify-tokens-700m
proxy_model_id: olmo3_14m
chinchilla_multiple: 2.5     # 2.5 × 20 × 14M = 700M tokens
nodes: 1
gpus: 1
global_batch_size: 16
variants: 1
seed: 42
# ... same sources
```

**Config C: `configs/test/verify_tokens_1400m.yaml`**
```yaml
name: verify-tokens-1400m
proxy_model_id: olmo3_14m
chinchilla_multiple: 5.0     # 5.0 × 20 × 14M = 1.4B tokens
nodes: 1
gpus: 1
global_batch_size: 16
variants: 1
seed: 42
# ... same sources
```

### 6. Launch Verification Runs

Launch each config and verify in WandB:

```bash
# Launch all three
olmix launch run --config configs/test/verify_tokens_140m.yaml
olmix launch run --config configs/test/verify_tokens_700m.yaml
olmix launch run --config configs/test/verify_tokens_1400m.yaml
```

After runs complete, verify in WandB:
| Config | Expected Tokens | Check `throughput/total tokens` |
|--------|-----------------|--------------------------------|
| 140m   | ~140M           | Should be ~140,000,000         |
| 700m   | ~700M           | Should be ~700,000,000         |
| 1400m  | ~1.4B           | Should be ~1,400,000,000       |

If any run shows different token counts, there's a bug in token calculation.

### 7. End-to-End Sanity Check

With test config values:
```yaml
# Swarm settings
proxy_model_id: olmo3_30m      # 30M params
chinchilla_multiple: 0.5       # Train on 300M tokens

# Constraint settings
target_model_id: olmo3_7b      # 7B params
target_chinchilla_multiple: 20 # Need 2.8T tokens
repetition_factor: 5.0         # Can repeat data 5x
```

Verify the math:
1. Swarm trains: 0.5 × 20 × 30M = 300M tokens
2. Target needs: 20 × 20 × 7B = 2.8T tokens
3. With 5x repetition, need at least 2.8T / 5 = 560B actual tokens
4. If sources have < 560B tokens, constraint will limit proposed mixture weights
