# Plan: Adopt OLMo-core Chinchilla Training Strategy

## Overview

Switch olmix from its current custom scaling laws to use OLMo-core's `WSDSChinchillaRunConfigurator` strategy, while keeping YAML-based data configuration.

**Goals:**
1. Use Chinchilla-based hyperparameter calculations (batch size, LR, duration)
2. Switch to WSDS scheduler (Warmup-Stable-Decay-Stable)
3. Use sequence length 8192
4. Keep YAML-style data organization
5. Replace `max_tokens` entirely with `chinchilla_multiple`
6. Apply Chinchilla calculations to anneal mode (remove special-casing)

---

## Configuration Comparison

### Model Architecture

| Aspect | olmix (current) | OLMo-core code_ablations | Planned |
|--------|-----------------|--------------------------|---------|
| Model variants | `olmo2_1M` - `olmo2_7B_v2` | `olmo3_7B`, `olmo2_1B_v2` | Keep all olmix variants |
| Tokenizer | `dolma2` | `dolma2` | `dolma2` (no change) |
| Sequence length | 2048, 4096, 8192 | Fixed 8192 | **Fixed 8192** |

### Training Strategy

| Aspect | olmix (current) | OLMo-core Chinchilla | Planned |
|--------|-----------------|----------------------|---------|
| LR formula | `0.0047 * (N/108M)^(-1/3)` | `0.0047 * (N/108M)^(-1/3) / 2` | **Halve LR** |
| LR scheduler | `CosWithWarmupAndLinearDecay` | `WSDS` (multi-period) | **WSDS** |
| Warmup | `params / (batch * seq_len)` steps | `num_params` tokens | **num_params tokens** |
| Duration | Fixed `max_tokens` from YAML | `20 * params * chinchilla_multiple` | **Chinchilla-based** |
| Anneal mode | Special-cased (fixed LR, batch) | N/A | **Remove special-casing** |

### Optimizer

| Aspect | olmix (current) | OLMo-core Chinchilla | Planned |
|--------|-----------------|----------------------|---------|
| Optimizer | `SkipStepAdamWConfig` | `SkipStepAdamWConfig` | No change |
| Weight decay | `0.033` | `0.1` | **0.1** |
| Beta1 | `0.9` | `0.9` | No change |
| Beta2 | Fixed `0.95` | Adaptive: `0.95` or `0.99` | **Adaptive** |
| Embedding WD | `0.0` | `0.0` | No change |

### Batch Size

| Aspect | olmix (current) | OLMo-core Chinchilla | Planned |
|--------|-----------------|----------------------|---------|
| Formula | `2048 * 160 * (N/108M)^(2/3)` | `2048 * 160 * (N/108M)^(2/3)` | Same formula |
| Seq adjustment | `/= seq_len // 2048` | None (assumes 2048) | **Remove adjustment** |
| Rounding | Power of 2 | Round to nearest | **Keep power of 2** |

### Data Configuration

| Aspect | olmix (current) | OLMo-core code_ablations | Planned |
|--------|-----------------|--------------------------|---------|
| Config format | YAML files | Python scripts | **Keep YAML** |
| Source mixing | `SourceMixtureConfig` | `SourceMixtureConfig` | No change |
| Source repetition | `max_repetition_ratio` | `max_repetition_ratio` | No change |
| Instance filter | None | `InstanceFilterConfig` | **Add support** |
| Num workers | 16 | 4 | **4** |

### Repetition Handling (Two Separate Mechanisms)

**1. Source-Level Repetition (`max_repetition_ratio`)**

Both repos use `SourceMixtureConfig.max_repetition_ratio` to control **data upsampling**:

```python
# In SourceMixtureConfig (olmo_core/data/source_mixture.py)
max_repetition_ratio: float = 1.0
# 1.0 = no repetition (each token seen once)
# 5.0 = data can be repeated up to 5x (each token seen up to 5 times)
```

This is used when a source has fewer tokens than needed for its target ratio. Setting `max_repetition_ratio=5` allows the source to be repeated up to 5 times to meet the target.

**How it works** (lines 311-315 in source_mixture.py):
```python
max_for_source = int(
    (num_for_source * source_config.max_source_fraction)
    * source_config.max_repetition_ratio
)
# If max_for_source < needed_for_source, raises error
```

**2. Instance-Level Repetition Filter (`InstanceFilterConfig`)** - OLMo-core only

OLMo-core has an additional filter that removes **individual sequences** containing repetitive token patterns:

```python
# In olmo_core/data/composable/data_loader.py
@dataclass
class InstanceFilterConfig(Config):
    repetition_max_period: int = 13   # Max period length to check
    repetition_min_period: int = 1    # Min period length to check
    repetition_max_count: int = 32    # Max allowed repetitions

# Usage in code_ablations:
instance_filter_config=InstanceFilterConfig(
    repetition_min_period=1,
    repetition_max_period=13,
    repetition_max_count=32,
)
```

This filters out sequences like `"the the the the..."` or `"123 123 123..."` that have repeating patterns. It's a **quality filter** applied at the instance level, not a data repetition control.

| Mechanism | Purpose | Level | olmix | OLMo-core |
|-----------|---------|-------|-------|-----------|
| `max_repetition_ratio` | Allow upsampling small sources | Source | Yes | Yes |
| `InstanceFilterConfig` | Remove repetitive sequences | Instance | **No** | Yes |

### Checkpointing

| Aspect | olmix (current) | OLMo-core Chinchilla | Planned |
|--------|-----------------|----------------------|---------|
| Ephemeral interval | 100 steps | 250 steps | **250 steps** |
| Save interval | 1000 steps | Chinchilla periods | **Chinchilla periods** |

---

## Implementation Plan

### Phase 1: Add Chinchilla Multiple to YAML Config

Update `olmix/aliases.py` `ExperimentConfig`:

```yaml
# Example YAML config
chinchilla_multiple: 5  # New field - trains for 5x Chinchilla optimal
# max_tokens removed - now calculated from chinchilla_multiple
```

### Phase 2: Update TransformerConfigBuilder

**File:** `olmix/model/transformer.py`

Changes:
1. Remove `sequence_length` parameter - fix at 8192
2. Remove `max_tokens` parameter - replaced by `chinchilla_multiple`
3. Add `chinchilla_multiple` parameter
4. Replace custom scaling laws with Chinchilla formulas
5. Replace scheduler with WSDS
6. Update optimizer settings (weight_decay, adaptive beta2)
7. Remove `TrainType.anneal` special-casing - use Chinchilla for all training

```python
from olmo_core.optim import WSDS, SchedulerUnits

# Constants
SEQUENCE_LENGTH = 8192
TOKENS_PER_PARAM = 20

class TransformerConfigBuilder:
    def __init__(
        self,
        ...
        chinchilla_multiple: float = 1.0,  # New parameter
        # Remove: sequence_length, max_tokens
    ):
        self.sequence_length = SEQUENCE_LENGTH  # Fixed
        self.chinchilla_multiple = chinchilla_multiple

    def configure_target_batch_size(self, num_params: int) -> int:
        """Chinchilla batch size formula."""
        batch_size = round(2048 * 160 * (num_params / 108_000_000) ** (2 / 3))
        return self.next_power_of_2(batch_size)  # Keep power-of-2 rounding

    def configure_duration(self, num_params: int) -> int:
        """Chinchilla optimal tokens."""
        return int(TOKENS_PER_PARAM * num_params * self.chinchilla_multiple)

    def configure_lr(self, num_params: int) -> float:
        """Chinchilla LR with halving."""
        lr = 0.0047 * (num_params / 108_000_000) ** (-1 / 3)
        return lr / 2.0  # Halve for stability

    def configure_optimizer(self, num_params: int, batch_size: int) -> SkipStepAdamWConfig:
        """Chinchilla optimizer with adaptive beta2."""
        beta2 = 0.95 if batch_size >= 524_288 else 0.99
        return SkipStepAdamWConfig(
            lr=self.configure_lr(num_params),
            weight_decay=0.1,  # Changed from 0.033
            betas=(0.9, beta2),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        )

    def configure_scheduler(self, num_params: int, batch_size: int) -> Scheduler:
        """WSDS scheduler with Chinchilla periods."""
        warmup = num_params  # 1 token per param
        max_tokens = self.configure_duration(num_params)

        # Generate periods: 0.5xC, 1xC, 2xC, ... up to target
        periods = []
        p = 0.5
        while p <= self.chinchilla_multiple:
            periods.append(p)
            p *= 2

        period_lengths = []
        for i, c in enumerate(periods):
            tokens_at_c = int(TOKENS_PER_PARAM * num_params * c)
            tokens_at_c = batch_size * round(tokens_at_c / batch_size)  # Round to batch
            if i == 0:
                period_lengths.append(tokens_at_c)
            else:
                prev_c = periods[i - 1]
                prev_tokens = int(TOKENS_PER_PARAM * num_params * prev_c)
                prev_tokens = batch_size * round(prev_tokens / batch_size)
                period_lengths.append(tokens_at_c - prev_tokens)

        return WSDS(
            units=SchedulerUnits.tokens,
            warmup=warmup,
            decay_fraction=0.1,
            period_lengths=period_lengths,
        )
```

### Phase 3: Update YAML Config Schema

**File:** `olmix/aliases.py`

```python
class InstanceFilterConfig(BaseModel):
    """Config for filtering repetitive sequences."""
    repetition_min_period: int = 1
    repetition_max_period: int = 13
    repetition_max_count: int = 32


class ExperimentConfig(BaseModel):
    # ... existing fields ...

    # Remove:
    # max_tokens: int
    # sequence_length: int

    # Add:
    chinchilla_multiple: float = 1.0  # Default to 1xC (Chinchilla optimal)
    instance_filter: InstanceFilterConfig | None = None  # Optional quality filter
```

**Example YAML with instance filter:**
```yaml
chinchilla_multiple: 5.0
instance_filter:
  repetition_min_period: 1
  repetition_max_period: 13
  repetition_max_count: 32
```

### Phase 4: Update Example Configs

**File:** `configs/examples/test_olmo2_30m.yaml`

```yaml
name: olmix-test-30m
proxy_model_id: olmo2_30m
tokenizer: dolma2

# Training strategy
chinchilla_multiple: 1.0  # 1x Chinchilla optimal
# sequence_length: removed (fixed at 8192)
# max_tokens: removed (calculated from chinchilla_multiple)

nodes: 1
gpus: 1
seed: 42

sources:
  - name: wikipedia
    paths:
      - "s3://ai2-llm/preprocessed/wikipedia-dolma-0823/..."
```

### Phase 5: Update Tests

Add tests for Chinchilla calculations:

```python
def test_chinchilla_batch_size():
    # 30M params should give specific batch size
    builder = TransformerConfigBuilder(...)
    batch_size = builder.configure_target_batch_size(30_000_000)
    assert batch_size == expected_value

def test_chinchilla_duration():
    # 5xC with 30M params = 20 * 30M * 5 = 3B tokens
    builder = TransformerConfigBuilder(chinchilla_multiple=5.0, ...)
    duration = builder.configure_duration(30_000_000)
    assert duration == 3_000_000_000
```

---

## Files to Modify

| File | Action |
|------|--------|
| `olmix/model/transformer.py` | Major update - Chinchilla calculations, WSDS scheduler, instance filter |
| `olmix/aliases.py` | Add `chinchilla_multiple`, `InstanceFilterConfig`; remove `max_tokens`/`sequence_length` |
| `configs/examples/*.yaml` | Update to new schema |
| `tests/test_*.py` | Add Chinchilla calculation tests |

---

## Verification

```bash
# 1. Test Chinchilla calculations
python -c "
from olmix.model.transformer import TransformerConfigBuilder

# Test with 30M model at 5xC
# Expected: batch ~X, duration = 20 * 30M * 5 = 3B tokens
"

# 2. Run tests
pytest tests/ -v

# 3. Test config parsing
olmix launch validate --config configs/examples/test_olmo2_30m.yaml
```

---

## Summary of Changes

| Setting | Current | Planned | Rationale |
|---------|---------|---------|-----------|
| Sequence length | Variable | Fixed 8192 | Match OLMo-core |
| LR | `0.0047 * ...` | `0.0047 * ... / 2` | Empirically better |
| Weight decay | 0.033 | 0.1 | Match OLMo-core |
| Beta2 | Fixed 0.95 | Adaptive | Better for small batches |
| Scheduler | Cosine | WSDS | Multi-period checkpoints |
| Duration | Manual `max_tokens` | `chinchilla_multiple` | More principled |
| Warmup | Steps-based | 1 token/param | Match OLMo-core |
| Num workers | 16 | 4 | Match OLMo-core |
| Anneal mode | Special-cased | Removed | Use Chinchilla uniformly |
| Instance filter | None | Optional `InstanceFilterConfig` | Quality filtering |

---

## Implementation Status

**Completed:** 2025-01-25

All phases implemented:
- Phase 1: Added `chinchilla_multiple` to YAML config
- Phase 2: Updated `TransformerConfigBuilder` with Chinchilla calculations
- Phase 3: Updated YAML config schema with `InstanceFilterConfig`
- Phase 4: Updated example configs
- Phase 5: Added comprehensive tests (34 new tests, all passing)

Additional files updated to support the changes:
- `olmix/cli.py`
- `olmix/launch/train.py`
- `olmix/launch/beaker.py`
- `olmix/launch/synthesize_mixture.py`
- `olmix/fit/utils.py`

### Bug Fixes During Implementation

1. **Small `chinchilla_multiple` handling**: For `chinchilla_multiple < 0.5`, the WSDS scheduler period generation was producing empty period lists. Fixed by adding a fallback to use a single period for the full duration.

2. **Warmup capping**: When `chinchilla_multiple` is very small (e.g., 0.017 for test configs), the default warmup (1 token per param) exceeded the total training duration. Fixed by capping warmup at 5% of total duration.
