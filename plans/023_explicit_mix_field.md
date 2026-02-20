# Plan 023: Require explicit `mix` field on all LaunchConfigs

## Goal

Standardize all LaunchConfigs to use an explicit top-level `mix` field for weights and repetition factors, instead of embedding them in the `data.sources` hierarchy. The `data` section describes *what data exists* (names + paths); the `mix` section describes *how to use it* (weights + repetition factors).

## Context

Previously, hand-written configs embedded weights at various levels of the source hierarchy (`SourceConfig.weight`, `TopicConfig.weight`, `QualityConfig.weight`). This was fragile — `SourceConfig` didn't even have a `weight` field, so source-level weights were silently dropped by Pydantic. Generated configs from `olmix generate` already used an explicit `mix` field.

Similarly, `max_repetition_factor` lived on `SourceConfig`, `TopicConfig`, and `QualityConfig` in the `data` section, duplicating the `repetition_factor` in `mix`. The launch code only ever read from `mix`.

## Changes made

### 1. Added `weight` to `SourceConfig` (`olmix/aliases.py`)

Added `weight: float | None = None` so source-level weights can be parsed (needed during migration).

### 2. Removed `max_repetition_factor` from data models (`olmix/aliases.py`)

Removed `max_repetition_factor` from `QualityConfig`, `TopicConfig`, and `SourceConfig`. This field was vestigial — the launch code (`olmix/launch/utils.py`) only reads `repetition_factor` from `MixEntry` in the `mix` field. All repetition config now lives exclusively in `mix`.

### 3. Added nested mix support (`olmix/aliases.py`)

Added `flatten_mix()` and `_flatten_mix_node()` to support hierarchical mix definitions. A `model_validator(mode="before")` on `LaunchConfig` auto-flattens nested mixes at parse time. This allows two equivalent formats:

**Nested** (recommended for hand-written configs):
```yaml
mix:
  dclm:
    weight: 0.8
    repetition_factor: 1.0
    software_development:
      weight: 0.625
      repetition_factor: 1.0
    science_math_and_technology:
      weight: 0.25
      repetition_factor: 1.0
```

**Flat** (used by `olmix generate`):
```yaml
mix:
  dclm:software_development:
    weight: 0.5
    repetition_factor: 1.0
  dclm:science_math_and_technology:
    weight: 0.2
    repetition_factor: 1.0
```

Nested weights are multiplicative (leaf weight = product of weights along root-to-leaf path). `repetition_factor` is inherited from the nearest ancestor that sets it.

### 4. Updated CLI (`olmix/cli.py`)

`_load_launch_configs()` now accepts both a single YAML file and a directory of YAML files for `--variants`.

### 5. Migrated all 31 configs (`config/examples/launch/`)

- Removed embedded weights from `data.sources` (source, topic, quality levels)
- Added explicit `mix` section with normalized weights (sum to 1.0) in nested format
- Added explicit `repetition_factor` on every leaf node for consistency
- Removed all `max_repetition_factor` from `data.sources`

**Config patterns:**
- `data_proportions/` (4 files) — 2-level: source > topic
- `quality_upsampling/` (8 files) — 3-level: source > topic > quality (best/high/med/low)
- `quality_thresholds/` (16 files) — 3-level: source > topic > vigintile buckets
- `training_duration/` (3 files) — 2-level: source > topic (equal weights)

### 6. Updated tests

- `tests/test_config.py` — Tests for `SourceConfig.weight`, nested mix (7 tests: flat passthrough, 2-level, 3-level, repetition_factor inheritance, default weights, programmatic construction, YAML loading), hand-written config loading
- `tests/test_cli.py` — Tests for single-file and directory loading

### 7. Updated README

Documented both nested and flat mix formats, updated CLI examples.

### 8. Migration scripts (one-time, in `scripts/`)

- `scripts/migrate_configs.py` — Initial migration: walk source tree, compute flat mix from product weights, remove nested weight fields
- `scripts/normalize_mix_weights.py` — Normalize mix weights to sum to 1.0
- `scripts/nest_mix_weights.py` — Convert flat colon-separated mix to nested format with relative weights

## How `repetition_factor` flows through the system

| Context | Field | Meaning |
|---------|-------|---------|
| `mix` (launch) | `repetition_factor: 2.0` | Per-source cap passed to OLMo Core as `max_repetition_ratio` — how many times this source's tokens can be repeated during training |
| `constraints` (fit) | `repetition_factor: 5.0` | Global scalar used by the optimizer — reject any proposed mix where *any* source would need >Nx repetition at the target token budget |

The launch path: `LaunchConfig.mix` → `MixEntry.repetition_factor` → `SourceInstance.repetition_factor` → `SourceMixtureConfig(max_repetition_ratio=...)` in OLMo Core.

## Files changed

| File | Change |
|------|--------|
| `olmix/aliases.py` | Added `SourceConfig.weight`, removed `max_repetition_factor` from 3 models, added `flatten_mix()` + `_flatten_mix_node()` + model validator |
| `olmix/cli.py` | `_load_launch_configs()` accepts file or dir |
| `config/examples/launch/**/*.yaml` (31 files) | Explicit nested `mix`, no embedded weights, no `max_repetition_factor` |
| `tests/test_config.py` | New tests for weight, nested mix, config loading |
| `tests/test_cli.py` | New tests for single-file loading |
| `README.md` | Mix format documentation |
| `scripts/migrate_configs.py` | One-time migration script |
| `scripts/normalize_mix_weights.py` | One-time normalization script |
| `scripts/nest_mix_weights.py` | One-time nesting script |

## Verification

- `make run-checks`: format, lint, pyright (0 errors), pytest (111 passed) all green
- `olmix launch preview` works on all 4 config patterns (single file and directory mode)
- All 31 configs parse as valid `LaunchConfig` with non-null `mix` summing to ~1.0
