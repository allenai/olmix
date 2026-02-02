# Plan: Reorganize Test Configs into Experiment Suites

## Goal

Restructure `configs/test/` into well-documented experiment suites that test:
1. **Training duration**: How performance changes with more training (fixed proportions)
2. **Data proportions**: How performance changes with different mixes (fixed training)

## Current State

```
configs/test/
├── verify_tokens_140m.yaml   # 0.5x Chinchilla (~8 min)
├── verify_tokens_700m.yaml   # 2.5x Chinchilla (~35 min)
└── verify_tokens_1400m.yaml  # 5.0x Chinchilla (~70 min)
```

All use same data mix:
- dclm:science_math_and_technology
- dclm:software_development
- dclm:education_and_jobs
- wikipedia
- arxiv

## Proposed Structure

```
configs/experiments/
├── README.md                      # Documentation with results
├── training_duration/             # Experiment 1: vary training length
│   ├── duration_0.5x.yaml        # 140M tokens, ~8 min
│   ├── duration_2.5x.yaml        # 700M tokens, ~35 min
│   └── duration_5.0x.yaml        # 1.4B tokens, ~70 min
└── data_proportions/              # Experiment 2: vary data mix
    ├── mix_baseline.yaml         # Current balanced mix
    ├── mix_heavy_code.yaml       # 50% code emphasis
    ├── mix_heavy_science.yaml    # 50% science emphasis
    └── mix_heavy_wiki.yaml       # 50% wikipedia emphasis
```

## Implementation

### Phase 1: Rename and Reorganize

1. Rename `configs/test/` → `configs/experiments/`
2. Create subdirectory `training_duration/`
3. Move and rename configs:
   - `verify_tokens_140m.yaml` → `training_duration/duration_0.5x.yaml`
   - `verify_tokens_700m.yaml` → `training_duration/duration_2.5x.yaml`
   - `verify_tokens_1400m.yaml` → `training_duration/duration_5.0x.yaml`

### Phase 2: Create Proportion Experiment Configs

All use `chinchilla_multiple: 0.5` (140M tokens, ~8 min each)

**mix_baseline.yaml**: Current balanced mix (control)
- Same proportions as training_duration configs

**mix_heavy_code.yaml**: Code-heavy mix
- dclm:software_development: 50%
- dclm:science_math_and_technology: 20%
- dclm:education_and_jobs: 10%
- wikipedia: 10%
- arxiv: 10%

**mix_heavy_science.yaml**: Science-heavy mix
- dclm:science_math_and_technology: 50%
- dclm:software_development: 10%
- dclm:education_and_jobs: 10%
- wikipedia: 10%
- arxiv: 20%

**mix_heavy_wiki.yaml**: Wikipedia-heavy mix
- wikipedia: 50%
- dclm:science_math_and_technology: 15%
- dclm:software_development: 15%
- dclm:education_and_jobs: 10%
- arxiv: 10%

### Phase 3: Create README.md

Include:
- Purpose of each experiment suite
- How to run experiments
- Results from completed runs (training_duration)
- Placeholder for proportion results

## Files to Create/Modify

| Action | Path |
|--------|------|
| Rename dir | `configs/test/` → `configs/experiments/` |
| Create dir | `configs/experiments/training_duration/` |
| Create dir | `configs/experiments/data_proportions/` |
| Move+rename | `verify_tokens_140m.yaml` → `training_duration/duration_0.5x.yaml` |
| Move+rename | `verify_tokens_700m.yaml` → `training_duration/duration_2.5x.yaml` |
| Move+rename | `verify_tokens_1400m.yaml` → `training_duration/duration_5.0x.yaml` |
| Create | `configs/experiments/README.md` |
| Create | `data_proportions/mix_baseline.yaml` |
| Create | `data_proportions/mix_heavy_code.yaml` |
| Create | `data_proportions/mix_heavy_science.yaml` |
| Create | `data_proportions/mix_heavy_wiki.yaml` |

## Verification

1. Verify directory structure created correctly
2. Verify configs are valid YAML
3. Test launch: `olmix launch run --config configs/experiments/training_duration/duration_0.5x.yaml`
4. Run one proportion experiment to verify it works

## Summary

This reorganization:
- Renames `configs/test/` → `configs/experiments/` with clear subdirectories
- Documents existing training duration results in README
- Adds 4 new proportion experiment configs for testing data mix effects
- Total: 7 experiment configs organized into 2 logical suites

## Status

**Completed** - All files created and validated.
