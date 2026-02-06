# Plan: Test Launch 2 Mixture Jobs on Beaker

## Goal

Run a minimal test of the olmix launch pipeline with 2 mixture variants on `ai2/jupiter` cluster with `high` priority.

## Changes Required

### 1. Update Test Config

**File:** `configs/examples/test_olmo2_30m.yaml`

Changes:
```yaml
cluster: ai2/saturn-cirrascale  →  cluster: ai2/jupiter
budget: ai2/oe-data             →  budget: ai2/oe-base
workspace: ai2/dolma2           →  workspace: ai2/oe-data
# Add:
priority: high
```

Current config already has `variants: 2` which is correct.

### 2. Run Dry-Run

```bash
olmix launch run --config configs/examples/test_olmo2_30m.yaml --dry-run
```

Answer `y` to both prompts. This will:
- Sample 2 mixtures
- Build Beaker experiment specs
- Print the specs without actually launching

### 3. Launch for Real

```bash
olmix launch run --config configs/examples/test_olmo2_30m.yaml
```

Answer `y` to both prompts. This will submit 2 jobs to Beaker.

## Verification

After launch:
```bash
# Check job status
olmix launch status --config configs/examples/test_olmo2_30m.yaml --group-id <GROUP_ID>

# Or use beaker CLI directly
beaker job list --cluster ai2/jupiter | grep olmix-test-30m
```

## Files to Modify

| File | Change |
|------|--------|
| `configs/examples/test_olmo2_30m.yaml` | Set `cluster: ai2/jupiter`, `workspace: ai2/oe-data`, `budget: ai2/oe-base`, add `priority: high` |
