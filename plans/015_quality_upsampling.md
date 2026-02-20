# Plan: Create quality_upsampling experiments with weighted quality buckets

## Context

**Current state:**
- `quality_thresholds/` experiments include/exclude vigintiles but don't assign weights to them
- `QualityConfig` in `olmix/aliases.py` has no `weight` field
- Mix generation distributes topic weights proportionally across included quality buckets by token count

**User request:**
- Create `experiments/quality_upsampling/` with weighted quality buckets
- Same topic distributions: heavy_adult, heavy_code, heavy_science, heavy_wiki
- Two weighting schemes instead of four threshold configs:
  1. **gradual**: top10%=50, 10-30%=30, 30-50%=20, 50-70%=0
  2. **aggressive**: top10%=70, 10-30%=30, rest=0

**Vigintile mapping** (vigintile_0019 doesn't exist in data):
| Percentile Range | Vigintiles | gradual Weight | aggressive Weight |
|------------------|------------|----------------|-------------------|
| top 10% | 0018, 0020 | 50 | 70 |
| 10-30% | 0015, 0016, 0017 | 30 | 30 |
| 30-50% | 0011, 0012, 0013, 0014 | 20 | 0 |
| 50-70% | 0007, 0008, 0009, 0010 | 0 | 0 |

**Note:** All vigintiles are explicitly included in configs (even with weight 0) for clarity.

---

## Implementation

### 0. Save plan to `plans/015_quality_upsampling.md`

Copy this plan to the repository's plans directory for documentation.

### 1. Add `weight` field to `QualityConfig` in `olmix/aliases.py`

```python
class QualityConfig(BaseModel):
    name: str
    paths: list[str]
    weight: float | None = None  # NEW: Optional weight for quality bucket
    max_repetition_factor: float = 1.0
```

### 2. Update mix generation in `olmix/launch/synthesize_mixture.py`

Modify the weight calculation to use quality bucket weights when provided:
- If `quality.weight` is set, use it for weighting within the topic
- If not set, fall back to proportional distribution by token count (current behavior)

### 3. Create directory structure

```
config/examples/launch/quality_upsampling/
├── README.md
├── batch_run.sh
├── heavy_adult/
│   ├── gradual.yaml
│   └── aggressive.yaml
├── heavy_code/
│   ├── gradual.yaml
│   └── aggressive.yaml
├── heavy_science/
│   ├── gradual.yaml
│   └── aggressive.yaml
└── heavy_wiki/
    ├── gradual.yaml
    └── aggressive.yaml
```

### 4. Config file structure (example: heavy_code/aggressive.yaml)

Quality buckets are grouped by percentile range with custom names and multiple paths:

```yaml
name: quality-upsampling-aggressive-heavy-code
description: Quality upsampling experiment - aggressive weighting with heavy code mix

# ... standard settings (same as quality_thresholds) ...

sources:
  - name: all_dressed
    weight: 98
    topics:
      - name: software_development
        weight: 50
        quality:
          - name: top10pct
            weight: 70
            paths:
              - "s3://.../software_development/vigintile_0018/*.npy"
              - "s3://.../software_development/vigintile_0020/*.npy"
          - name: 10-30pct
            weight: 30
            paths:
              - "s3://.../software_development/vigintile_0015/*.npy"
              - "s3://.../software_development/vigintile_0016/*.npy"
              - "s3://.../software_development/vigintile_0017/*.npy"
      # ... other topics with same quality weight pattern
```

Weights are normalized within each topic: 70 + 30 = 100 → top10pct gets 70%, 10-30pct gets 30%.

---

## Files to Modify

| File | Change |
|------|--------|
| `olmix/aliases.py` | Add `weight: float \| None = None` to `QualityConfig` |
| `olmix/launch/synthesize_mixture.py` | Use quality weights in mix calculation when provided |
| `config/examples/launch/quality_upsampling/` | Create 8 new config files + README + batch_run.sh |

---

## Verification

1. **Run tests:** `pytest tests/ -v`
2. **Dry-run a config:**
   ```bash
   olmix launch run --config config/examples/launch/quality_upsampling/heavy_code/gradual.yaml --dry-run
   ```
3. **Verify mix output shows different weights for quality buckets**

---

## Notes

- The `weight` field on quality buckets is optional for backwards compatibility
- Weights within a topic are normalized (e.g., 70 + 30 = 100 → 70% and 30%)
- Quality buckets can have custom names (e.g., `top10pct`) and multiple paths
- Vigintiles with weight 0 are simply omitted from configs (cleaner than explicit weight: 0)
