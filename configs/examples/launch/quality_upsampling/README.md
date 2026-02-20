# Quality Upsampling Experiments

Tests how assigning different weights to quality buckets (rather than just including/excluding them) affects BPB metrics across different topic distributions.

## Experiment Design

- **Model**: olmo3_14m (14M parameters)
- **Training**: 0.5x Chinchilla (140M tokens)
- **Quality buckets**: vigintiles (5% buckets) from DCLM quality classifier, grouped by percentile range
- **Weighting schemes**: gradual, aggressive

### Quality Weight Schemes

| Scheme | top10pct | 10-30pct | 30-50pct |
|--------|----------|----------|----------|
| **gradual** | 50% | 30% | 20% |
| **aggressive** | 70% | 30% | - |

Vigintile groupings:
- **top10pct**: vigintile_0018, vigintile_0020
- **10-30pct**: vigintile_0015, vigintile_0016, vigintile_0017
- **30-50pct**: vigintile_0011, vigintile_0012, vigintile_0013, vigintile_0014

Note: vigintile_0019 doesn't exist in the data.

### Topic Distributions

| Distribution | Emphasis |
|--------------|----------|
| heavy_adult | 50% adult content weight |
| heavy_code | 50% software_development weight |
| heavy_science | 50% science_math_and_technology weight |
| heavy_wiki | Higher wikipedia weight (30%) |

## Config Structure

Quality buckets are grouped under custom names with multiple paths:

```yaml
quality:
  - name: top10pct
    weight: 70
    paths:
      - "s3://.../vigintile_0018/*.npy"
      - "s3://.../vigintile_0020/*.npy"
  - name: 10-30pct
    weight: 30
    paths:
      - "s3://.../vigintile_0015/*.npy"
      - "s3://.../vigintile_0016/*.npy"
      - "s3://.../vigintile_0017/*.npy"
```

Weights are normalized within each topic: 70 + 30 = 100 → top10pct gets 70%, 10-30pct gets 30%.

## Comparison with quality_thresholds

The `quality_thresholds` experiments use binary inclusion/exclusion:
- **top10pct**: Only include vigintiles 0018, 0020
- **top30pct**: Include vigintiles 0015-0020
- **top50pct**: Include vigintiles 0011-0020
- **top70pct**: Include vigintiles 0007-0020

The `quality_upsampling` experiments use weighted inclusion:
- Quality buckets are grouped by percentile range
- Each group gets a specified weight (e.g., 70% for top10pct, 30% for 10-30pct)
- Allows "soft" quality preferences rather than hard cutoffs

## Running These Experiments

```bash
# Single experiment
olmix launch run --config configs/experiments/quality_upsampling/heavy_code/gradual.yaml

# All quality upsampling experiments
./configs/experiments/quality_upsampling/batch_run.sh
```

## Implementation Notes

The quality bucket weighting is implemented via the `weight` field on `QualityConfig` in `olmix/aliases.py`. When quality buckets have explicit weights:
- Weights are normalized to sum to 1 within each topic
- Topic weight is then distributed according to these normalized quality weights
- If no weights are specified, the default behavior (proportional to token count) is used
