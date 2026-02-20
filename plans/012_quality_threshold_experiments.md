# Plan: Quality Threshold Experiment System

## Goal

Create a system to generate experiment configs that test how data quality thresholds affect model performance. Quality thresholds filter data to only include top N% quality data (e.g., threshold 0.3 = top 30%).

## Key Design Decisions

1. **Quality buckets are optional** - Existing configs without quality_buckets continue to work (backward compatible)
2. **Nested structure** - quality_buckets nest under topics (or directly under sources if no topics)
3. **Threshold semantics** - 0.3 means include buckets with threshold ≤ 0.3 (top 30%)
4. **Script-based generation** - Standalone script generates YAML configs, no core olmix changes required for basic use

## Implementation

### Phase 1: Schema Changes (`olmix/aliases.py`)

Add `QualityConfig` class (insert after TopicConfig ~line 88):

```python
class QualityConfig(BaseModel):
    """Configuration for a quality level within a topic or source.

    Name can be any string like "vigintile_0001", "high", "medium", "low", etc.
    """
    name: str  # e.g., "vigintile_0001", "high", "low"
    paths: list[str]
    max_repetition_factor: float = 1.0
```

Update `TopicConfig` to support optional quality:
- Add field: `quality: list[QualityConfig] | None = None`
- Make `paths` optional: `paths: list[str] | None = None`
- Add validator: paths XOR quality (exactly one must be provided)

Update `SourceConfig` similarly:
- Add field: `quality: list[QualityConfig] | None = None`
- Validator: exactly one of paths/topics/quality

### Phase 2: Update Path Resolution (`olmix/launch/synthesize_mixture.py`)

Update `get_leaf_configs()` function to handle quality nesting:
- source.topics[].quality[].paths → `(source:topic:quality_name, paths)`
- source.quality[].paths → `(source:quality_name, paths)`

### Phase 3: Create Experiment Configs

Manually create `config/examples/launch/quality_thresholds/`:
- `quality_top10pct.yaml` - top 10% quality (vigintile_0019, vigintile_0020)
- `quality_top30pct.yaml` - top 30% quality (vigintile_0015 - vigintile_0020)
- `quality_top50pct.yaml` - top 50% quality (vigintile_0011 - vigintile_0020)
- `quality_top70pct.yaml` - top 70% quality (vigintile_0007 - vigintile_0020)
- `quality_top100pct.yaml` - all data (vigintile_0001 - vigintile_0020)

**Vigintile structure (20 buckets, 5% each):**
- `vigintile_0020` = top 5% (best quality)
- `vigintile_0019` = next 5%
- ...
- `vigintile_0001` = bottom 5% (worst quality)

**Mapping threshold to vigintiles:**
- top 10% → vigintile_0019, vigintile_0020
- top 30% → vigintile_0015 through vigintile_0020
- top 50% → vigintile_0011 through vigintile_0020
- top 70% → vigintile_0007 through vigintile_0020
- top 100% → vigintile_0001 through vigintile_0020

**Base path (real all_dressed data with vigintiles):**
```
s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/{topic}/vigintile_{0001-0020}/*.npy
```

**Topics with quality buckets (all_dressed source):**
- science_math_and_technology
- software_development
- education_and_jobs
- news_and_politics
- adult_content
- art_and_design

**Sources without quality (for backward compatibility testing):**
- arxiv - direct paths, no topics or quality
- wikipedia - direct paths, no topics or quality

All configs use:
- 0.5x Chinchilla (140M tokens, ~8 min)
- Equal proportions across sources/topics
- 14M model on Jupiter cluster

## Example Config (`quality_top30pct.yaml`)

```yaml
name: quality-top30pct
description: Quality threshold experiment - top 30% quality data
chinchilla_multiple: 0.5
proxy_model_id: olmo3_14m
cluster: ai2/jupiter

sources:
  # Source with topics and quality levels
  - name: all_dressed
    topics:
      - name: science_math_and_technology
        quality:
          - name: vigintile_0015
            paths: ["s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0015/*.npy"]
          - name: vigintile_0016
            paths: ["s3://ai2-llm/preprocessed/cc_all_dressed/all_dressed_v3/dclm_plus2_vigilantes/allenai/dolma2-tokenizer/science_math_and_technology/vigintile_0016/*.npy"]
          # ... vigintile_0017 through vigintile_0020 (top 30%)
      # ... other topics

  # Source without topics or quality (backward compatibility)
  - name: arxiv
    paths:
      - "s3://ai2-llm/preprocessed/proof-pile-2/v0_decontaminated-0625_tokenized/arxiv/train/allenai/dolma2-tokenizer/*.npy"
    max_repetition_factor: 1.5

  - name: wikipedia
    paths:
      - "s3://ai2-llm/preprocessed/wikipedia-dolma-0823/allenai/dolma2-tokenizer/*.npy"
    max_repetition_factor: 2.0
```

## Files Created/Modified

| Action | File |
|--------|------|
| Create | `plans/012_quality_threshold_experiments.md` - This plan |
| Modify | `olmix/aliases.py` - Add QualityConfig, update TopicConfig/SourceConfig |
| Modify | `olmix/launch/synthesize_mixture.py` - Update get_leaf_configs() |
| Create | `config/examples/launch/quality_thresholds/quality_top10pct.yaml` |
| Create | `config/examples/launch/quality_thresholds/quality_top30pct.yaml` |
| Create | `config/examples/launch/quality_thresholds/quality_top50pct.yaml` |
| Create | `config/examples/launch/quality_thresholds/quality_top70pct.yaml` |
| Create | `config/examples/launch/quality_thresholds/quality_top100pct.yaml` |

## Verification

1. **Schema validation:**
   ```bash
   python -c "from olmix.aliases import QualityConfig, TopicConfig; print('OK')"
   ```

2. **Config validation:**
   ```bash
   for f in config/examples/launch/quality_thresholds/quality_*.yaml; do
       python -c "import yaml; yaml.safe_load(open('$f'))" && echo "$f: valid"
   done
   ```

3. **Dry run test:**
   ```bash
   olmix launch run --config config/examples/launch/quality_thresholds/quality_top30pct.yaml --dry-run
   ```

4. **Launch one experiment:**
   ```bash
   yes | olmix launch run --config config/examples/launch/quality_thresholds/quality_top30pct.yaml
   ```

## Summary

- Extends olmix config schema with optional `quality` nesting under topics/sources
- `QualityConfig` uses `name` field (supports strings like "vigintile_0001", "high", "low")
- Manually creates 5 experiment configs: `quality_top10pct.yaml` through `quality_top100pct.yaml`
- Vigintiles (20 buckets named `vigintile_0001` - `vigintile_0020`, 5% each)
- Includes arxiv and wikipedia sources without quality for backward compatibility
- All experiments use 0.5x Chinchilla (~8 min each) on Jupiter cluster
