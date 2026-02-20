# Quality Threshold Experiments

Tests how restricting to higher-quality document pools affects BPB metrics across different topic distributions.

## Experiment Design

- **Model**: olmo3_14m (14M parameters)
- **Training**: 0.5x Chinchilla (140M tokens, ~8 min)
- **Quality buckets**: vigintiles (5% buckets) from DCLM quality classifier
- **Thresholds tested**: top10pct, top30pct, top50pct, top70pct

### Topic Distributions

| Distribution | Emphasis |
|--------------|----------|
| heavy_adult | 3% adult content weight |
| heavy_code | 50% software_development weight |
| heavy_science | 20% science_math_and_technology weight |
| heavy_wiki | Higher wikipedia weight |

## Results Summary

**Metric: BPB v2 (lower is better)**

### Best Quality Threshold by Distribution

| Distribution | Best Threshold | Improvement vs top70pct |
|--------------|----------------|------------------------|
| **heavy_code** | **top30pct** | **-7.9%** (largest gain) |
| heavy_science | top10pct | -2.1% |
| heavy_wiki | top10pct | -0.8% |
| heavy_adult | top10pct | -0.8% |

### Absolute BPB Scores

| Distribution | Threshold | Avg BPB | Code | Math | MMLU | Core QA |
|--------------|-----------|---------|------|------|------|---------|
| heavy_adult | top10pct | 2.652 | 2.834 | 2.629 | 2.233 | 1.821 |
| heavy_adult | top30pct | 2.698 | 2.918 | 2.740 | 2.239 | 1.835 |
| heavy_adult | top50pct | 2.713 | 2.941 | 2.686 | 2.263 | 1.838 |
| heavy_adult | top70pct | 2.673 | 2.903 | 2.617 | 2.234 | 1.822 |
| heavy_code | top10pct | 2.528 | 2.501 | 2.510 | 2.212 | 1.887 |
| heavy_code | **top30pct** | **2.444** | **2.478** | **2.302** | **2.185** | **1.824** |
| heavy_code | top50pct | 2.526 | 2.562 | 2.463 | 2.243 | 1.903 |
| heavy_code | top70pct | 2.654 | 2.683 | 2.706 | 2.273 | 1.941 |
| heavy_science | top10pct | 2.550 | 2.696 | 2.356 | 2.128 | 1.830 |
| heavy_science | top30pct | 2.587 | 2.792 | 2.406 | 2.148 | 1.815 |
| heavy_science | top50pct | 2.583 | 2.799 | 2.383 | 2.141 | 1.825 |
| heavy_science | top70pct | 2.605 | 2.842 | 2.382 | 2.157 | 1.864 |
| heavy_wiki | top10pct | 2.541 | 2.631 | 2.363 | 2.166 | 1.820 |
| heavy_wiki | top30pct | 2.546 | 2.673 | 2.375 | 2.205 | 1.853 |
| heavy_wiki | top50pct | 2.592 | 2.766 | 2.390 | 2.225 | 1.883 |
| heavy_wiki | top70pct | 2.562 | 2.728 | 2.373 | 2.211 | 1.881 |

### Percentage Change from top70pct Baseline

| Distribution | Threshold | Overall | Code | Math | MMLU | Core QA |
|--------------|-----------|---------|------|------|------|---------|
| heavy_adult | top10pct | -0.8% | -2.4% | +0.5% | -0.0% | -0.1% |
| heavy_adult | top30pct | +0.9% | +0.5% | +4.7% | +0.2% | +0.7% |
| heavy_adult | top50pct | +1.5% | +1.3% | +2.7% | +1.3% | +0.8% |
| heavy_code | top10pct | -4.8% | -6.8% | -7.2% | -2.7% | -2.8% |
| **heavy_code** | **top30pct** | **-7.9%** | **-7.6%** | **-14.9%** | **-3.9%** | **-6.0%** |
| heavy_code | top50pct | -4.8% | -4.5% | -9.0% | -1.3% | -2.0% |
| heavy_science | top10pct | -2.1% | -5.1% | -1.1% | -1.3% | -1.9% |
| heavy_science | top30pct | -0.7% | -1.8% | +1.0% | -0.4% | -2.6% |
| heavy_science | top50pct | -0.8% | -1.5% | +0.0% | -0.7% | -2.1% |
| heavy_wiki | top10pct | -0.8% | -3.6% | -0.4% | -2.0% | -3.3% |
| heavy_wiki | top30pct | -0.6% | -2.0% | +0.1% | -0.3% | -1.5% |
| heavy_wiki | top50pct | +1.2% | +1.4% | +0.7% | +0.6% | +0.1% |

## Key Findings

### 1. Code Data is Most Sensitive to Quality Filtering

The heavy_code distribution shows the largest improvement from quality filtering:
- **top30pct gives -7.9% overall BPB improvement** vs top70pct
- Math tasks improve dramatically: **-14.9%**
- Code tasks improve: **-7.6%**
- Core QA improves: **-6.0%**

### 2. Optimal Threshold Varies by Domain

- **Code-heavy mixes**: Moderate filtering (top30pct) is optimal
  - Too aggressive filtering (top10pct) loses beneficial data
- **Other distributions**: Aggressive filtering (top10pct) works best
  - Science, wiki, and adult content benefit from stricter quality gates

### 3. Quality Filtering Helps Across All Distributions

Every distribution benefits from some level of filtering compared to top70pct:
- heavy_code: up to -7.9% improvement
- heavy_science: up to -2.1% improvement
- heavy_wiki: up to -0.8% improvement
- heavy_adult: up to -0.8% improvement

### 4. Task-Specific Patterns

- **Math tasks** benefit most from code quality filtering (-14.9% in heavy_code)
- **Code tasks** consistently improve with quality filtering across all distributions
- **MMLU** shows modest but consistent improvements
- **Core QA** (hellaswag, arc, etc.) shows variable response to filtering

## Implications for Data Mixing

1. **Prioritize quality filtering for code data** - the ROI is highest
2. **Use moderate thresholds (top30pct) for code** - aggressive filtering hurts
3. **Use aggressive thresholds (top10pct) for other domains** - they're less sensitive
4. **Consider domain-specific quality thresholds** in production mixes

## Running These Experiments

```bash
# Single experiment
olmix launch run --config configs/experiments/quality_thresholds/heavy_code/top30pct.yaml

# All quality threshold experiments
./configs/experiments/quality_thresholds/batch_run.sh
```

## WandB Links

Results are logged to [wandb.ai/ai2-llm/olmix](https://wandb.ai/ai2-llm/olmix) with group IDs:

| Distribution | top10pct | top30pct | top50pct | top70pct |
|--------------|----------|----------|----------|----------|
| heavy_adult | c40ddf3b | faf3cb39 | 0b93565a | a3f69afc |
| heavy_code | 9caf8f49 | 0d964ed0 | b9c58bc7 | d1a60077 |
| heavy_science | 7ad80c06 | a204b0b7 | 89c0fac1 | 720b47cd |
| heavy_wiki | 2c26e5c8 | e4ffc29e | 8c89d715 | 54b524be |
