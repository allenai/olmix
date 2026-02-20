# olmix Experiments

Experiment configs for validating olmix training and evaluating data mixing strategies.

## Experiment Suites

| Suite | Directory | Description |
|-------|-----------|-------------|
| [Training Duration](training_duration/) | `training_duration/` | Tests BPB improvement with longer training (0.5x-5.0x Chinchilla) |
| [Data Proportions](data_proportions/) | `data_proportions/` | Tests different data mixes (code, science, wiki emphasis) |
| [Quality Upsampling](quality_upsampling/) | `quality_upsampling/` | Tests quality-weighted data mixing strategies |

## Running Experiments

```bash
# Single experiment
olmix launch run --config configs/experiments/training_duration/duration_0.5x.yaml

# All experiments in a suite
for f in configs/experiments/training_duration/*.yaml; do
  olmix launch run --config $f
done
```

## Model & Infrastructure

All experiments use:
- Model: olmo3_14m (14M parameters)
- Tokenizer: dolma2
- Cluster: ai2/jupiter (1 GPU)
- Eval: 60 BPB tasks every 1000 steps (aligned with olmo-core's FULL_TASKS_SMALL_COMPUTE)

## Experiment Tracking

Launch metadata is saved to `output/mixes/<experiment_path>/<config_name>/<timestamp>-<group_id>.json`, mirroring the config hierarchy with config-based grouping. For example:
- Config: `configs/experiments/quality_thresholds/heavy_code/top10pct.yaml`
- Output: `output/mixes/quality_thresholds/heavy_code/top10pct/20260204_143025-abc123.json`

This structure groups all runs of the same config together, with each run uniquely identified by its timestamp (`YYYYMMDD_HHMMSS` UTC) and group ID.

The metadata includes:
- Full config contents (for reproducibility even if the YAML file changes)
- Beaker experiment IDs and URLs
- WandB group URL
- Git commit and branch
- Generated mix configurations
