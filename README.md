# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)

> [!WARNING]
> This project is under active development. We are migrating from our internal infrastructure to open source — expect rough edges, missing docs, and breaking changes.

Toolkit for optimizing pretraining data mixtures. Learns from small-scale proxy experiments ("swarms") to predict how data mixing ratios affect downstream performance, then proposes optimized mixtures for full-scale training.

## Installation

```bash
git clone https://github.com/allenai/olmix.git
cd olmix
uv pip install -e ".[dev]"
```

## Getting started

Start from one of the experiment configs in [`configs/experiments/`](configs/experiments/):

| Suite | What it tests |
|-------|--------------|
| [`data_proportions/`](configs/experiments/data_proportions/) | Varying topic weights across sources |
| [`quality_thresholds/`](configs/experiments/quality_thresholds/) | Including/excluding quality vigintiles |
| [`quality_upsampling/`](configs/experiments/quality_upsampling/) | Weighting quality buckets within topics |
| [`training_duration/`](configs/experiments/training_duration/) | Effect of training length on mixture quality |


## Usage

**1. Sample candidate mixtures.** The input config defines sources hierarchically (source → topic → quality bucket) with S3 paths. This step scans those paths to count available tokens, uses the token counts as Dirichlet priors, and samples `variants` mixture configurations. Each mix flattens the hierarchy into fully-qualified leaf keys (e.g. `"dclm:software_development"`) with a `[weight, repetition_factor]` pair. Repetition factors are computed from how many tokens are needed vs. available. Output is written to `output/mixes/`.

```bash
olmix mix generate --config configs/experiments/data_proportions/mix_baseline.yaml
```

**2. Preview training commands.** Takes the sampled mixtures from step 1 and renders the full OLMo training command for each variant — model config, data paths, mixing weights, eval settings. Prints to stdout without launching anything.

```bash
olmix launch preview --config configs/experiments/data_proportions/mix_baseline.yaml
```

**3. Launch a swarm.** Submits one Beaker job per variant (steps 1-2 happen automatically if no pre-generated mix file is provided). Each job trains a proxy model on its mixture and logs eval metrics to W&B under a shared group ID. Launch metadata (Beaker experiment IDs, W&B group link, git commit) is saved alongside the mix JSON in `output/mixes/`.

```bash
olmix launch run --config configs/experiments/data_proportions/mix_baseline.yaml
```

**4. Fit and propose an optimal mixture.** Pulls eval metrics from completed swarm runs in W&B, pairs them with the mixing weights used in each run, and fits a regression model (LightGBM or log-linear) predicting metric values from weights. Then searches for weights that optimize the predicted metrics, subject to constraints. Outputs the proposed mixture weights, correlation plots, and predicted performance to the output directory.

```bash
olmix fit \
  --experiment-groups WANDB_GROUP_ID \
  --config configs/experiments/data_proportions/mix_baseline.yaml \
  --group-metrics all_metrics \
  --regression-type lightgbm
```

## Development

```bash
make run-checks   # format + lint + typecheck + test
```

## Citation

```bibtex
@article{chen2026olmix,
  title={Olmix: A Framework for Data Mixing Throughout LM Development},
  author={Chen, Mayee F and Murray, Tyler and Heineman, David and Jordan, Matt and Hajishirzi, Hannaneh and Re, Christopher and Soldaini, Luca and Lo, Kyle},
  year={2026},
  month={February}
}
```

## License

[Apache 2.0](LICENSE)
