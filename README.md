# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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

**Sample candidate mixtures** from a config (outputs a JSON of mixing ratios):
```bash
olmix mix generate --config configs/experiments/data_proportions/mix_baseline.yaml
```

**Preview training commands** without launching (dry run):
```bash
olmix launch preview --config configs/experiments/data_proportions/mix_baseline.yaml
```

**Launch swarm experiments** to Beaker (one training run per mixture variant):
```bash
olmix launch run --config configs/experiments/data_proportions/mix_baseline.yaml
```

**Fit regression and propose an optimal mixture** from completed swarm results in W&B:
```bash
olmix fit \
  --experiment-groups WANDB_GROUP_ID \
  --config path/to/config.yaml \
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
