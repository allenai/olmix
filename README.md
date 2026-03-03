# Olmix

[![CI](https://github.com/allenai/olmix/actions/workflows/main.yml/badge.svg)](https://github.com/allenai/olmix/actions/workflows/main.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
![Status: WIP](https://img.shields.io/badge/status-WIP-yellow)

> [!WARNING]
> This project is under active development. We are migrating from our internal infrastructure to open source — expect rough edges, missing docs, and breaking changes.

Toolkit for optimizing pretraining data mixtures. Learns from small-scale proxy experiments ("swarms") to predict how data mixing ratios affect downstream performance, then proposes optimized mixtures for full-scale training.

## End-To-End Mixing Flow

```text
+----------------------------+
| Define Data Sources        |
+----------------------------+
             |
             v
+--------------------------------------------------------------+
| Compute Priors                                               |
| (olmix priors compute --config ...)                         |
+--------------------------------------------------------------+
             |
             v
+--------------------------------------------------------------------------+
| Generate Swarm Variants                                                  |
| (olmix generate --config ... --base ... --output ...)                   |
+--------------------------------------------------------------------------+
             |
             v
+--------------------------------------------------------------+
| Launch Proxy Runs                                            |
| (olmix launch run --variants ...)                            |
+--------------------------------------------------------------+
             |
             v
+-----------------------------------------+
| Collect + Export ratios.csv + metrics.csv |
+-----------------------------------------+
             |
             v
+--------------------------------------------------------------+
| Fit Regressors + Propose Optimized Mix                      |
| (olmix fit --config ... --output-dir ...)                   |
+--------------------------------------------------------------+
             |
             v
+------------------------------------+
| Use Mix for Full-Scale Training    |
+------------------------------------+
```

## Installation

```bash
git clone https://github.com/allenai/olmix.git
cd olmix
uv pip install -e ".[dev]"
```

## Quickstart

Choose one path:

- Pattern 1 (fastest): you already have swarm results in CSV, and want optimized mix weights.
- Pattern 2 (end-to-end): you want to generate variants, launch proxy runs, then fit.

### Pattern 1: CSV -> Fit

Prepare:

- `ratios.csv` with one row per run and domain columns summing to ~1.0.
- `metrics.csv` with one row per run and evaluation metric columns.
- A fit config YAML (see `configs/examples/fit/example.yaml`).

Minimal example files:

```csv
# ratios.csv
run,name,index,dclm,wikipedia,arxiv
hz0dfydj,my-swarm-0000,0,0.45,0.30,0.25
pj0hxxl7,my-swarm-0001,1,0.60,0.20,0.20
sqleanmq,my-swarm-0002,2,0.33,0.33,0.34
```

```csv
# metrics.csv
run,name,index,arc_challenge_bpb,hellaswag_bpb,mmlu_stem_bpb
hz0dfydj,my-swarm-0000,0,1.23,0.87,1.45
pj0hxxl7,my-swarm-0001,1,1.15,0.91,1.38
sqleanmq,my-swarm-0002,2,1.20,0.89,1.42
```

Run:

```bash
olmix fit --config configs/examples/fit/example.yaml --output-dir output/my_fit
```

### Pattern 2: Generate -> Launch -> Fit

1) Compute priors for your generation config:

```bash
olmix priors compute --config configs/examples/generate/example.yaml
```

2) Generate launch variants:

```bash
olmix generate \
  --config configs/examples/generate/example.yaml \
  --base configs/examples/launch/data_proportions/mix_baseline.yaml \
  --output output/my_variants/
```

3) Launch the swarm:

```bash
olmix launch run --variants output/my_variants/
```

4) Export swarm outputs to `ratios.csv` and `metrics.csv`, then fit:

```bash
olmix fit --config configs/examples/fit/example.yaml --output-dir output/my_fit
```

## Required Inputs (At A Glance)

- For fit: a YAML config with `swarm.ratios`, `swarm.metrics`, and `priors`.
- For generate: a generation config with `data`, `priors`, `swarm`, and `max_tokens`.
- For launch: generated variant YAML files plus Beaker environment access.
- For end-to-end: a way to export run mixtures and eval metrics to CSV for fitting.

## What You Get

- A hashed fit output directory with `config.json` for reproducibility.
- Regression diagnostics (`*_fit.png`, `*_correlations.json`, `interaction_matrix.*`).
- Proposed optimal mixture files (`*_optimal.json`, `*_optimal.png`) unless `fit_only: true`.
- Launch metadata under `output/mixes/...` when using `olmix launch run`.

## CLI And Config Reference

Detailed arguments and config-field breakdowns live in `docs/cli/`:

- [`docs/cli/README.md`](docs/cli/README.md)
- [`docs/cli/fit.md`](docs/cli/fit.md)
- [`docs/cli/generate.md`](docs/cli/generate.md)
- [`docs/cli/launch.md`](docs/cli/launch.md)
- [`docs/cli/priors.md`](docs/cli/priors.md)

`olmix fit` is the canonical fit command. `olmix-fit` is a legacy compatibility alias.

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
