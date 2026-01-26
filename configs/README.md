# OLMix Configuration Files

This directory contains example and production configuration files for OLMix experiments.

## Directory Structure

```
configs/
├── README.md
└── examples/
    ├── test_olmo2_30m.yaml      # Minimal test config
    └── mixture_olmo2_30m.yaml   # Multi-source mixture example
```

## Usage

### Validate a configuration
```bash
olmix launch validate --config configs/examples/test_olmo2_30m.yaml
```

### Dry run (see what would be launched)
```bash
olmix launch run --config configs/examples/test_olmo2_30m.yaml --dry-run
```

### Launch to Beaker
```bash
olmix launch run --config configs/examples/test_olmo2_30m.yaml
```

## Available Models

| Model ID | Parameters | Description |
|----------|------------|-------------|
| `olmo2_1m` | ~1M | Tiny test model |
| `olmo2_30m` | ~30M | Small proxy model |
| `olmo2_60m` | ~60M | Medium-small proxy model |
| `olmo2_190m` | ~190M | Medium proxy model |
| `olmo2_1b` | ~1B | Large model |
| `olmo2_7b` | ~7B | Full-scale model |

## Available Tokenizers

| Tokenizer | Vocab Size | Description |
|-----------|------------|-------------|
| `dolma2` | 100,278 | Default tokenizer (recommended) |
| `gpt_neox` | 50,304 | Legacy tokenizer |

## Configuration Fields

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Experiment name |
| `budget` | string | Beaker budget (e.g., `ai2/oe-data`) |
| `workspace` | string | Beaker workspace (e.g., `ai2/dolma2`) |
| `cluster` | string | Beaker cluster (e.g., `ai2/saturn-cirrascale`) |
| `proxy_model_id` | string | Model identifier (e.g., `olmo2_30m`) |
| `tokenizer` | string | Tokenizer identifier (e.g., `dolma2`) |
| `nodes` | int | Number of nodes |
| `gpus` | int | GPUs per node |
| `variants` | int | Number of mixture variants |
| `max_tokens` | int | Tokens per variant |
| `sequence_length` | int | Sequence length (2048 or 4096) |
| `seed` | int | Random seed |
| `sources` | list | Data sources configuration |

### Optional Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `description` | string | `""` | Experiment description |
| `priority` | string | `normal` | Job priority (low/normal/high/urgent) |
| `mix_temperature` | float | `1.0` | Mixture sampling temperature |
| `min_strength` | float | `0.1` | Min Dirichlet strength |
| `max_strength` | float | `5.0` | Max Dirichlet strength |
| `train_type` | string | `pretrain` | Training type (pretrain/anneal) |
| `weka` | bool | `false` | Use Weka filesystem |
| `preemptible` | bool | `true` | Allow job preemption |

### Source Configuration

Each source in `sources` can have:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Source name |
| `paths` | list[string] | required | S3/GCS paths to data |
| `max_source_ratio` | float | `1.0` | Max proportion of this source |
| `max_repetition_factor` | float | `1.0` | Max data repetition |
