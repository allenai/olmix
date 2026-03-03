# `olmix generate` Reference

## Purpose

Sample mixture variants from a `GenerationConfig` and write one self-contained
launch config YAML per variant.

## Command Syntax

```bash
olmix generate --config <generation.yaml> --base <launch-base.yaml> --output <dir>
```

## Command Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `-c, --config` | Yes | Path to generation config YAML |
| `--base` | Yes | Path to base launch config YAML template |
| `-o, --output` | Yes | Output directory for generated variant YAML files |

## Minimal Example

```bash
olmix generate \
  --config configs/examples/generate/example.yaml \
  --base configs/examples/launch/data_proportions/mix_baseline.yaml \
  --output output/my_variants/
```

## Output

One launch config YAML per variant, named `<swarm-name>-<group-id>-<index>.yaml`.

Example:

```text
output/my_variants/
  example-swarm-a1b2c3d4-0000.yaml
  example-swarm-a1b2c3d4-0001.yaml
```

## `GenerationConfig` Schema

Top-level keys:

```yaml
name: my-swarm
data:
priors:
swarm:
max_tokens:
```

### `data`

`data.sources` is hierarchical. A source can define exactly one shape:

- `paths` (flat source)
- `topics` (source with topic-level children)
- `quality` (source with quality-level children)

Mixture reuse pattern:

- A topic-level `weight` pins that topic's relative share.
- Topics without `weight` remain free and are resampled each variant.

### `priors`

Leaf-level priors used by generation:

| Field | Description |
|------|-------------|
| `relative_sizes` | Dirichlet prior center at leaf level |
| `token_counts` | Leaf token counts for repetition bounds |

Use `olmix priors compute` to generate this block from data paths.

### `swarm`

| Field | Description | Default |
|------|-------------|---------|
| `variants` | Number of generated mixtures | `1` |
| `seed` | Random seed | `42` |
| `min_strength` / `max_strength` | Dirichlet concentration range | `0.1` / `5.0` |
| `min_source_strength` / `max_source_strength` | Source-level concentration override | None |
| `min_topic_strength` / `max_topic_strength` | Topic-level concentration override | None |
| `minimum_weight` | Global zeroing threshold | `0.002` |
| `minimum_source_weight` / `minimum_topic_weight` | Level-specific threshold overrides | None |
| `nonzero_weight` | Source/topic keys that must remain nonzero | None |
| `manual_prior` | Source-level prior override | None |
| `manual_topic_prior` | Topic-level prior override | None |
| `repetition_factor` | Maximum repetition multiplier | `1.0` |
| `enable_bound` | Enforce repetition bounds while sampling | `true` |
| `existing_mix_file` | Existing swarm pickle for near-duplicate rejection | None |

## Common Failure Modes

- Missing `priors` block or mismatched keys between priors and data leaves.
- Invalid source/topic structure (multiple shape types at same node).
- Unreachable cloud paths when computing priors.
- Bounds too strict for requested `variants` and `max_tokens`.
