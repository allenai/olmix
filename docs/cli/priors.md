# `olmix priors compute` Reference

## Purpose

Scan configured data paths and compute a `priors` block containing:

- `relative_sizes`
- `token_counts`

The output can be pasted into a generation config.

## Command Syntax

```bash
olmix priors compute --config <config-with-data-sources.yaml> [--no-cache] [--output <file>]
```

## Command Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `-c, --config` | Yes | Config file containing `data.sources` |
| `-n, --no-cache` | No | Disable cached token counts |
| `-o, --output` | No | Write YAML output to file; default prints to stdout |

## Minimal Example

```bash
olmix priors compute --config configs/examples/generate/example.yaml
```

Write to file:

```bash
olmix priors compute \
  --config configs/examples/generate/example.yaml \
  --output output/priors.yaml
```

## Input Rules

- `--config` can be either:
  - a full generation config containing `data`, or
  - a document where root is directly `data`.
- Paths must be reachable from your environment.

## Output Format

YAML:

```yaml
priors:
  relative_sizes:
    source_or_leaf: 0.123
  token_counts:
    source_or_leaf: 123456789
```

## Common Failure Modes

- Invalid `data.sources` schema.
- Unreachable cloud/local paths.
- Slow scans when cache is disabled (`--no-cache`).
