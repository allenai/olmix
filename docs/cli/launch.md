# `olmix launch` Reference

## Purpose

Operate on generated variant launch configs:

- submit a swarm
- preview commands
- inspect status
- cancel a running group

## `olmix launch run`

Submit one training job per variant config.

```bash
olmix launch run --variants <variant-file-or-dir> [--dry-run]
```

| Flag | Required | Description |
|------|----------|-------------|
| `-v, --variants` | Yes | YAML file or directory of YAML files |
| `--dry-run` | No | Build launch configs and print without submitting |

Notes:

- Command prompts for confirmation before launching.
- Metadata is saved under `output/mixes/<experiment-name>/<group-id>.json`.

## `olmix launch preview`

Preview launch commands without submitting.

```bash
olmix launch preview --variants <variant-file-or-dir>
```

| Flag | Required | Description |
|------|----------|-------------|
| `-v, --variants` | Yes | YAML file or directory of YAML files |

## `olmix launch status`

Query statuses for an experiment group.

```bash
olmix launch status --config <launch-config.yaml> [--group-id <id>]
```

| Flag | Required | Description |
|------|----------|-------------|
| `-c, --config` | Yes | Any generated launch config YAML |
| `-g, --group-id` | No | Override group id (otherwise taken from config) |

## `olmix launch cancel`

Cancel jobs for an experiment group.

```bash
olmix launch cancel --config <launch-config.yaml> [--group-id <id>]
```

| Flag | Required | Description |
|------|----------|-------------|
| `-c, --config` | Yes | Any generated launch config YAML |
| `-g, --group-id` | No | Override group id (otherwise taken from config) |

## Requirements

- Beaker credentials/environment for run/status/cancel.
- Variant configs produced by `olmix generate` or equivalent launch YAMLs.

## Common Failure Modes

- `--variants` path has no YAML files.
- Missing `group_id` and no `--group-id` provided for status/cancel.
- Beaker auth or cluster access issues.
