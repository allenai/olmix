# Plan: Reorganize output/mixes/ to mirror config/examples/launch/ hierarchy

## Context

**Current state:**
- Config paths: `config/examples/launch/<group>/<subgroup>/<name>.yaml`
- Output paths: `output/mixes/<config-name>_<group_uuid>.json` (flat)

**Examples:**
| Config | Current Output | Desired Output |
|--------|----------------|----------------|
| `config/examples/launch/training_duration/duration_0.5x.yaml` | `output/mixes/duration-0.5x_e326e56e.json` | `output/mixes/training_duration/duration_0.5x_e326e56e.json` |
| `config/examples/launch/quality_thresholds/heavy_code/top10pct.yaml` | `output/mixes/quality-top10pct-heavy-code_333ccd44.json` | `output/mixes/quality_thresholds/heavy_code/top10pct_333ccd44.json` |

---

## CLI Launching Pattern

The CLI keeps the single-file pattern. For batch operations, users use shell scripting:

```bash
# Launch all experiments in a directory
for f in config/examples/launch/quality_thresholds/**/*.yaml; do
    olmix launch run --config "$f"
done
```

With the output path fix, this produces organized outputs:
```
output/mixes/quality_thresholds/heavy_code/top10pct_abc123.json
output/mixes/quality_thresholds/heavy_code/top30pct_def456.json
output/mixes/quality_thresholds/heavy_science/top10pct_789ghi.json
...
```

---

## Files to Modify

| File | Change |
|------|--------|
| `olmix/cli.py` | Update `_save_launch_metadata()` to derive output path from config path |
| `olmix/launch/launch_utils.py` | Update `mk_mixes()` to derive output path from config path |
| `config/examples/launch/README.md` | Update documentation to reflect new output structure |
| `config/examples/launch/data_proportions/batch_run.sh` | Create batch launch script |
| `config/examples/launch/quality_thresholds/batch_run.sh` | Create batch launch script |
| `config/examples/launch/training_duration/batch_run.sh` | Create batch launch script |

## Implementation

### 1. Add helper function in `olmix/cli.py`

```python
def _get_output_path_from_config(config_path: Path, group_uuid: str) -> Path:
    """Derive output path from config path, mirroring the config hierarchy.

    Example:
        config/examples/launch/quality_thresholds/heavy_code/top10pct.yaml
        -> output/mixes/quality_thresholds/heavy_code/top10pct_<uuid>.json
    """
    config_path = Path(config_path).resolve()

    # Find the experiments/ directory in the path
    parts = config_path.parts
    try:
        experiments_idx = parts.index("experiments")
    except ValueError:
        # Fallback: use just the filename stem if not in experiments/
        return Path(f"output/mixes/{config_path.stem}_{group_uuid}.json")

    # Get the relative path after experiments/
    relative_parts = parts[experiments_idx + 1:]
    # Replace .yaml extension with _{uuid}.json
    relative_path = Path(*relative_parts)
    output_name = f"{relative_path.stem}_{group_uuid}.json"
    output_dir = relative_path.parent

    return Path("output/mixes") / output_dir / output_name
```

### 2. Update `_save_launch_metadata()` in `olmix/cli.py` (line 37)

**Before:**
```python
output_path = Path(f"output/mixes/{experiment_name}_{group_uuid}.json")
```

**After:**
```python
output_path = _get_output_path_from_config(config_path, group_uuid)
```

### 3. Update `mk_mixes()` in `olmix/launch/launch_utils.py` (line 115)

Add helper function at top of file (or import from cli.py):

```python
def _get_output_path_from_config(config_path: Path, group_uuid: str) -> Path:
    """Derive output path from config path, mirroring the config hierarchy."""
    config_path = Path(config_path).resolve()
    parts = config_path.parts
    try:
        experiments_idx = parts.index("experiments")
    except ValueError:
        return Path(f"output/mixes/{config_path.stem}_{group_uuid}.json")

    relative_parts = parts[experiments_idx + 1:]
    relative_path = Path(*relative_parts)
    output_name = f"{relative_path.stem}_{group_uuid}.json"
    output_dir = relative_path.parent

    return Path("output/mixes") / output_dir / output_name
```

**Before (line 115):**
```python
output = Path(f"output/mixes/{config.name}_{group_uuid}.json")
```

**After:**
```python
output = _get_output_path_from_config(config_file, group_uuid)
```

### 4. Create batch_run.sh scripts

**`config/examples/launch/data_proportions/batch_run.sh`:**
```bash
#!/bin/bash
# Batch launch all data proportion experiments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/*.yaml; do
    echo "Launching: $config"
    olmix launch run --config "$config"
done
```

**`config/examples/launch/training_duration/batch_run.sh`:**
```bash
#!/bin/bash
# Batch launch all training duration experiments
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/*.yaml; do
    echo "Launching: $config"
    olmix launch run --config "$config"
done
```

**`config/examples/launch/quality_thresholds/batch_run.sh`:**
```bash
#!/bin/bash
# Batch launch all quality threshold experiments (nested directories)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for config in "$SCRIPT_DIR"/**/*.yaml; do
    echo "Launching: $config"
    olmix launch run --config "$config"
done
```

### 5. Update README (line 324)

**Before:**
```
Launch metadata is saved to `output/mixes/{name}_{group_id}.json` with:
```

**After:**
```
Launch metadata is saved to `output/mixes/<experiment_path>/{name}_{group_id}.json`, mirroring the config hierarchy. For example:
- Config: `config/examples/launch/quality_thresholds/heavy_code/top10pct.yaml`
- Output: `output/mixes/quality_thresholds/heavy_code/top10pct_abc123.json`
```

## Verification

1. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

2. **Dry-run an experiment and verify output path:**
   ```bash
   olmix launch run --config config/examples/launch/training_duration/duration_0.5x.yaml --dry-run
   ```
   Should show output path like: `output/mixes/training_duration/duration_0.5x_<uuid>.json`

3. **Test with nested config:**
   ```bash
   olmix launch run --config config/examples/launch/quality_thresholds/heavy_code/top10pct.yaml --dry-run
   ```
   Should show: `output/mixes/quality_thresholds/heavy_code/top10pct_<uuid>.json`

4. **Verify batch scripts are executable:**
   ```bash
   ls -la config/examples/launch/*/batch_run.sh
   ```

## Notes

- Existing output files in `output/mixes/` will remain unchanged (flat structure)
- New launches will use the hierarchical structure
- The helper function gracefully falls back to using filename if config isn't under `experiments/`
- Batch scripts must be made executable with `chmod +x`
