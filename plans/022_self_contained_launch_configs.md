# Self-contained launch configs from `olmix generate`

## Context

After splitting `ExperimentConfig` into `GenerationConfig` + `LaunchConfig` + `VariantConfig`, the launch flow requires joining two files at launch time:

```bash
olmix generate -c gen.yaml -o variants/                        # produces VariantConfig (name + mix only)
olmix launch run -c launch.yaml --variants variants/           # joins launch config + variants
```

This is awkward: `data.sources` is duplicated between the generation config and the launch config, the domain names must match across files, and the user has to remember to pass two arguments. The output of `olmix generate` should be self-contained — each file is a complete launch config ready to submit.

## New workflow

```bash
olmix generate -c gen.yaml --base launch.yaml -o variants/     # produces self-contained LaunchConfigs
olmix launch run -v variants/                                   # just launches, no joining
```

## Changes

### 1. Add `mix` and `group_id` fields to `LaunchConfig`; remove `VariantConfig` (`olmix/aliases.py`)

- Move `MixEntry` class above `LaunchConfig` (it was defined after it, causing a forward-reference error)
- Add two optional fields to `LaunchConfig`:
  ```python
  mix: dict[str, MixEntry] | None = None    # sampled weights per domain
  group_id: str | None = None               # shared across variants in a batch
  ```
- Delete `VariantConfig` class entirely — it's no longer produced or consumed

Both new fields default to `None` so existing launch config YAMLs parse without change.

### 2. Update exports (`olmix/__init__.py`, `olmix/launch/__init__.py`)

- Remove `VariantConfig` from `olmix/__init__.py` imports and `__all__`
- Remove `mk_experiments` from `olmix/launch/__init__.py` optional beaker exports
- Add `launch_noninteractive` to `olmix/launch/__init__.py` exports

### 3. Update `olmix/launch/beaker.py`

- Remove `VariantConfig` import
- Remove `mk_experiments()` function (was a helper that created `ExperimentInstance` from `VariantConfig`)
- Change `mk_experiment_group()` signature from `(config: LaunchConfig, variants: list[VariantConfig], group_uuid: str)` to `(configs: list[LaunchConfig], group_uuid: str)` — it now builds instances directly from each config's `.mix` field
- Add `launch_noninteractive()` helper (see section 7 below)

### 4. Rewrite CLI commands (`olmix/cli.py`)

Add `from __future__ import annotations` and `TYPE_CHECKING` guard for `LaunchConfig`.

**Remove `_get_output_path_from_config()`** — no longer needed.

**Replace `_load_variants()` with `_load_launch_configs()`** — same glob logic but returns `list[LaunchConfig]` instead of `list[VariantConfig]`.

**Update `_save_launch_metadata()`** — accept `configs: list[LaunchConfig]` instead of `config_path: Path` + `variants_dir: str`. Uses `configs[0]` for base info and `base.model_dump(mode="json")` for the config snapshot.

**`generate` command:**
- Add `--base` flag (required, `click.Path(exists=True)`)
- Load `GenerationConfig` from `--config` and `LaunchConfig` from `--base`
- For each sampled mix, stamp it onto a copy of base: `base_config.model_copy(update={"name": variant_name, "mix": mix, "group_id": group_uuid})`
- Write each as a complete `LaunchConfig` YAML via `yaml.dump(variant_config.model_dump(mode="json"), ...)`

**`launch run` command:**
- Remove `--config` flag entirely
- Load all configs from `--variants` dir via `_load_launch_configs()`
- Extract `group_uuid` from `configs[0].group_id` (fall back to `generate_uuid()[:8]`)
- Pass `configs` list to `mk_experiment_group(configs, group_uuid)`
- Use `launch_noninteractive()` instead of `blc.launch()` (see section 7)

**`launch preview` command:**
- Remove `--config` flag entirely
- Same pattern as `launch run` — load configs from `--variants` dir

**`launch status` / `launch cancel`:**
- Make `--group-id` optional (was required)
- Auto-extract from config's `group_id` field: `gid = group_id or launch_cfg.group_id`
- Raise `click.BadParameter` if neither is provided

### 5. Update tests

**`tests/test_config.py`:**
- Remove `VariantConfig` from imports
- Remove `TestVariantConfig` class
- Add `TestLaunchConfigMix` class with 3 tests:
  - `test_launch_config_without_mix` — backward compat (mix=None, group_id=None)
  - `test_launch_config_with_mix` — both fields populated
  - `test_launch_config_with_mix_from_yaml` — round-trip through YAML

**`tests/test_cli.py`:**
- `test_generate_help`: Add assertion for `--base`
- `test_launch_run_help`: Remove assertion for `--config`
- `test_launch_preview_help`: Remove assertion for `--config`

**`tests/test_imports.py`:**
- Replace `VariantConfig` with `MixEntry` in aliases import test
- Remove `mk_experiments` from beaker import test

### 6. Update README (`README.md`)

Document the new workflow:
- `olmix generate -c gen.yaml --base launch.yaml -o variants/` → self-contained LaunchConfig YAMLs
- `olmix launch preview -v variants/` and `olmix launch run -v variants/` — no `--config` needed
- Show example output YAML with full infra/training/data/eval/mix/group_id

### 7. Non-interactive Beaker launch (`olmix/launch/beaker.py`)

Add `launch_noninteractive(config, torchrun)` helper that bypasses two problems with `BeakerLaunchConfig.launch()` from olmo-core:

**Problem 1: Interactive prompts.** Gantry's `launch_experiment()` has multiple `prompt.Confirm.ask()` / `prompt.Prompt.ask()` calls (experiment name, GitHub token, group creation, overwrite confirmation). `BeakerLaunchConfig` does not expose gantry's `yes` parameter, so these prompts fire interactively. When launching multiple experiments in a loop, stdin exhaustion after the first experiment causes the remaining ones to silently fail.

**Problem 2: Blocking follow mode.** `BeakerLaunchConfig` defaults to `follow=True`, which means gantry streams experiment logs and blocks until the experiment completes. In a loop of 4 experiments, the first one blocks forever — the remaining 3 never launch.

**Solution:** `launch_noninteractive()` calls `config._build_recipe(beaker, torchrun=torchrun, follow=False)` to get the gantry recipe, sets `recipe.yes = True`, then calls `recipe.launch()`. This skips all interactive prompts and returns immediately after submission.

```python
def launch_noninteractive(config: BeakerLaunchConfig, torchrun: bool = False):
    from olmo_core.launch.beaker import get_beaker_client

    with get_beaker_client(workspace=config.workspace) as beaker:
        recipe, recipe_launch_kwargs = config._build_recipe(
            beaker, torchrun=torchrun, follow=False
        )
        recipe.yes = True
        return recipe.launch(client=beaker, **recipe_launch_kwargs)
```

The CLI's own `click.confirm("Proceed with this configuration?")` remains the single interactive gate before launching.

## Files changed

| File | Change |
|------|--------|
| `olmix/aliases.py` | Move `MixEntry` up; add `mix`, `group_id` to `LaunchConfig`; remove `VariantConfig` |
| `olmix/__init__.py` | Remove `VariantConfig` export |
| `olmix/launch/__init__.py` | Remove `mk_experiments`; add `launch_noninteractive` export |
| `olmix/launch/beaker.py` | Remove `mk_experiments`; update `mk_experiment_group`; add `launch_noninteractive` |
| `olmix/cli.py` | Add `--base` to `generate`; simplify `launch run`/`preview`; update `status`/`cancel`; use `launch_noninteractive` |
| `tests/test_config.py` | Remove `TestVariantConfig`; add `TestLaunchConfigMix` |
| `tests/test_cli.py` | Update help assertions |
| `tests/test_imports.py` | Remove `VariantConfig`; add `MixEntry` |
| `README.md` | Update workflow docs |

## Files NOT changed

| File | Why |
|------|-----|
| `olmix/launch/utils.py` | `mk_source_instances` unchanged — still takes `dict[str, MixEntry]` |
| `olmix/generate/utils.py` | `mk_mixes` unchanged — still returns `list[dict[str, MixEntry]]` |
| `olmix/generate/synthesize_mixture.py` | No changes needed |
| `olmix/fit/utils.py` | No changes needed (new fields default to `None`) |
| `olmix/fit/core.py` | No changes needed |

## Verification

1. `make run-checks` passes (format, lint, pyright 0 errors, 97 tests pass)
2. `olmix generate -c configs/generations/example.yaml --base configs/experiments/data_proportions/mix_baseline.yaml -o /tmp/test_gen` — produces 4 self-contained LaunchConfig YAMLs with `mix` and `group_id` fields
3. `olmix launch preview -v /tmp/test_gen/` — prints training commands for all 4 variants (no `--config` needed)
4. `olmix launch run -v /tmp/test_gen/` — submits all 4 experiments to Beaker, confirmed 4/4 `STATUS_QUEUED`
5. `olmix launch run --help` — shows `-v/--variants` only, no `--config`
6. Existing `LaunchConfig.from_yaml("configs/experiments/...")` still works (mix=None, group_id=None)
