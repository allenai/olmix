# Unify eval definitions into typed configs (Issue #17)

## Context

GitHub issue #17: evals are specified in both `model.evaluators` and `fit.constants` with three different naming conventions for the same underlying tasks. The fix: create two typed eval configs (`InLoopEvalConfig` and `OfflineEvalConfig`) that make these naming conventions explicit and user-configurable, inline in both fit and experiment YAML configs.

## Design

### Nested tasks define families

Tasks are grouped by family through YAML nesting — no separate `task_families` key needed.

**InLoopEvalConfig** — for WandB in-loop metrics:
- `tasks`: `dict[str, dict[str, str]]` — family → {task_id: wandb_metric_name}
- Used by `olmix launch` (task_ids) and `olmix fit` (metric_names as CSV columns)

**OfflineEvalConfig** — for cookbook-eval offline metrics:
- `tasks`: `dict[str, list[str]]` — family → list of metric names (= CSV column names)
- Used by `olmix fit` only

Both provide computed properties:
- `metric_names` → flattened list of all CSV column names
- `task_families` → `dict[str, list[str]]` for aggregation
- `task_ids` (InLoop only) → flattened list of olmo-core IDs

### YAML examples

InLoop (in experiment configs):
```yaml
eval:
  type: inloop
  tasks:
    math:
      gsm8k_gold_bpb_5shot: "eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)"
      minerva_math_algebra_gold_bpb_0shot: "eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB v2)"
      # ... (8 tasks)
    code:
      codex_humaneval_gold_bpb_3shot: "eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2)"
      # ... (19 tasks)
    qa:
      arc_challenge_test_rc_5shot: "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"
      # ... (33 tasks)
```

Offline (in fit configs):
```yaml
eval:
  type: offline
  tasks:
    math:
      - minerva_math_algebra::olmes
      # ... (7 tasks)
    code:
      - codex_humaneval:3shot::none
      - mbpp:3shot::none
      - mt_mbpp_v2fix:bash
      # ... (19 tasks)
    qa:
      - arc_challenge:rc::olmes
      # ... (26+ tasks)
```

## Implementation

### Step 1: Add eval config models to `olmix/fit/config.py`

New Pydantic models with Annotated discriminated union:

```python
class InLoopEvalConfig(BaseModel):
    type: Literal["inloop"] = "inloop"
    tasks: dict[str, dict[str, str]]       # {family: {task_id: metric_name}}

    @property
    def metric_names(self) -> list[str]: ...
    @property
    def task_ids(self) -> list[str]: ...
    @property
    def task_families(self) -> dict[str, list[str]]: ...

class OfflineEvalConfig(BaseModel):
    type: Literal["offline"] = "offline"
    tasks: dict[str, list[str]]            # {family: [metric_names]}

    @property
    def metric_names(self) -> list[str]: ...
    @property
    def task_families(self) -> dict[str, list[str]]: ...

EvalConfig = Annotated[Union[InLoopEvalConfig, OfflineEvalConfig], Discriminator("type")]
```

Add `eval: EvalConfig` as required field on `FitConfig`.

### Step 2: Add eval to `ExperimentConfig` in `olmix/aliases.py`

- Add `eval: InLoopEvalConfig` as required field on `ExperimentConfig`
- Remove `eval_tasks: list[str] | None = None` from `TrainingConfig`

### Step 3: Update `olmix/model/transformer.py`

Replace `DEFAULT_EVAL_TASKS` usage with eval config:
- `TransformerConfigBuilder.__init__` takes eval task_ids from the config
- Remove import of `DEFAULT_EVAL_TASKS`

### Step 4: Update `olmix/fit/cli.py`

Pass eval config data to `run_fit`:
- `eval_metrics=cfg.eval.metric_names`
- `task_families=cfg.eval.task_families` (new parameter)

### Step 5: Update `olmix/fit/core.py`

Add `task_families: dict[str, list[str]] | None = None` parameter to `run_fit`. Use it instead of `ALL_TASK_FAMILIES`:
- Remove `from olmix.fit.constants import ALL_TASK_FAMILIES`
- Lines 196, 206, 227: use the `task_families` parameter

### Step 6: Update `configs/fits/dclm_baseline.yaml`

Add `eval` section with `type: offline`, all 110 tasks from the DCLM CSV, and 3 task families matching CSV column names.

### Step 7: Update all 31 `configs/experiments/*.yaml`

Add `eval` section with `type: inloop` to each config. All 31 use the same 60 tasks grouped into 3 families (math: 8, code: 19, qa: 33). The WandB metric names follow the pattern `eval/downstream/{task_id} (BPB v2)`.

### Step 8: Update launch CLI (`olmix/cli.py`)

Update `launch_run` and `launch_preview` to pass eval task_ids from `experiment_config.eval.task_ids` to the training pipeline instead of relying on `DEFAULT_EVAL_TASKS`.

### Step 9: Clean up dead code

**Delete `olmix/fit/constants.py`** — all contents moved to YAML or dead:
- `ALL_TASK_FAMILIES` → now in eval config YAML
- `WandbMetrics`, `OlmoEvalMetrics`, `ALL_WANDB_METRICS`, `ALL_OLMO_EVAL_METRICS`, `ObjectiveWeights`, `BASE_METRICS_PATH` → dead code

**Clean up `olmix/model/evaluators.py`** — remove dead enums:
- Remove `DownstreamEvaluators`, `DownstreamEvaluatorsSmall`, `CodeTasks`
- Remove `DEFAULT_EVAL_TASKS` (no longer needed as default — configs are explicit)

### Step 10: Update exports and tests

**`olmix/fit/__init__.py`**: Remove old constant exports, add `InLoopEvalConfig`, `OfflineEvalConfig`.

**`olmix/model/__init__.py`**: Remove dead enum exports.

**`tests/test_fit_config.py`**: Add eval config tests (both types, discriminator, properties).

**`tests/test_imports.py`**: Remove dead import assertions, add new ones.

**`tests/test_config.py`**: Update ExperimentConfig tests to include eval section.

### Step 11: Update README

Add `eval` section to the fit config reference in the README.

## Key files

| File | Change |
|---|---|
| `olmix/fit/config.py` | Add `InLoopEvalConfig`, `OfflineEvalConfig`, `EvalConfig`; add `eval` to `FitConfig` |
| `olmix/aliases.py` | Add `eval: InLoopEvalConfig` to `ExperimentConfig`; remove `eval_tasks` from `TrainingConfig` |
| `olmix/model/transformer.py` | Use eval config task_ids instead of `DEFAULT_EVAL_TASKS` |
| `olmix/fit/cli.py` | Pass `eval_metrics` and `task_families` from config |
| `olmix/fit/core.py` | Add `task_families` param; remove `ALL_TASK_FAMILIES` import |
| `olmix/fit/constants.py` | **Delete** |
| `olmix/model/evaluators.py` | Remove all dead enums + `DEFAULT_EVAL_TASKS` |
| `olmix/cli.py` | Pass eval task_ids through launch pipeline |
| `configs/fits/dclm_baseline.yaml` | Add `eval` section (offline, 110 tasks, 3 families) |
| `configs/experiments/**/*.yaml` (31 files) | Add `eval` section (inloop, 60 tasks, 3 families) |
| `olmix/fit/__init__.py` | Swap exports |
| `olmix/model/__init__.py` | Remove dead exports |
| `tests/test_fit_config.py` | Add eval config tests |
| `tests/test_config.py` | Update ExperimentConfig tests |
| `tests/test_imports.py` | Update import assertions |
| `README.md` | Add `eval` section to config reference |

## Verification

1. `python -m pytest tests/ -x -q` — all tests pass (including unit tests that load `configs/` fixtures and verify eval config properties)
2. `ruff check olmix/` — lint clean
3. `olmix fit --config configs/fits/dclm_baseline.yaml --output-dir /tmp/test` — end-to-end fit with offline eval config
4. `olmix launch run --config configs/experiments/data_proportions/mix_baseline.yaml --dry-run` — launch with inloop eval config (no Beaker needed)
