# Plan: Auto-generate Evaluation Bash Scripts on Swarm Launch

## Goal

When olmix launches a swarm, automatically generate a bash script containing `olmo-cookbook-eval evaluate` commands for all variants. No tight dependency on olmo-cookbook - user runs the script from a separate environment.

## Design

### Dashboard Naming Convention
```
-d olmix-{group_id}
```
Example: `-d olmix-8f3a9c2b`

### Generated Script Location
```
output/eval_scripts/olmix-{group_id}.sh
```

### Script Contents
```bash
#!/bin/bash
# Evaluation script for swarm: {config.name}
# Group ID: {group_id}
# Generated: {timestamp}
# Variants: {num_variants}

# Usage: Run from olmo-cookbook environment
# ./olmix-{group_id}.sh [--dry-run]

DASHBOARD="olmix-{group_id}"
TASKS="core_9task"  # Default task group
CLUSTER="aus"

# Parse args
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
fi

# Variant 0000
olmo-cookbook-eval evaluate \
    s3://ai2-llm/checkpoints/{beaker_user}/{run_name_0}/step{final_step} \
    -d "$DASHBOARD" \
    -t "$TASKS" \
    -c "$CLUSTER" \
    -v olmo_core \
    $DRY_RUN

# Variant 0001
olmo-cookbook-eval evaluate \
    ...
```

## Changes Required

### 1. Add Script Generation Function in `olmix/launch/eval_script.py` (new file)

```python
from olmix.fit.constants import OLMO3_OFFLINE_TASKS

def get_offline_eval_tasks(config: ExperimentConfig) -> list[str]:
    """Get offline eval tasks from config or default to OLMO3_OFFLINE_TASKS."""
    if config.eval_tasks is not None:
        return config.eval_tasks
    return OLMO3_OFFLINE_TASKS

def generate_eval_script(
    group: ExperimentGroup,
    beaker_user: str,
    output_dir: str = "output/eval_scripts",
) -> str:
    """Generate bash script for running olmo-cookbook-eval on all swarm variants."""
```

Key logic:
- Get offline tasks from config or default to `OLMO3_OFFLINE_TASKS` (108 tasks)
- Predict checkpoint paths from run names and cluster
- Write executable bash script with all tasks
- Return path to generated script

### 2. Hook into Launch Workflow in `olmix/launch/beaker.py`

After `mk_launch_configs()` builds configs, call script generator:

```python
from olmix.launch.eval_script import generate_eval_script

def mk_launch_configs(...):
    # ... existing code ...

    # Generate eval script
    script_path = generate_eval_script(group, beaker_user)
    print(f"Eval script generated: {script_path}")

    return configs
```

### 3. Add Offline BPB Task List to `olmix/fit/constants.py`

There are two types of BPB metrics:
1. **In-loop BPB** (42 metrics) - `eval/downstream/...` paths logged to WandB during training
2. **Offline BPB tasks** (22 tasks) - task names with `:bpb:` for olmo-cookbook-eval

Add the **offline BPB tasks** for the eval script (these are the ones that produce BPB metrics):

```python
# In olmix/fit/constants.py - add this list
OFFLINE_BPB_TASKS = [
    # Code BPB
    "codex_humaneval:3shot:bpb::none",
    "mbpp:3shot:bpb::none",
    # Minerva math BPB (7)
    "minerva_math_algebra:bpb::olmes",
    "minerva_math_counting_and_probability:bpb::olmes",
    "minerva_math_geometry:bpb::olmes",
    "minerva_math_intermediate_algebra:bpb::olmes",
    "minerva_math_number_theory:bpb::olmes",
    "minerva_math_prealgebra:bpb::olmes",
    "minerva_math_precalculus:bpb::olmes",
    # Gen tasks BPB (5)
    "coqa:rc:bpb::gen2mc",
    "drop:rc:bpb::gen2mc",
    "jeopardy:rc:bpb::gen2mc",
    "naturalqs:rc:bpb::gen2mc",
    "squad:rc:bpb::gen2mc",
    # Other BPB
    "medmcqa:rc:bpb::none",
    "sciq:bpb::olmo1",
]
```

**Note:** The in-loop BPB metrics (MMLU, ARC, hellaswag, etc.) are already computed during training and logged to WandB. These 17 offline tasks supplement the in-loop metrics for additional coverage.

### 4. Add Config Options for Eval Script in `ExperimentConfig`

Add these new fields to `olmix/aliases.py`:

```python
# In ExperimentConfig class
eval_tasks: list[str] | None = None  # If None, uses OFFLINE_BPB_TASKS
eval_cluster: str = "aus"            # Cluster for running evals
eval_model_backend: str = "olmo_core"  # Model backend (hf, vllm, olmo_core)
```

### 5. Document Eval Tasks in Example Configs

**BPB Metrics:** olmo-cookbook-eval computes BPB (bits-per-byte) metrics by default via `--compute-gold-bpb` flag (defaults to True). No special flag needed.

Update `configs/examples/mixture_olmo2_30m.yaml`:

```yaml
# Offline evaluation settings for mixing optimization
# Default uses OLMO3_OFFLINE_TASKS (108 tasks) from olmix/fit/constants.py
# which includes: core tasks, all 57 MMLU subjects, minerva math, basic skills,
# multilingual code (17 languages), gen tasks, and more.
#
# BPB metrics are computed automatically by olmo-cookbook-eval (--compute-gold-bpb=True by default)
#
# Override with specific tasks if needed:
# eval_tasks:
#   - arc_challenge:rc::olmes:full
#   - mmlu_abstract_algebra:rc::olmes
eval_cluster: aus
eval_model_backend: olmo_core
```

## Checkpoint Path Prediction

Based on cluster and config:

| Cluster Pattern | Checkpoint Root |
|-----------------|-----------------|
| `augusta` | `gs://ai2-llm/checkpoints/{user}/{run_name}` |
| `jupiter`, `saturn` (weka=True) | `/weka/oe-training-default/ai2-llm/checkpoints/{user}/{run_name}` |
| Default (S3) | `s3://ai2-llm/checkpoints/{user}/{run_name}` |

Run name format: `{config.name}-{group_id}-{variant_idx:04d}`

**Step Detection:** At runtime, script auto-detects the latest checkpoint step in the directory using `aws s3 ls` or `gsutil ls`.

## Files to Modify

| File | Changes |
|------|---------|
| `olmix/fit/constants.py` | Add `OLMO3_OFFLINE_TASKS` list (108 tasks from regmixer) |
| `olmix/launch/eval_script.py` | New file - script generation logic |
| `olmix/launch/beaker.py` | Hook to generate script after building configs |
| `olmix/aliases.py` | Add `eval_tasks`, `eval_cluster`, `eval_model_backend` config fields |
| `configs/examples/mixture_olmo2_30m.yaml` | Add eval settings + document available task groups |
| `plans/007_eval_script_generation.md` | Copy of this plan for project documentation |

## Example Output

After running:
```bash
olmix launch run --config configs/test/verify_tokens_140m.yaml
```

Generated script at `output/eval_scripts/olmix-58362402.sh`:
```bash
#!/bin/bash
# Evaluation script for swarm: verify-tokens-140m
# Group ID: 58362402
# Generated: 2026-01-31T12:00:00
# Variants: 1

set -e

DASHBOARD="olmix-58362402"
# 108 offline tasks from OLMO3_OFFLINE_TASKS (truncated for readability)
# Full list includes: core (7), basic_skills (6), code (2), minerva_math (7),
# mmlu (57), mt_mbpp_v2fix (17), gen (5), other (5)
TASKS=(
    # Core
    "arc_challenge:rc::olmes:full"
    "arc_easy:rc::olmes:full"
    "csqa:rc::olmes:full"
    "hellaswag:rc::olmes:full"
    "piqa:rc::olmes:full"
    "socialiqa:rc::olmes:full"
    "winogrande:rc::olmes:full"
    # Basic skills
    "basic_skills_arithmetic:rc::olmes"
    "basic_skills_coding:rc::olmes"
    # ... (6 total)
    # Code
    "codex_humaneval:3shot:bpb::none"
    "mbpp:3shot:bpb::none"
    # Minerva math
    "minerva_math_algebra::olmes"
    # ... (7 total)
    # MMLU (57 subjects)
    "mmlu_abstract_algebra:rc::olmes"
    "mmlu_anatomy:rc::olmes"
    # ... (57 total)
    # Multilingual code
    "mt_mbpp_v2fix:python"
    # ... (17 languages)
    # Gen tasks
    "coqa:rc::gen2mc"
    "drop:rc::gen2mc"
    # ... (5 total)
    # Other
    "lambada"
    "medmcqa:rc::none"
    "sciq:rc::olmo3"
    "ultrachat_masked_ppl"
    "wildchat_masked_ppl"
)
CLUSTER="aus"
MODEL_BACKEND="olmo_core"

DRY_RUN=""
[[ "$1" == "--dry-run" ]] && DRY_RUN="--dry-run"

# Helper: find latest step in S3 checkpoint directory
find_latest_step() {
    local checkpoint_dir="$1"
    aws s3 ls "${checkpoint_dir}/" | grep -oP 'step\d+' | sort -t'p' -k2 -n | tail -1
}

# Variant 0000: verify-tokens-140m-58362402-0000
CKPT_DIR_0="s3://ai2-llm/checkpoints/kylel/verify-tokens-140m-58362402-0000"
LATEST_STEP_0=$(find_latest_step "$CKPT_DIR_0")
echo "Evaluating variant 0000: $CKPT_DIR_0/$LATEST_STEP_0"

# Build task args
TASK_ARGS=""
for t in "${TASKS[@]}"; do TASK_ARGS="$TASK_ARGS -t $t"; done

olmo-cookbook-eval evaluate \
    "$CKPT_DIR_0/$LATEST_STEP_0" \
    -d "$DASHBOARD" \
    $TASK_ARGS \
    -c "$CLUSTER" \
    -v "$MODEL_BACKEND" \
    $DRY_RUN

echo "Done! Results will be at: s3://ai2-llm/evaluation/$DASHBOARD/"
```

## Usage Workflow

1. Launch swarm from olmix environment:
   ```bash
   olmix launch run --config my_swarm.yaml
   # Output: Eval script generated: output/eval_scripts/olmix-abc12345.sh
   ```

2. Wait for training to complete (monitor in WandB)

3. Switch to olmo-cookbook environment and run evaluations:
   ```bash
   cd ~/ai2/olmo-cookbook
   source .venv/bin/activate
   bash ~/ai2/olmix/output/eval_scripts/olmix-abc12345.sh
   ```
   (Script auto-detects latest checkpoint step)

4. Pull results in olmix fit:
   ```bash
   olmix fit fit -g abc12345 --dashboard olmix-abc12345 --pull-from-dashboard
   ```

## Verification

1. Launch a test swarm and verify script is generated
2. Check script has correct checkpoint paths
3. Run with `--dry-run` to verify olmo-cookbook-eval command syntax
4. After training completes, run actual evals and verify results appear in dashboard
