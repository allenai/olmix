# Plan: Add Missing In-Loop BPB Tasks

## Goal

Add the 19 missing BPB tasks to enable full in-loop evaluation coverage, matching OlmoEvalMetrics without separate offline oe-eval jobs.

## Scope Change: Using MMLU Aggregates

**Original scope**: 88 tasks missing (including 57 individual MMLU subjects)
**Revised scope**: 19 tasks missing (using 4 MMLU aggregates instead)

## Current State

| Category | In-Loop (olmo_eval) | Offline (OlmoEvalMetrics) | Status |
|----------|---------------------|---------------------------|--------|
| Core QA | 7 | 7 | ✓ Complete |
| MMLU | 4 aggregates | 57 individual | ✓ Use aggregates |
| Math Minerva | 7 | 7 | ✓ Complete |
| Math GSM8K | 1 | 1 | ✓ Complete |
| Code | 2 | 2 | ✓ Complete |
| Multilingual code | 17 | 17 | ✓ Complete |
| Basic skills | 0 (only acc) | 6 BPB | ✗ Need to add |
| Gen tasks | 0 | 5 | ✗ Need to add |
| Science/medical | 1 (sciq only) | 8 | ✗ Need to add |

## Tasks to Add (19 total)

### Basic skills BPB (6 tasks)
- basic_skills_arithmetic
- basic_skills_coding
- basic_skills_common_knowledge
- basic_skills_logical_reasoning
- basic_skills_pattern
- basic_skills_string_operations

### Gen tasks BPB (5 tasks)
- coqa
- drop
- jeopardy
- naturalqs
- squad

### Science/medical BPB (8 tasks)
- lab_bench_dbqa
- lab_bench_protocolqa
- lambada
- medmcqa
- medqa_en
- qasper_yesno
- sciriff_yesno
- (sciq already exists as sciq_rc_0shot_bpb)

## Architecture

```
oe-eval-internal/create_in_loop_data.py → OLMo-in-loop-evals/oe_eval_tasks/ → OLMo-core DownstreamEvaluator → WandB
        (generates request data)              (task data in PyPI pkg)              (runs during training)
```

**Source Repositories**:
- **OLMo-in-loop-evals**: `git@github.com:allenai/OLMo-in-loop-evals.git` → `ai2-olmo-eval` PyPI package
- **oe-eval-internal**: `~/ai2/oe-eval-internal` → Contains create_in_loop_data.py script

## Implementation Plan

### Phase 1: Clone OLMo-in-loop-evals

```bash
cd ~/ai2
git clone git@github.com:allenai/OLMo-in-loop-evals.git
cd OLMo-in-loop-evals
git checkout -b add-19-bpb-tasks
```

### Phase 2: Generate Task Data

For each task category, run create_in_loop_data.py:

```bash
cd ~/ai2/oe-eval-internal

# Basic skills BPB (6 tasks)
python scripts/create_in_loop_data.py \
    --task basic_skills_arithmetic:rc:bpb::olmes \
           basic_skills_coding:rc:bpb::olmes \
           basic_skills_common_knowledge:rc:bpb::olmes \
           basic_skills_logical_reasoning:rc:bpb::olmes \
           basic_skills_pattern:rc:bpb::olmes \
           basic_skills_string_operations:rc:bpb::olmes \
    --task_sub_names bpb_5shot \
    --olmo-dir ~/ai2/OLMo-in-loop-evals/src/olmo_eval

# Gen tasks BPB (5 tasks) - verify oe-eval format first
python scripts/create_in_loop_data.py \
    --task coqa:gen2mc:bpb drop:gen2mc:bpb jeopardy:gen2mc:bpb naturalqs:gen2mc:bpb squad:gen2mc:bpb \
    --task_sub_names bpb_gen2mc \
    --olmo-dir ~/ai2/OLMo-in-loop-evals/src/olmo_eval

# Science/medical BPB (7 tasks, excluding sciq which exists)
python scripts/create_in_loop_data.py \
    --task lab_bench_dbqa:bpb lab_bench_protocolqa:bpb lambada:bpb medmcqa:bpb::none \
           medqa_en:bpb::none qasper_yesno:bpb::olmes sciriff_yesno:bpb::olmes \
    --task_sub_names bpb \
    --olmo-dir ~/ai2/OLMo-in-loop-evals/src/olmo_eval
```

### Phase 3: Add LABEL_TO_TASK_MAP Entries

Edit `OLMo-in-loop-evals/src/olmo_eval/tasks.py` to add entries:

```python
# Basic skills BPB (6 tasks)
"basic_skills_arithmetic_bpb_5shot": (
    OEEvalTask,
    {"dataset_path": "basic_skills_arithmetic", "dataset_name": "bpb_5shot", "metric_type": "bpb"},
),
# ... repeat for other basic_skills

# Gen tasks BPB (5 tasks)
"coqa_bpb_gen2mc": (
    OEEvalTask,
    {"dataset_path": "coqa", "dataset_name": "bpb_gen2mc", "metric_type": "bpb"},
),
# ... repeat for other gen tasks

# Science/medical BPB (7 tasks)
"lab_bench_dbqa_bpb": (
    OEEvalTask,
    {"dataset_path": "lab_bench_dbqa", "dataset_name": "bpb", "metric_type": "bpb"},
),
# ... repeat for other science/medical tasks
```

### Phase 4: Create PR for OLMo-in-loop-evals

```bash
cd ~/ai2/OLMo-in-loop-evals
git add src/olmo_eval/oe_eval_tasks/ src/olmo_eval/tasks.py
git commit -m "Add 19 new BPB tasks for basic_skills, gen, and science/medical categories"
git push -u origin add-19-bpb-tasks
# Create PR on GitHub
```

### Phase 5: Create OLMo-core branch pointing to custom in-loop-evals

Since Beaker jobs use OLMo-core (not local), need a dependency chain.

```bash
cd ~/ai2/OLMo-core
git checkout -b add-inloop-evals
```

Edit `pyproject.toml` to use git branch instead of PyPI:
```toml
# Change line 63 from:
eval = ["ai2-olmo-eval==0.8.7"]
# To:
eval = ["ai2-olmo-eval @ git+https://github.com/allenai/OLMo-in-loop-evals.git@add-19-bpb-tasks"]
```

```bash
git add pyproject.toml
git commit -m "Point eval dependency to branch with 19 new BPB tasks"
git push -u origin add-inloop-evals
```

### Phase 6: Update olmix dependencies

**File**: `~/ai2/olmix/pyproject.toml`

Update to point to OLMo-core branch:
```toml
dependencies = [
    # Change from:
    # "ai2-olmo-core @ git+https://github.com/allenai/OLMo-core.git@main",
    # To:
    "ai2-olmo-core[eval] @ git+https://github.com/allenai/OLMo-core.git@add-inloop-evals",
]
```

### Phase 7: Update olmix constants

**File**: `~/ai2/olmix/olmix/fit/constants.py`

1. Update `WandbMetrics` to add new in-loop task names
2. Update `OlmoEvalMetrics` to use 4 MMLU aggregates instead of 57 individual subjects

```python
class WandbMetrics(Enum):
    # ... existing 21 metrics ...

    # Add 19 new BPB tasks
    basic_skills_arithmetic_bpb = "eval/downstream/basic_skills_arithmetic_bpb_5shot (BPB v2)"
    # ... etc
```

## Files to Modify

| Repository | File | Changes |
|------------|------|---------|
| OLMo-in-loop-evals | `src/olmo_eval/tasks.py` | Add 19 new LABEL_TO_TASK_MAP entries |
| OLMo-in-loop-evals | `src/olmo_eval/oe_eval_tasks/` | Add 19 new task data directories |
| OLMo-core | `pyproject.toml` | Point eval dep to in-loop-evals branch |
| olmix | `pyproject.toml` | Point to OLMo-core branch with `[eval]` extra |
| olmix | `olmix/fit/constants.py` | Update WandbMetrics (add 19), simplify OlmoEvalMetrics (57→4 MMLU) |

## Verification

**Note**: Real verification requires Beaker jobs since olmo_eval has version conflicts with local olmix environment.

1. **Local test** - Verify task data files were generated:
   ```bash
   ls ~/ai2/OLMo-in-loop-evals/src/olmo_eval/oe_eval_tasks/basic_skills_arithmetic/bpb_5shot/
   # Should show: config.json, requests.jsonl.gz
   ```

2. **Update olmix dependencies** - After pushing all branches:
   ```bash
   cd ~/ai2/olmix
   # Edit pyproject.toml to point to OLMo-core branch
   pip install -e .
   ```

3. **Run olmix tests** - Verify WandbMetrics enum loads:
   ```bash
   pytest tests/test_imports.py
   ```

4. **Test via Beaker job** - The real verification:
   ```bash
   # Launch small proxy model training with new evaluators
   olmix launch configs/test/verify_tokens_1400m.yaml
   ```
   The Beaker container will:
   - Install OLMo-core from the custom branch
   - OLMo-core will install ai2-olmo-eval from the custom branch
   - The new tasks will be available for in-loop eval

5. **Check WandB** - After training job completes, verify new metrics appear:
   - `eval/downstream/basic_skills_arithmetic_bpb_5shot (BPB v2)`
   - etc.

## Risks

| Risk | Mitigation |
|------|------------|
| Some oe-eval tasks may not support BPB | Verify each task format before running create_in_loop_data.py |
| Gen tasks (gen2mc) may have different format | Check oe-eval documentation for gen2mc BPB support |

## Timeline

1. Clone and setup: Immediate
2. Generate task data: May need iteration to verify formats
3. PR to OLMo-in-loop-evals: Submit after verification
4. Update olmix: After PR merged or install from branch
