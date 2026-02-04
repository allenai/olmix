# Plan: Align WandbMetrics with olmo-core's FULL_TASKS_SMALL_COMPUTE

## Context

- OLMo-core's `FULL_TASKS_SMALL_COMPUTE` defines 60 tasks (excluding copycolors sanity check)
- olmix's `WandbMetrics` currently has 39 metrics
- olmix's `DEFAULT_EVAL_TASKS` currently has 27 tasks
- Goal: Align all to use the same 60 tasks

## Pre-requisite

Reinstall to get the updated olmo-core with task_groups.py:
```bash
cd /Users/kylel/ai2/olmix
pip install -e .
```

(The `[eval]` extra is already specified in pyproject.toml's ai2-olmo-core dependency)

## Files to Modify

| File | Change |
|------|--------|
| `olmix/fit/constants.py` | Update WandbMetrics enum to 60 tasks |
| `olmix/model/evaluators.py` | Import DEFAULT_EVAL_TASKS from olmo-core |

## Full WandbMetrics Mapping (60 tasks)

```python
class WandbMetrics(Enum):
    """In-loop BPB metrics logged to WandB during training.

    Maps to olmo-core's FULL_TASKS_SMALL_COMPUTE (60 tasks, excluding copycolors).
    """

    # Core QA RC (7)
    arc_challenge_bpb = "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"
    arc_easy_bpb = "eval/downstream/arc_easy_test_rc_5shot (BPB v2)"
    hellaswag_bpb = "eval/downstream/hellaswag_rc_5shot (BPB v2)"
    winogrande_bpb = "eval/downstream/winogrande_val_rc_5shot (BPB v2)"
    csqa_bpb = "eval/downstream/csqa_val_rc_5shot (BPB v2)"
    piqa_bpb = "eval/downstream/piqa_val_rc_5shot (BPB v2)"
    socialiqa_bpb = "eval/downstream/socialiqa_val_rc_5shot (BPB v2)"

    # MMLU Val RC (4)
    mmlu_stem_val_bpb = "eval/downstream/mmlu_stem_val_rc_5shot (BPB v2)"
    mmlu_humanities_val_bpb = "eval/downstream/mmlu_humanities_val_rc_5shot (BPB v2)"
    mmlu_social_sciences_val_bpb = "eval/downstream/mmlu_social_sciences_val_rc_5shot (BPB v2)"
    mmlu_other_val_bpb = "eval/downstream/mmlu_other_val_rc_5shot (BPB v2)"

    # MMLU Test RC (4)
    mmlu_stem_bpb = "eval/downstream/mmlu_stem_test_rc_5shot (BPB v2)"
    mmlu_humanities_bpb = "eval/downstream/mmlu_humanities_test_rc_5shot (BPB v2)"
    mmlu_social_sciences_bpb = "eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB v2)"
    mmlu_other_bpb = "eval/downstream/mmlu_other_test_rc_5shot (BPB v2)"

    # Math - GSM8K (1)
    gsm8k_bpb = "eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)"

    # Math - Minerva (7)
    minerva_algebra_bpb = "eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB v2)"
    minerva_counting_bpb = "eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB v2)"
    minerva_geometry_bpb = "eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB v2)"
    minerva_intermediate_algebra_bpb = "eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB v2)"
    minerva_number_theory_bpb = "eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB v2)"
    minerva_prealgebra_bpb = "eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB v2)"
    minerva_precalculus_bpb = "eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB v2)"

    # Code (2) - NOTE: 3shot not 0shot
    codex_humaneval_bpb = "eval/downstream/codex_humaneval_gold_bpb_3shot (BPB v2)"
    codex_mbpp_bpb = "eval/downstream/codex_mbpp_gold_bpb_3shot (BPB v2)"

    # Generative QA BPB (6)
    coqa_bpb = "eval/downstream/coqa_bpb_0shot (BPB v2)"
    drop_bpb = "eval/downstream/drop_bpb_5shot (BPB v2)"
    jeopardy_bpb = "eval/downstream/jeopardy_bpb_5shot (BPB v2)"
    lambada_bpb = "eval/downstream/lambada_bpb_0shot (BPB v2)"
    naturalqs_bpb = "eval/downstream/naturalqs_bpb_5shot (BPB v2)"
    squad_bpb = "eval/downstream/squad_bpb_5shot (BPB v2)"

    # MT MBPP - All 17 languages (17)
    mt_mbpp_bash_bpb = "eval/downstream/mt_mbpp_bash_gold_bpb_3shot (BPB v2)"
    mt_mbpp_c_bpb = "eval/downstream/mt_mbpp_c_gold_bpb_3shot (BPB v2)"
    mt_mbpp_cpp_bpb = "eval/downstream/mt_mbpp_cpp_gold_bpb_3shot (BPB v2)"
    mt_mbpp_csharp_bpb = "eval/downstream/mt_mbpp_csharp_gold_bpb_3shot (BPB v2)"
    mt_mbpp_go_bpb = "eval/downstream/mt_mbpp_go_gold_bpb_3shot (BPB v2)"
    mt_mbpp_haskell_bpb = "eval/downstream/mt_mbpp_haskell_gold_bpb_3shot (BPB v2)"
    mt_mbpp_java_bpb = "eval/downstream/mt_mbpp_java_gold_bpb_3shot (BPB v2)"
    mt_mbpp_javascript_bpb = "eval/downstream/mt_mbpp_javascript_gold_bpb_3shot (BPB v2)"
    mt_mbpp_matlab_bpb = "eval/downstream/mt_mbpp_matlab_gold_bpb_3shot (BPB v2)"
    mt_mbpp_php_bpb = "eval/downstream/mt_mbpp_php_gold_bpb_3shot (BPB v2)"
    mt_mbpp_python_bpb = "eval/downstream/mt_mbpp_python_gold_bpb_3shot (BPB v2)"
    mt_mbpp_r_bpb = "eval/downstream/mt_mbpp_r_gold_bpb_3shot (BPB v2)"
    mt_mbpp_ruby_bpb = "eval/downstream/mt_mbpp_ruby_gold_bpb_3shot (BPB v2)"
    mt_mbpp_rust_bpb = "eval/downstream/mt_mbpp_rust_gold_bpb_3shot (BPB v2)"
    mt_mbpp_scala_bpb = "eval/downstream/mt_mbpp_scala_gold_bpb_3shot (BPB v2)"
    mt_mbpp_swift_bpb = "eval/downstream/mt_mbpp_swift_gold_bpb_3shot (BPB v2)"
    mt_mbpp_typescript_bpb = "eval/downstream/mt_mbpp_typescript_gold_bpb_3shot (BPB v2)"

    # Basic Skills RC (6)
    basic_skills_arithmetic_bpb = "eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2)"
    basic_skills_coding_bpb = "eval/downstream/basic_skills_coding_rc_5shot (BPB v2)"
    basic_skills_common_knowledge_bpb = "eval/downstream/basic_skills_common_knowledge_rc_5shot (BPB v2)"
    basic_skills_logical_reasoning_bpb = "eval/downstream/basic_skills_logical_reasoning_rc_5shot (BPB v2)"
    basic_skills_pattern_bpb = "eval/downstream/basic_skills_pattern_rc_5shot (BPB v2)"
    basic_skills_string_operations_bpb = "eval/downstream/basic_skills_string_operations_rc_5shot (BPB v2)"

    # Science/Medical RC (6)
    lab_bench_dbqa_bpb = "eval/downstream/lab_bench_dbqa_rc_3shot (BPB v2)"
    lab_bench_protocolqa_bpb = "eval/downstream/lab_bench_protocolqa_rc_3shot (BPB v2)"
    medmcqa_bpb = "eval/downstream/medmcqa_rc_5shot (BPB v2)"
    medqa_en_bpb = "eval/downstream/medqa_en_rc_5shot (BPB v2)"
    qasper_yesno_bpb = "eval/downstream/qasper_yesno_rc_5shot (BPB v2)"
    sciriff_yesno_bpb = "eval/downstream/sciriff_yesno_rc_5shot (BPB v2)"
```

**Total: 60 tasks** (7 + 4 + 4 + 1 + 7 + 2 + 6 + 17 + 6 + 6)

## Changes from Current WandbMetrics (39 → 60)

### Tasks to ADD (21 new):
- MMLU val (4): mmlu_*_val_rc_5shot
- MT MBPP (14 new langs): bash, c, csharp, go, haskell, javascript, matlab, php, python, r, ruby, scala, swift, typescript
- Gen QA shot changes (3): coqa_bpb_0shot (was 5shot), lambada_bpb_0shot (was bpb)

### Tasks to UPDATE:
- codex_humaneval: 0shot → 3shot
- codex_mbpp: 0shot → 3shot
- lab_bench_dbqa: rc → rc_3shot
- lab_bench_protocolqa: rc → rc_3shot
- medmcqa: rc → rc_5shot
- medqa_en: rc → rc_5shot
- qasper_yesno: rc → rc_5shot
- sciriff_yesno: rc → rc_5shot

## olmix/model/evaluators.py Changes

```python
"""Evaluator configurations for olmix experiments."""

from enum import Enum

from olmo_core.eval.task_groups import FULL_TASKS_SMALL_COMPUTE

# Use olmo-core's task list, excluding copycolors sanity check (60 tasks)
DEFAULT_EVAL_TASKS: list[str] = [t for t in FULL_TASKS_SMALL_COMPUTE if "copycolors" not in t]


class CodeTasks(Enum):
    """Enum of code-related evaluation tasks."""
    BASIC_SKILLS_CODING_RC_5SHOT = "basic_skills_coding_rc_5shot"
    CODEX_HUMANEVAL = "codex_humaneval_gold_bpb_3shot"  # Changed from 0shot
    CODEX_MBPP = "codex_mbpp_gold_bpb_3shot"  # Changed from 0shot
```

## Verification

1. **Reinstall dependencies**:
   ```bash
   pip install -e .
   ```

2. **Verify task count from olmo-core**:
   ```bash
   python -c "from olmo_core.eval.task_groups import FULL_TASKS_SMALL_COMPUTE; print(len(FULL_TASKS_SMALL_COMPUTE))"
   ```
   Should output: `61` (includes copycolors)

3. **Verify DEFAULT_EVAL_TASKS count**:
   ```bash
   python -c "from olmix.model.evaluators import DEFAULT_EVAL_TASKS; print(len(DEFAULT_EVAL_TASKS))"
   ```
   Should output: `60` (excludes copycolors)

4. **Verify WandbMetrics count**:
   ```bash
   python -c "from olmix.fit.constants import WandbMetrics; print(len(list(WandbMetrics)))"
   ```
   Should output: `60`

5. **Import tests**:
   ```bash
   pytest tests/test_imports.py -v
   ```

6. **Test run with all 60 tasks**:
   ```bash
   yes | olmix launch run --config configs/experiments/training_duration/duration_0.5x.yaml
   ```
   Verify all 60 BPB v2 metrics appear in WandB.
