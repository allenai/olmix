# Plan: Clean Up constants.py Metrics

## Goal

Simplify `olmix/fit/constants.py` from 2124 lines to ~200 lines with clear separation:
1. **WandB metrics** - in-loop BPB metrics logged during training (21)
2. **Offline eval metrics** - task names for olmo-cookbook-eval (109)

## Current State (Problems)

| Class | Members | Issues |
|-------|---------|--------|
| `WandbMetrics` | 42 | Duplicates (21 v1 + 21 v2) |
| `GroupedWandbMetrics` | 37 | Mixes offline task names; only default used |
| `ObjectiveWeights` | 13 | Completely unused |

## New Structure

### 1. `WandbMetrics` - Keep only v2 (21 metrics)

```python
class WandbMetrics(Enum):
    """In-loop BPB metrics logged to WandB during training."""
    # Core (7)
    arc_challenge_bpb = "eval/downstream/arc_challenge_test_rc_5shot (BPB v2)"
    arc_easy_bpb = "eval/downstream/arc_easy_test_rc_5shot (BPB v2)"
    csqa_bpb = "eval/downstream/csqa_val_rc_5shot (BPB v2)"
    hellaswag_bpb = "eval/downstream/hellaswag_rc_5shot (BPB v2)"
    piqa_bpb = "eval/downstream/piqa_val_rc_5shot (BPB v2)"
    socialiqa_bpb = "eval/downstream/socialiqa_val_rc_5shot (BPB v2)"
    winogrande_bpb = "eval/downstream/winogrande_val_rc_5shot (BPB v2)"

    # MMLU (4)
    mmlu_humanities_bpb = "eval/downstream/mmlu_humanities_test_rc_5shot (BPB v2)"
    mmlu_other_bpb = "eval/downstream/mmlu_other_test_rc_5shot (BPB v2)"
    mmlu_social_sciences_bpb = "eval/downstream/mmlu_social_sciences_test_rc_5shot (BPB v2)"
    mmlu_stem_bpb = "eval/downstream/mmlu_stem_test_rc_5shot (BPB v2)"

    # Math (8)
    minerva_algebra_bpb = "eval/downstream/minerva_math_algebra_gold_bpb_0shot (BPB v2)"
    minerva_counting_bpb = "eval/downstream/minerva_math_counting_and_probability_gold_bpb_0shot (BPB v2)"
    minerva_geometry_bpb = "eval/downstream/minerva_math_geometry_gold_bpb_0shot (BPB v2)"
    minerva_intermediate_algebra_bpb = "eval/downstream/minerva_math_intermediate_algebra_gold_bpb_0shot (BPB v2)"
    minerva_number_theory_bpb = "eval/downstream/minerva_math_number_theory_gold_bpb_0shot (BPB v2)"
    minerva_prealgebra_bpb = "eval/downstream/minerva_math_prealgebra_gold_bpb_0shot (BPB v2)"
    minerva_precalculus_bpb = "eval/downstream/minerva_math_precalculus_gold_bpb_0shot (BPB v2)"
    gsm8k_bpb = "eval/downstream/gsm8k_gold_bpb_5shot (BPB v2)"

    # Code (2)
    codex_humaneval_bpb = "eval/downstream/codex_humaneval_gold_bpb_0shot (BPB v2)"
    codex_mbpp_bpb = "eval/downstream/codex_mbpp_gold_bpb_0shot (BPB v2)"


ALL_WANDB_METRICS = [m.value for m in WandbMetrics]
```

### 2. Remove `GroupedWandbMetrics` entirely

Replace usages with `ALL_WANDB_METRICS` list.
Remove CLI options `--group-average` and `--group-metrics`.

### 3. `OlmoEvalMetrics` - Full list from olmo3:dev:1b:bpb (109 tasks)

```python
class OlmoEvalMetrics(Enum):
    """Offline BPB metrics for olmo-cookbook-eval.

    Source: olmo3:dev:1b:bpb task group in olmo-cookbook (109 tasks).
    """
    # ARC (2)
    arc_challenge_bpb = "arc_challenge:bpb::olmes:full"
    arc_easy_bpb = "arc_easy:bpb::olmes:full"

    # Basic skills (6)
    basic_skills_arithmetic = "basic_skills_arithmetic:bpb::olmes"
    basic_skills_coding = "basic_skills_coding:bpb::olmes"
    basic_skills_common_knowledge = "basic_skills_common_knowledge:bpb::olmes"
    basic_skills_logical_reasoning = "basic_skills_logical_reasoning:bpb::olmes"
    basic_skills_pattern = "basic_skills_pattern:bpb::olmes"
    basic_skills_string_operations = "basic_skills_string_operations:bpb::olmes"

    # Code (2)
    codex_humaneval = "codex_humaneval:3shot:bpb::none"
    mbpp = "mbpp:3shot:bpb::none"

    # Core QA (5)
    csqa = "csqa:bpb::olmes:full"
    hellaswag = "hellaswag:bpb::olmes:full"
    piqa = "piqa:bpb::olmes:full"
    socialiqa = "socialiqa:bpb::olmes:full"
    winogrande = "winogrande:bpb::olmes:full"

    # Gen tasks (5)
    coqa = "coqa:bpb::gen2mc"
    drop = "drop:bpb::gen2mc"
    jeopardy = "jeopardy:bpb::gen2mc"
    naturalqs = "naturalqs:bpb::gen2mc"
    squad = "squad:bpb::gen2mc"

    # Science/medical (7)
    lab_bench_dbqa = "lab_bench_dbqa:bpb"
    lab_bench_protocolqa = "lab_bench_protocolqa:bpb"
    lambada = "lambada:bpb"
    medmcqa = "medmcqa:bpb::none"
    medqa_en = "medqa_en:bpb::none"
    qasper_yesno = "qasper_yesno:bpb::olmes"
    sciq = "sciq:bpb::olmo3"
    sciriff_yesno = "sciriff_yesno:bpb::olmes"

    # Math - Minerva (7)
    minerva_algebra = "minerva_math_algebra:bpb::olmes"
    minerva_counting = "minerva_math_counting_and_probability:bpb::olmes"
    minerva_geometry = "minerva_math_geometry:bpb::olmes"
    minerva_intermediate_algebra = "minerva_math_intermediate_algebra:bpb::olmes"
    minerva_number_theory = "minerva_math_number_theory:bpb::olmes"
    minerva_prealgebra = "minerva_math_prealgebra:bpb::olmes"
    minerva_precalculus = "minerva_math_precalculus:bpb::olmes"

    # MMLU (57)
    mmlu_abstract_algebra = "mmlu_abstract_algebra:bpb::olmes"
    mmlu_anatomy = "mmlu_anatomy:bpb::olmes"
    mmlu_astronomy = "mmlu_astronomy:bpb::olmes"
    mmlu_business_ethics = "mmlu_business_ethics:bpb::olmes"
    mmlu_clinical_knowledge = "mmlu_clinical_knowledge:bpb::olmes"
    mmlu_college_biology = "mmlu_college_biology:bpb::olmes"
    mmlu_college_chemistry = "mmlu_college_chemistry:bpb::olmes"
    mmlu_college_computer_science = "mmlu_college_computer_science:bpb::olmes"
    mmlu_college_mathematics = "mmlu_college_mathematics:bpb::olmes"
    mmlu_college_medicine = "mmlu_college_medicine:bpb::olmes"
    mmlu_college_physics = "mmlu_college_physics:bpb::olmes"
    mmlu_computer_security = "mmlu_computer_security:bpb::olmes"
    mmlu_conceptual_physics = "mmlu_conceptual_physics:bpb::olmes"
    mmlu_econometrics = "mmlu_econometrics:bpb::olmes"
    mmlu_electrical_engineering = "mmlu_electrical_engineering:bpb::olmes"
    mmlu_elementary_mathematics = "mmlu_elementary_mathematics:bpb::olmes"
    mmlu_formal_logic = "mmlu_formal_logic:bpb::olmes"
    mmlu_global_facts = "mmlu_global_facts:bpb::olmes"
    mmlu_high_school_biology = "mmlu_high_school_biology:bpb::olmes"
    mmlu_high_school_chemistry = "mmlu_high_school_chemistry:bpb::olmes"
    mmlu_high_school_computer_science = "mmlu_high_school_computer_science:bpb::olmes"
    mmlu_high_school_european_history = "mmlu_high_school_european_history:bpb::olmes"
    mmlu_high_school_geography = "mmlu_high_school_geography:bpb::olmes"
    mmlu_high_school_government_and_politics = "mmlu_high_school_government_and_politics:bpb::olmes"
    mmlu_high_school_macroeconomics = "mmlu_high_school_macroeconomics:bpb::olmes"
    mmlu_high_school_mathematics = "mmlu_high_school_mathematics:bpb::olmes"
    mmlu_high_school_microeconomics = "mmlu_high_school_microeconomics:bpb::olmes"
    mmlu_high_school_physics = "mmlu_high_school_physics:bpb::olmes"
    mmlu_high_school_psychology = "mmlu_high_school_psychology:bpb::olmes"
    mmlu_high_school_statistics = "mmlu_high_school_statistics:bpb::olmes"
    mmlu_high_school_us_history = "mmlu_high_school_us_history:bpb::olmes"
    mmlu_high_school_world_history = "mmlu_high_school_world_history:bpb::olmes"
    mmlu_human_aging = "mmlu_human_aging:bpb::olmes"
    mmlu_human_sexuality = "mmlu_human_sexuality:bpb::olmes"
    mmlu_international_law = "mmlu_international_law:bpb::olmes"
    mmlu_jurisprudence = "mmlu_jurisprudence:bpb::olmes"
    mmlu_logical_fallacies = "mmlu_logical_fallacies:bpb::olmes"
    mmlu_machine_learning = "mmlu_machine_learning:bpb::olmes"
    mmlu_management = "mmlu_management:bpb::olmes"
    mmlu_marketing = "mmlu_marketing:bpb::olmes"
    mmlu_medical_genetics = "mmlu_medical_genetics:bpb::olmes"
    mmlu_miscellaneous = "mmlu_miscellaneous:bpb::olmes"
    mmlu_moral_disputes = "mmlu_moral_disputes:bpb::olmes"
    mmlu_moral_scenarios = "mmlu_moral_scenarios:bpb::olmes"
    mmlu_nutrition = "mmlu_nutrition:bpb::olmes"
    mmlu_philosophy = "mmlu_philosophy:bpb::olmes"
    mmlu_prehistory = "mmlu_prehistory:bpb::olmes"
    mmlu_professional_accounting = "mmlu_professional_accounting:bpb::olmes"
    mmlu_professional_law = "mmlu_professional_law:bpb::olmes"
    mmlu_professional_medicine = "mmlu_professional_medicine:bpb::olmes"
    mmlu_professional_psychology = "mmlu_professional_psychology:bpb::olmes"
    mmlu_public_relations = "mmlu_public_relations:bpb::olmes"
    mmlu_security_studies = "mmlu_security_studies:bpb::olmes"
    mmlu_sociology = "mmlu_sociology:bpb::olmes"
    mmlu_us_foreign_policy = "mmlu_us_foreign_policy:bpb::olmes"
    mmlu_virology = "mmlu_virology:bpb::olmes"
    mmlu_world_religions = "mmlu_world_religions:bpb::olmes"

    # Multilingual code (17)
    mt_mbpp_bash = "mt_mbpp_v2fix:bash"
    mt_mbpp_c = "mt_mbpp_v2fix:c"
    mt_mbpp_cpp = "mt_mbpp_v2fix:cpp"
    mt_mbpp_csharp = "mt_mbpp_v2fix:csharp"
    mt_mbpp_go = "mt_mbpp_v2fix:go"
    mt_mbpp_haskell = "mt_mbpp_v2fix:haskell"
    mt_mbpp_java = "mt_mbpp_v2fix:java"
    mt_mbpp_javascript = "mt_mbpp_v2fix:javascript"
    mt_mbpp_matlab = "mt_mbpp_v2fix:matlab"
    mt_mbpp_php = "mt_mbpp_v2fix:php"
    mt_mbpp_python = "mt_mbpp_v2fix:python"
    mt_mbpp_r = "mt_mbpp_v2fix:r"
    mt_mbpp_ruby = "mt_mbpp_v2fix:ruby"
    mt_mbpp_rust = "mt_mbpp_v2fix:rust"
    mt_mbpp_scala = "mt_mbpp_v2fix:scala"
    mt_mbpp_swift = "mt_mbpp_v2fix:swift"
    mt_mbpp_typescript = "mt_mbpp_v2fix:typescript"


```

### 4. Remove entirely

- `GroupedWandbMetrics` (37 groups)
- `ObjectiveWeights` (13 weighting schemes)
- CLI options `--group-average` and `--group-metrics`

## Files to Modify

| File | Changes |
|------|---------|
| `olmix/fit/constants.py` | Complete rewrite: 2124 → ~200 lines |
| `olmix/fit/cli.py` | Remove `GroupedWandbMetrics`, use `ALL_WANDB_METRICS`; remove `--group-average`/`--group-metrics` |
| `olmix/fit/utils.py` | Change type hints from `GroupedWandbMetrics` to `list[str]` |
| `olmix/fit/__init__.py` | Update exports |
| `tests/test_imports.py` | Update to test new structure |

## Verification

### 1. Import test
```bash
python -c "from olmix.fit.constants import WandbMetrics, OlmoEvalMetrics, ALL_WANDB_METRICS"
```

### 2. CLI test
```bash
olmix fit --help
```

### 3. Unit tests
```bash
pytest tests/test_imports.py -v
```

### 4. Validate OlmoEvalMetrics against olmo-cookbook

Generate a test script that verifies each task name is valid with olmo-cookbook-eval:

```bash
#!/bin/bash
# validate_olmo_eval_tasks.sh
# Run from olmo-cookbook environment

set -e

# Test checkpoint (use any existing checkpoint)
CHECKPOINT="s3://ai2-llm/checkpoints/test/step0"

# Each task from OlmoEvalMetrics
TASKS=(
    "arc_challenge:bpb::olmes:full"
    "arc_easy:bpb::olmes:full"
    # ... all 109 tasks
)

echo "Validating ${#TASKS[@]} tasks with olmo-cookbook-eval --dry-run..."

for task in "${TASKS[@]}"; do
    echo -n "  $task: "
    if olmo-cookbook-eval evaluate "$CHECKPOINT" -t "$task" --dry-run 2>/dev/null; then
        echo "OK"
    else
        echo "FAILED"
        exit 1
    fi
done

echo "All tasks validated!"
```

This script should be generated as part of the implementation and run against olmo-cookbook to verify all task names are compatible.
