# Plan: Update olmix for OLMo-in-loop-evals Changes

## Summary

The latest commits in OLMo-in-loop-evals (dc7e88e, c0988ee) made two key changes:
1. **Removed** 6 `basic_skills_*_bpb_5shot` tasks (RC tasks already calculate BPB)
2. **Converted** 6 science/medical tasks from `*_bpb` to `*_rc` with `compute_gold_bpb: true`

This breaks olmix because it references task names that no longer exist.

## Changes Required

### 1. `olmix/model/evaluators.py` - Update DEFAULT_EVAL_TASKS

Replace the 6 basic_skills BPB task names with RC task names:

```diff
- "basic_skills_arithmetic_bpb_5shot",
- "basic_skills_coding_bpb_5shot",
- "basic_skills_common_knowledge_bpb_5shot",
- "basic_skills_logical_reasoning_bpb_5shot",
- "basic_skills_pattern_bpb_5shot",
- "basic_skills_string_operations_bpb_5shot",
+ "basic_skills_arithmetic_rc_5shot",
+ "basic_skills_coding_rc_5shot",
+ "basic_skills_common_knowledge_rc_5shot",
+ "basic_skills_logical_reasoning_rc_5shot",
+ "basic_skills_pattern_rc_5shot",
+ "basic_skills_string_operations_rc_5shot",
```

### 2. `olmix/fit/constants.py` - Update WandbMetrics enum

Update metric names to match the new task labels:

**Basic skills (6 tasks)** - change `_bpb_5shot` to `_rc_5shot`:
```diff
- basic_skills_arithmetic_bpb = "eval/downstream/basic_skills_arithmetic_bpb_5shot (BPB v2)"
+ basic_skills_arithmetic_bpb = "eval/downstream/basic_skills_arithmetic_rc_5shot (BPB v2)"
```
(Same pattern for all 6 basic_skills tasks)

**Science/medical (6 tasks)** - change `_bpb` to `_rc`:
```diff
- lab_bench_dbqa_bpb = "eval/downstream/lab_bench_dbqa_bpb (BPB v2)"
+ lab_bench_dbqa_bpb = "eval/downstream/lab_bench_dbqa_rc (BPB v2)"
```
(Same pattern for: lab_bench_protocolqa, medmcqa, medqa_en, qasper_yesno, sciriff_yesno)

Note: `lambada_bpb` stays as-is because it wasn't converted to RC format.

## Files to Modify

| File | Change |
|------|--------|
| `olmix/model/evaluators.py` | Update 6 basic_skills task names in DEFAULT_EVAL_TASKS |
| `olmix/fit/constants.py` | Update 12 metric names in WandbMetrics enum |

## Verification

1. Run import tests:
   ```bash
   pytest tests/test_imports.py -v
   ```

2. Verify WandbMetrics enum loads correctly:
   ```bash
   python -c "from olmix.fit.constants import WandbMetrics; print(len(list(WandbMetrics)))"
   ```
   Should output `39` (no change in count, just name updates)

3. Verify DEFAULT_EVAL_TASKS references valid task labels by cross-checking with OLMo-in-loop-evals tasks.py
