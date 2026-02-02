# Plan: Enable In-Loop Evaluations in olmix Trainer

## Goal

Add `DownstreamEvaluatorCallbackConfig` to olmix's `TransformerConfigBuilder` so that BPB metrics are logged to WandB during training.

## Background: Why olmix is Different from olmo-core

### What olmix Reuses from olmo-core (directly)
- `TransformerConfig` - Model architecture
- `TokenizerConfig` - Tokenizer configuration
- `TrainerConfig` - Base trainer config
- Callbacks: `CheckpointerCallback`, `ConfigSaverCallback`, `GPUMemoryMonitorCallback`, `ProfilerCallback`, `WandBCallback`
- Data components: `NumpyDataLoaderConfig`, `NumpyFSLDatasetConfig`, `SourceMixtureConfig`
- Optimizer: `SkipStepAdamWConfig`, schedulers

### What olmix Customizes
- **Chinchilla scaling laws**: Custom batch size, learning rate, duration calculations
- **Multi-period scheduler**: WSDS scheduler aligned with Chinchilla training phases
- **Source mixture handling**: Inline logic + cloud glob expansion
- **Callback configuration**: Only includes 5 callbacks (missing evaluators)

### Why the Evaluator Callback is Missing
olmix's `TransformerConfigBuilder.build_callbacks()` only returns:
```python
{
    "gpu_monitor": GPUMemoryMonitorCallback(),
    "config_saver": ConfigSaverCallback(),
    "profiler": ProfilerCallback(enabled=self.profile),
    "checkpointer": CheckpointerCallback(...),
    "wandb": WandBCallback(...),
}
```

**Missing**: `DownstreamEvaluatorCallbackConfig` which is available in olmo-core but not wired into olmix.

### Evidence: regmixer vs olmix
- **regmixer** runs have `downstream_evaluator` callback with 70+ tasks and log BPB metrics
- **olmix** runs (including verify_tokens from plan 006) have 0 BPB metrics

## Implementation Summary

### Files Modified

| File | Changes |
|------|---------|
| `olmix/model/transformer.py` | Import DownstreamEvaluatorCallbackConfig, add eval_tasks/eval_interval params, update build_callbacks() |
| `olmix/model/evaluators.py` | Add DEFAULT_EVAL_TASKS list (39 BPB tasks) |
| `olmix/launch/train.py` | Add --eval-task, --eval-interval, --no-eval CLI options |

### DEFAULT_EVAL_TASKS (39 tasks)

```python
DEFAULT_EVAL_TASKS = [
    # Core QA BPB (7)
    "arc_challenge_test_rc_5shot",
    "arc_easy_test_rc_5shot",
    "csqa_val_rc_5shot",
    "hellaswag_rc_5shot",
    "piqa_val_rc_5shot",
    "socialiqa_val_rc_5shot",
    "winogrande_val_rc_5shot",
    # MMLU BPB (4)
    "mmlu_humanities_test_rc_5shot",
    "mmlu_other_test_rc_5shot",
    "mmlu_social_sciences_test_rc_5shot",
    "mmlu_stem_test_rc_5shot",
    # Math - Minerva BPB (7)
    "minerva_math_algebra_gold_bpb_0shot",
    "minerva_math_counting_and_probability_gold_bpb_0shot",
    "minerva_math_geometry_gold_bpb_0shot",
    "minerva_math_intermediate_algebra_gold_bpb_0shot",
    "minerva_math_number_theory_gold_bpb_0shot",
    "minerva_math_prealgebra_gold_bpb_0shot",
    "minerva_math_precalculus_gold_bpb_0shot",
    # GSM8K BPB (1)
    "gsm8k_gold_bpb_5shot",
    # Code BPB (2)
    "codex_humaneval_gold_bpb_0shot",
    "codex_mbpp_gold_bpb_0shot",
    # Basic skills BPB (6) - NEW
    "basic_skills_arithmetic_bpb_5shot",
    "basic_skills_coding_bpb_5shot",
    "basic_skills_common_knowledge_bpb_5shot",
    "basic_skills_logical_reasoning_bpb_5shot",
    "basic_skills_pattern_bpb_5shot",
    "basic_skills_string_operations_bpb_5shot",
    # Gen tasks BPB (5) - NEW
    "coqa_bpb_5shot",
    "drop_bpb_5shot",
    "jeopardy_bpb_5shot",
    "naturalqs_bpb_5shot",
    "squad_bpb_5shot",
    # Science/medical BPB (7) - NEW
    "lab_bench_dbqa_bpb",
    "lab_bench_protocolqa_bpb",
    "lambada_bpb",
    "medmcqa_bpb",
    "medqa_en_bpb",
    "qasper_yesno_bpb",
    "sciriff_yesno_bpb",
]
```

### CLI Options Added

```bash
# Use default 39 tasks with default interval (1000 steps)
olmix launch run --config config.yaml

# Custom eval interval
olmix launch run --config config.yaml --eval-interval 500

# Custom task list
olmix launch run --config config.yaml -e arc_challenge_test_rc_5shot -e hellaswag_rc_5shot

# Disable evaluations entirely
olmix launch run --config config.yaml --no-eval
```

## Verification

### Step 1: Local Import Test
```bash
cd ~/ai2/olmix
python -c "from olmix.model.transformer import TransformerConfigBuilder; print('Import OK')"
```

### Step 2: Launch Verify Tokens Jobs with 39 WandbMetrics BPB Tasks
```bash
# These will use the updated trainer with 39 BPB evaluator tasks
olmix launch run --config configs/test/verify_tokens_140m.yaml
olmix launch run --config configs/test/verify_tokens_700m.yaml
olmix launch run --config configs/test/verify_tokens_1400m.yaml
```

### Step 3: Monitor Beaker Jobs
```bash
beaker experiment get <EXP_ID> --format json | jq '.jobs[0].status'
```

### Step 4: Verify 39 WandbMetrics BPB Metrics in WandB
```python
import wandb
api = wandb.Api()

run = api.run('ai2-llm/olmix/<RUN_ID>')
history = run.history()

# Count BPB metrics (should be 39)
bpb_cols = [c for c in history.columns if 'BPB v2' in c]
print(f'BPB metrics found: {len(bpb_cols)} (expected: 39)')

# Verify the 18 new tasks are present
new_keywords = ['basic_skills', 'coqa', 'drop', 'jeopardy', 'naturalqs', 'squad',
               'lab_bench', 'lambada', 'medmcqa', 'medqa', 'qasper', 'sciriff']
new_bpb = [c for c in bpb_cols if any(kw in c for kw in new_keywords)]
print(f'New BPB metrics: {len(new_bpb)} (expected: 18)')
```

## Status

- [x] Phase 1: Add DownstreamEvaluatorCallbackConfig import
- [x] Phase 2: Add eval_tasks/eval_interval parameters to TransformerConfigBuilder
- [x] Phase 3: Update build_callbacks() to include evaluator
- [x] Phase 4: Add DEFAULT_EVAL_TASKS to evaluators.py (39 tasks)
- [x] Phase 5: Add CLI options to train.py
- [x] Local import tests pass
- [ ] Launch verification jobs
- [ ] Verify WandB metrics
