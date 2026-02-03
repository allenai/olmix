# olmix Experiments

Experiment configs for validating olmix training and evaluating data mixing strategies.

## Experiment Suites

### 1. Training Duration (`training_duration/`)

Tests how BPB metrics improve with longer training, holding data proportions constant.

| Config | Chinchilla | Tokens | Runtime | Steps |
|--------|------------|--------|---------|-------|
| duration_0.5x.yaml | 0.5x | 140M | ~8 min | 1,061 |
| duration_2.5x.yaml | 2.5x | 700M | ~35 min | 5,301 |
| duration_5.0x.yaml | 5.0x | 1.4B | ~70 min | 10,602 |

**Results** (BPB v2, lower is better):

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| **Reasoning & Knowledge** |||||
| hellaswag_rc_5shot | 1.609 | 1.440 | **1.413** | -12.2% |
| arc_challenge_test_rc_5shot | 2.042 | 1.864 | **1.866** | -8.6% |
| arc_easy_test_rc_5shot | 2.076 | **1.799** | 1.858 | -13.3% |
| piqa_val_rc_5shot | 1.995 | 1.836 | **1.819** | -8.8% |
| winogrande_val_rc_5shot | 1.784 | 1.646 | **1.632** | -8.5% |
| socialiqa_val_rc_5shot | 2.120 | **1.762** | 1.826 | -16.9% |
| csqa_val_rc_5shot | 2.701 | 2.269 | **2.070** | -23.4% |
| **MMLU** |||||
| mmlu_humanities_test_rc_5shot | 1.990 | 1.623 | **1.601** | -19.5% |
| mmlu_other_test_rc_5shot | 2.459 | 2.266 | **2.233** | -9.2% |
| mmlu_social_sciences_test_rc_5shot | 1.798 | 1.682 | **1.640** | -8.8% |
| mmlu_stem_test_rc_5shot | 2.897 | 2.695 | **2.583** | -10.8% |
| **Math** |||||
| gsm8k_gold_bpb_5shot | 2.095 | 1.662 | **1.622** | -22.6% |
| minerva_math_algebra_gold_bpb_0shot | 2.884 | 2.140 | **2.101** | -27.2% |
| minerva_math_counting_and_probability | 2.358 | 1.794 | **1.738** | -26.3% |
| minerva_math_geometry_gold_bpb_0shot | 3.106 | 2.326 | **2.194** | -29.4% |
| minerva_math_intermediate_algebra | 3.289 | 2.301 | **2.289** | -30.4% |
| minerva_math_number_theory | 2.698 | 2.020 | **1.975** | -26.8% |
| minerva_math_prealgebra | 2.537 | 1.930 | **1.871** | -26.3% |
| minerva_math_precalculus | 3.438 | 2.202 | **2.123** | -38.2% |
| **Code** |||||
| codex_humaneval_gold_bpb_0shot | 3.220 | 2.717 | **2.679** | -16.8% |
| codex_mbpp_gold_bpb_0shot | 4.308 | 3.749 | **3.681** | -14.6% |
| basic_skills_coding_rc_5shot | 4.490 | **3.804** | 3.889 | -15.3% |
| **Basic Skills** |||||
| basic_skills_arithmetic_rc_5shot | 2.804 | **2.728** | 2.733 | -2.7% |
| basic_skills_common_knowledge_rc_5shot | 2.181 | 2.104 | **2.054** | -5.8% |
| basic_skills_logical_reasoning_rc_5shot | 1.706 | 1.312 | **1.296** | -24.0% |
| basic_skills_pattern_rc_5shot | 3.531 | 2.238 | **2.109** | -40.3% |
| basic_skills_string_operations_rc_5shot | 4.244 | 3.941 | **3.849** | -9.3% |

**Key Findings:**

1. **Math tasks show largest improvements** (26-38% reduction in BPB)
   - precalculus: -38.2%
   - intermediate_algebra: -30.4%
   - geometry: -29.4%

2. **Pattern recognition improves dramatically**: -40.3%

3. **Most gains happen 0.5x → 2.5x**, with diminishing returns from 2.5x → 5.0x

4. **Average improvement across all 27 tasks**: ~18% BPB reduction from 140M to 1.4B tokens

### 2. Data Proportions (`data_proportions/`)

Tests how different data mixes affect BPB metrics, holding training duration constant (0.5x Chinchilla, ~8 min).

| Config | Emphasis | Runtime |
|--------|----------|---------|
| mix_baseline.yaml | Balanced | ~8 min |
| mix_heavy_code.yaml | 50% code | ~8 min |
| mix_heavy_science.yaml | 50% science | ~8 min |
| mix_heavy_wiki.yaml | 50% wikipedia | ~8 min |

**Actual Mixes Generated:**

| Config | Actual Proportions |
|--------|-------------------|
| mix_baseline | 94.2% education, 5.8% science |
| mix_heavy_code | 62.4% code, 25% science, 12.4% education, 0.2% arxiv |
| mix_heavy_science | 71.3% science, 14.2% code, 14.2% education, 0.2% arxiv |
| mix_heavy_wiki | 37.1% science, 37.1% code, 24.8% education, 0.6% wiki, 0.4% arxiv |

**Results** (BPB v2, lower is better):

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| **Reasoning & Knowledge** ||||||
| hellaswag_rc_5shot | 1.595 | 1.632 | **1.587** | 1.598 | Sci |
| arc_challenge_test_rc_5shot | 2.037 | 2.052 | **1.955** | 2.038 | Sci |
| arc_easy_test_rc_5shot | 2.026 | 1.920 | **1.873** | 1.994 | Sci |
| piqa_val_rc_5shot | 1.943 | 1.967 | **1.884** | 1.957 | Sci |
| winogrande_val_rc_5shot | **1.754** | 1.836 | 1.807 | 1.796 | Base |
| socialiqa_val_rc_5shot | **1.976** | 2.094 | 2.044 | 2.175 | Base |
| csqa_val_rc_5shot | **2.393** | 2.549 | 2.537 | 2.606 | Base |
| **MMLU** ||||||
| mmlu_humanities_test_rc_5shot | 1.982 | 2.013 | **1.937** | 1.977 | Sci |
| mmlu_other_test_rc_5shot | 2.401 | 2.418 | **2.275** | 2.409 | Sci |
| mmlu_social_sciences_test_rc_5shot | 1.777 | 1.866 | **1.750** | 1.844 | Sci |
| mmlu_stem_test_rc_5shot | 2.786 | 2.835 | **2.646** | 2.665 | Sci |
| **Math** ||||||
| gsm8k_gold_bpb_5shot | 2.031 | 2.080 | **2.017** | 2.045 | Sci |
| minerva_math_algebra | 2.838 | **2.337** | 2.386 | 2.422 | Code |
| minerva_math_counting_prob | 2.330 | 2.010 | **1.989** | 2.061 | Sci |
| minerva_math_geometry | 3.067 | **2.498** | 2.577 | 2.596 | Code |
| minerva_math_intermediate_algebra | 3.245 | **2.554** | 2.662 | 2.699 | Code |
| minerva_math_number_theory | 2.650 | **2.192** | 2.241 | 2.290 | Code |
| minerva_math_prealgebra | 2.505 | **2.151** | 2.160 | 2.223 | Code |
| minerva_math_precalculus | 3.359 | **2.478** | 2.648 | 2.654 | Code |
| **Code** ||||||
| codex_humaneval_gold | 3.112 | **2.165** | 2.359 | 2.175 | Code |
| codex_mbpp_gold | 4.257 | **3.184** | 3.515 | 3.279 | Code |
| basic_skills_coding_rc_5shot | 4.469 | **3.015** | 3.377 | 3.041 | Code |
| **Basic Skills** ||||||
| basic_skills_arithmetic_rc_5shot | 2.920 | 2.855 | **2.782** | 2.807 | Sci |
| basic_skills_common_knowledge_rc_5shot | 2.211 | 2.170 | **2.062** | 2.100 | Sci |
| basic_skills_logical_reasoning_rc_5shot | 1.646 | 1.677 | **1.609** | 1.619 | Sci |
| basic_skills_pattern_rc_5shot | 3.554 | 3.382 | 3.237 | **3.062** | Wiki |
| basic_skills_string_operations_rc_5shot | 4.052 | 4.031 | 4.081 | **3.907** | Wiki |

**Key Findings:**

1. **Heavy Science wins most tasks** (14/27): Best for reasoning, MMLU, and general knowledge
2. **Heavy Code dominates math** (6/27): Especially minerva_math tasks (17-26% improvement over baseline)
3. **Code tasks strongly prefer code mix**: 30-33% improvement on coding benchmarks
4. **Baseline wins social/commonsense** (3/27): winogrande, socialiqa, csqa
5. **Wiki mix wins pattern/string tasks** (2/27): Likely due to diverse text patterns

**Summary by Category:**
- Reasoning: Science mix best
- MMLU: Science mix best
- Math: Code mix best (except gsm8k, counting_prob)
- Code: Code mix best (30%+ improvement)
- Basic Skills: Mixed (Science for most, Wiki for pattern/string)

## Running Experiments

```bash
# Single experiment
olmix launch run --config configs/experiments/training_duration/duration_0.5x.yaml

# All duration experiments
for f in configs/experiments/training_duration/*.yaml; do
  olmix launch run --config $f
done

# All proportion experiments
for f in configs/experiments/data_proportions/*.yaml; do
  olmix launch run --config $f
done
```

## Model & Infrastructure

All experiments use:
- Model: olmo3_14m (14M parameters)
- Tokenizer: dolma2
- Cluster: ai2/jupiter (1 GPU)
- Eval: 27 BPB tasks every 1000 steps
