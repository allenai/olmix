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

**Results** (60 BPB v2 tasks, lower is better):

#### Core QA RC (7 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| hellaswag | 1.611 | 1.448 | **1.401** | -13.0% |
| arc_challenge_test | 2.046 | 1.890 | **1.782** | -12.9% |
| arc_easy_test | 2.019 | 1.871 | **1.694** | -16.1% |
| piqa_val | 1.981 | 1.839 | **1.780** | -10.1% |
| winogrande_val | 1.737 | 1.655 | **1.622** | -6.6% |
| socialiqa_val | 2.077 | 1.867 | **1.853** | -10.8% |
| csqa_val | 2.382 | **2.099** | 2.183 | -8.3% |

#### MMLU Test RC (4 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| mmlu_humanities_test | 1.972 | 1.688 | **1.631** | -17.3% |
| mmlu_other_test | 2.420 | 2.248 | **2.174** | -10.2% |
| mmlu_social_sciences_test | 1.787 | 1.672 | **1.618** | -9.4% |
| mmlu_stem_test | 2.890 | 2.736 | **2.576** | -10.9% |

#### MMLU Val RC (4 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| mmlu_humanities_val | 2.035 | 1.741 | **1.687** | -17.1% |
| mmlu_other_val | 2.517 | 2.303 | **2.257** | -10.3% |
| mmlu_social_sciences_val | 1.762 | 1.658 | **1.593** | -9.6% |
| mmlu_stem_val | 2.871 | 2.745 | **2.615** | -8.9% |

#### Math - GSM8K & Minerva (8 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| gsm8k | 2.101 | 1.695 | **1.612** | -23.3% |
| minerva_math_algebra | 2.950 | 2.172 | **2.079** | -29.5% |
| minerva_math_counting_and_probability | 2.414 | 1.806 | **1.733** | -28.2% |
| minerva_math_geometry | 3.147 | 2.348 | **2.221** | -29.4% |
| minerva_math_intermediate_algebra | 3.363 | 2.367 | **2.265** | -32.7% |
| minerva_math_number_theory | 2.735 | 2.049 | **1.973** | -27.8% |
| minerva_math_prealgebra | 2.590 | 1.960 | **1.867** | -27.9% |
| minerva_math_precalculus | 3.493 | 2.264 | **2.142** | -38.7% |

#### Code (3 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| codex_humaneval | 3.213 | **2.660** | 2.682 | -16.5% |
| codex_mbpp | 3.749 | 3.091 | **3.066** | -18.2% |
| basic_skills_coding | 4.553 | 3.900 | **3.762** | -17.4% |

#### Generative QA (6 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| coqa | 2.604 | 2.041 | **1.985** | -23.8% |
| drop | **5.208** | 5.634 | 5.466 | 4.9% |
| jeopardy | 2.731 | **2.447** | 2.452 | -10.2% |
| lambada | 2.721 | 2.248 | **2.153** | -20.9% |
| naturalqs | 2.707 | **2.602** | 2.625 | -3.0% |
| squad | 2.539 | 1.893 | **1.812** | -28.6% |

#### Basic Skills (5 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| basic_skills_arithmetic | 2.763 | 2.737 | **2.661** | -3.7% |
| basic_skills_common_knowledge | 2.185 | 2.017 | **1.953** | -10.6% |
| basic_skills_logical_reasoning | 1.676 | 1.358 | **1.226** | -26.8% |
| basic_skills_pattern | 3.454 | 2.254 | **2.075** | -39.9% |
| basic_skills_string_operations | 4.270 | **3.970** | 3.987 | -6.6% |

#### Science/Medical (6 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| lab_bench_dbqa | 5.764 | 5.871 | **5.698** | -1.1% |
| lab_bench_protocolqa | 2.675 | 2.400 | **2.336** | -12.7% |
| medmcqa | 3.084 | 2.883 | **2.810** | -8.9% |
| medqa_en | 2.861 | 2.652 | **2.591** | -9.4% |
| qasper_yesno | 2.071 | **0.743** | 0.911 | -56.0% |
| sciriff_yesno | 2.199 | **1.286** | 1.620 | -26.3% |

#### MT MBPP - Multilingual Code (17 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| mt_mbpp_bash | 4.267 | 3.614 | **3.452** | -19.1% |
| mt_mbpp_c | 3.346 | 2.545 | **2.473** | -26.1% |
| mt_mbpp_cpp | 3.441 | 2.668 | **2.596** | -24.6% |
| mt_mbpp_csharp | 2.995 | 2.378 | **2.294** | -23.4% |
| mt_mbpp_go | 3.593 | 2.958 | **2.819** | -21.6% |
| mt_mbpp_haskell | 3.858 | 3.336 | **3.248** | -15.8% |
| mt_mbpp_java | 2.917 | 2.224 | **2.141** | -26.6% |
| mt_mbpp_javascript | 3.563 | 2.846 | **2.785** | -21.8% |
| mt_mbpp_matlab | 3.457 | 2.950 | **2.892** | -16.3% |
| mt_mbpp_php | 3.620 | 2.789 | **2.601** | -28.2% |
| mt_mbpp_python | 4.299 | **3.702** | 3.704 | -13.8% |
| mt_mbpp_r | 4.013 | 3.153 | **2.964** | -26.2% |
| mt_mbpp_ruby | 4.351 | 3.561 | **3.441** | -20.9% |
| mt_mbpp_rust | 3.989 | 3.314 | **3.253** | -18.5% |
| mt_mbpp_scala | 4.227 | 3.487 | **3.380** | -20.0% |
| mt_mbpp_swift | 3.560 | 2.981 | **2.833** | -20.4% |
| mt_mbpp_typescript | 3.472 | 2.779 | **2.705** | -22.1% |

**Key Findings:**

1. **Math tasks show largest improvements** (28-39% reduction in BPB)
   - precalculus: -38.7%
   - intermediate_algebra: -32.7%
   - algebra: -29.5%

2. **Pattern recognition improves dramatically**: -39.9%

3. **qasper_yesno shows unusual behavior**: -56% at 2.5x but regresses at 5.0x

4. **Most gains happen 0.5x → 2.5x**, with diminishing returns from 2.5x → 5.0x

5. **Average improvement across all 60 tasks**: -18.5% BPB reduction from 140M to 1.4B tokens

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

**Results** (60 BPB v2 tasks, lower is better):

#### Core QA RC (7 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| hellaswag | **1.588** | 1.626 | 1.621 | 1.605 | Base |
| arc_challenge_test | 2.018 | 2.014 | **1.887** | 1.997 | Sci |
| arc_easy_test | 1.975 | 2.028 | **1.869** | 2.030 | Sci |
| piqa_val | 1.936 | 1.997 | 1.952 | **1.933** | Wiki |
| winogrande_val | **1.762** | 1.857 | 1.822 | 1.831 | Base |
| socialiqa_val | **1.981** | 2.222 | 2.158 | 2.047 | Base |
| csqa_val | 2.474 | 2.598 | 2.498 | **2.435** | Wiki |

#### MMLU Test RC (4 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| mmlu_humanities_test | **1.963** | 2.033 | 1.998 | 1.969 | Base |
| mmlu_other_test | 2.363 | 2.418 | **2.319** | 2.372 | Sci |
| mmlu_social_sciences_test | **1.771** | 1.878 | 1.790 | 1.829 | Base |
| mmlu_stem_test | 2.770 | 2.719 | **2.672** | 2.676 | Sci |

#### MMLU Val RC (4 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| mmlu_humanities_val | 2.014 | 2.080 | 2.046 | **2.012** | Wiki |
| mmlu_other_val | 2.449 | 2.500 | **2.409** | 2.437 | Sci |
| mmlu_social_sciences_val | **1.744** | 1.877 | 1.770 | 1.817 | Base |
| mmlu_stem_val | 2.783 | 2.723 | **2.667** | 2.685 | Sci |

#### Math - GSM8K & Minerva (8 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| gsm8k | **2.028** | 2.081 | 2.138 | 2.031 | Base |
| minerva_math_algebra | 2.806 | **2.376** | 2.518 | 2.377 | Code |
| minerva_math_counting_and_probability | 2.316 | **2.015** | 2.130 | 2.027 | Code |
| minerva_math_geometry | 3.028 | **2.549** | 2.715 | 2.584 | Code |
| minerva_math_intermediate_algebra | 3.202 | **2.612** | 2.821 | 2.659 | Code |
| minerva_math_number_theory | 2.623 | **2.217** | 2.384 | 2.233 | Code |
| minerva_math_prealgebra | 2.472 | 2.177 | 2.297 | **2.176** | Wiki |
| minerva_math_precalculus | 3.339 | **2.584** | 2.823 | 2.605 | Code |

#### Code (3 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| codex_humaneval | 3.225 | **2.050** | 2.373 | 2.094 | Code |
| codex_mbpp | 3.734 | **2.612** | 2.877 | 2.620 | Code |
| basic_skills_coding | 4.409 | **2.935** | 3.439 | 2.948 | Code |

#### Generative QA (6 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| coqa | 2.465 | 2.540 | **2.440** | 2.475 | Sci |
| drop | 6.238 | 5.695 | **5.242** | 5.699 | Sci |
| jeopardy | 2.696 | 2.848 | **2.694** | 2.726 | Sci |
| lambada | **2.676** | 2.774 | 2.755 | 2.754 | Base |
| naturalqs | 2.767 | 2.872 | 2.692 | **2.685** | Wiki |
| squad | 2.450 | **2.436** | 2.463 | 2.478 | Code |

#### Basic Skills (5 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| basic_skills_arithmetic | 2.860 | **2.725** | 2.833 | 2.796 | Code |
| basic_skills_common_knowledge | 2.197 | 2.203 | **2.100** | 2.132 | Sci |
| basic_skills_logical_reasoning | **1.602** | 1.657 | 1.680 | 1.624 | Base |
| basic_skills_pattern | 3.246 | **3.069** | 3.237 | 3.211 | Code |
| basic_skills_string_operations | 4.122 | **4.009** | 4.065 | 4.114 | Code |

#### Science/Medical (6 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| lab_bench_dbqa | 5.608 | 5.518 | 5.531 | **5.427** | Wiki |
| lab_bench_protocolqa | 2.650 | 2.509 | 2.463 | **2.441** | Wiki |
| medmcqa | 3.045 | 3.013 | **2.851** | 2.895 | Sci |
| medqa_en | 2.889 | 2.746 | **2.543** | 2.712 | Sci |
| qasper_yesno | 1.951 | 1.942 | 1.956 | **1.693** | Wiki |
| sciriff_yesno | 2.153 | 2.089 | 2.288 | **1.931** | Wiki |

#### MT MBPP - Multilingual Code (17 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| mt_mbpp_bash | 4.290 | **2.920** | 3.339 | 3.058 | Code |
| mt_mbpp_c | 3.310 | **2.013** | 2.373 | 2.158 | Code |
| mt_mbpp_cpp | 3.384 | **2.030** | 2.441 | 2.186 | Code |
| mt_mbpp_csharp | 2.985 | **1.794** | 2.116 | 1.927 | Code |
| mt_mbpp_go | 3.545 | **2.507** | 2.764 | 2.597 | Code |
| mt_mbpp_haskell | 3.812 | **2.868** | 3.279 | 2.973 | Code |
| mt_mbpp_java | 2.870 | **1.658** | 1.999 | 1.781 | Code |
| mt_mbpp_javascript | 3.512 | **2.275** | 2.612 | 2.360 | Code |
| mt_mbpp_matlab | 3.416 | **2.579** | 2.802 | 2.646 | Code |
| mt_mbpp_php | 3.479 | **2.167** | 2.528 | 2.306 | Code |
| mt_mbpp_python | 4.265 | **3.258** | 3.452 | 3.292 | Code |
| mt_mbpp_r | 3.861 | **2.734** | 3.010 | 2.827 | Code |
| mt_mbpp_ruby | 4.188 | **3.080** | 3.423 | 3.212 | Code |
| mt_mbpp_rust | 3.920 | **2.819** | 3.202 | 2.985 | Code |
| mt_mbpp_scala | 4.099 | **2.920** | 3.246 | 2.980 | Code |
| mt_mbpp_swift | 3.507 | **2.464** | 2.840 | 2.535 | Code |
| mt_mbpp_typescript | 3.434 | **2.281** | 2.598 | 2.395 | Code |

**Key Findings:**

1. **Heavy Code wins most tasks** (30/60): Dominates all code-related tasks
   - All 17 MT MBPP languages: ~32% average improvement over baseline
   - Core code tasks: 30-36% improvement (humaneval, mbpp, basic_skills_coding)
   - Most Minerva math tasks: 15-23% improvement

2. **Heavy Science wins 12/60**: Best for reasoning and medical/science knowledge
   - Medical tasks: medmcqa, medqa_en
   - Drop, coqa, jeopardy (generative QA)
   - MMLU other/stem (test and val)

3. **Baseline wins 9/60**: Best for social/commonsense reasoning
   - socialiqa, winogrande, hellaswag
   - MMLU social_sciences and humanities
   - lambada, gsm8k, logical_reasoning

4. **Heavy Wiki wins 9/60**: Best for diverse text patterns and science papers
   - qasper_yesno, sciriff_yesno (scientific paper understanding)
   - lab_bench_dbqa, lab_bench_protocolqa
   - naturalqs, csqa, piqa

**Summary by Category:**
- Code tasks (20 total): Code mix dominates (20/20 wins)
- Math (8 total): Code mix best (6/8 wins)
- Reasoning (7 total): Mixed (Base 3, Sci 2, Wiki 2)
- MMLU (8 total): Mixed (Base 3, Sci 4, Wiki 1)
- Generative QA (6 total): Mixed (Sci 3, Base 1, Wiki 1, Code 1)
- Science/Medical (6 total): Wiki best (4/6 wins)
- Basic Skills (5 total): Code best (4/5 wins)

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
- Eval: 60 BPB tasks every 1000 steps (aligned with olmo-core's FULL_TASKS_SMALL_COMPUTE)

## Experiment Tracking

Launch metadata is saved to `output/mixes/{name}_{group_id}.json` with:
- Beaker experiment IDs and URLs
- WandB group URL
- Git commit and branch
- Mix configurations
