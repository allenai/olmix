# Data Proportions Experiments

Tests how different data mixes affect BPB metrics, holding training duration constant (0.5x Chinchilla, ~8 min).

## Configs

| Config | Emphasis | Runtime |
|--------|----------|---------|
| mix_baseline.yaml | Balanced | ~8 min |
| mix_heavy_code.yaml | 50% code | ~8 min |
| mix_heavy_science.yaml | 50% science | ~8 min |
| mix_heavy_wiki.yaml | 50% wikipedia | ~8 min |

### Actual Mixes Generated

| Config | Actual Proportions |
|--------|-------------------|
| mix_baseline | 94.2% education, 5.8% science |
| mix_heavy_code | 62.4% code, 25% science, 12.4% education, 0.2% arxiv |
| mix_heavy_science | 71.3% science, 14.2% code, 14.2% education, 0.2% arxiv |
| mix_heavy_wiki | 37.1% science, 37.1% code, 24.8% education, 0.6% wiki, 0.4% arxiv |

## Results

60 BPB v2 tasks, lower is better.

### Core QA RC (7 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| hellaswag | **1.588** | 1.626 | 1.621 | 1.605 | Base |
| arc_challenge_test | 2.018 | 2.014 | **1.887** | 1.997 | Sci |
| arc_easy_test | 1.975 | 2.028 | **1.869** | 2.030 | Sci |
| piqa_val | 1.936 | 1.997 | 1.952 | **1.933** | Wiki |
| winogrande_val | **1.762** | 1.857 | 1.822 | 1.831 | Base |
| socialiqa_val | **1.981** | 2.222 | 2.158 | 2.047 | Base |
| csqa_val | 2.474 | 2.598 | 2.498 | **2.435** | Wiki |

### MMLU Test RC (4 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| mmlu_humanities_test | **1.963** | 2.033 | 1.998 | 1.969 | Base |
| mmlu_other_test | 2.363 | 2.418 | **2.319** | 2.372 | Sci |
| mmlu_social_sciences_test | **1.771** | 1.878 | 1.790 | 1.829 | Base |
| mmlu_stem_test | 2.770 | 2.719 | **2.672** | 2.676 | Sci |

### MMLU Val RC (4 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| mmlu_humanities_val | 2.014 | 2.080 | 2.046 | **2.012** | Wiki |
| mmlu_other_val | 2.449 | 2.500 | **2.409** | 2.437 | Sci |
| mmlu_social_sciences_val | **1.744** | 1.877 | 1.770 | 1.817 | Base |
| mmlu_stem_val | 2.783 | 2.723 | **2.667** | 2.685 | Sci |

### Math - GSM8K & Minerva (8 tasks)

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

### Code (3 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| codex_humaneval | 3.225 | **2.050** | 2.373 | 2.094 | Code |
| codex_mbpp | 3.734 | **2.612** | 2.877 | 2.620 | Code |
| basic_skills_coding | 4.409 | **2.935** | 3.439 | 2.948 | Code |

### Generative QA (6 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| coqa | 2.465 | 2.540 | **2.440** | 2.475 | Sci |
| drop | 6.238 | 5.695 | **5.242** | 5.699 | Sci |
| jeopardy | 2.696 | 2.848 | **2.694** | 2.726 | Sci |
| lambada | **2.676** | 2.774 | 2.755 | 2.754 | Base |
| naturalqs | 2.767 | 2.872 | 2.692 | **2.685** | Wiki |
| squad | 2.450 | **2.436** | 2.463 | 2.478 | Code |

### Basic Skills (5 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| basic_skills_arithmetic | 2.860 | **2.725** | 2.833 | 2.796 | Code |
| basic_skills_common_knowledge | 2.197 | 2.203 | **2.100** | 2.132 | Sci |
| basic_skills_logical_reasoning | **1.602** | 1.657 | 1.680 | 1.624 | Base |
| basic_skills_pattern | 3.246 | **3.069** | 3.237 | 3.211 | Code |
| basic_skills_string_operations | 4.122 | **4.009** | 4.065 | 4.114 | Code |

### Science/Medical (6 tasks)

| Task | Baseline | Heavy Code | Heavy Science | Heavy Wiki | Best |
|------|----------|------------|---------------|------------|------|
| lab_bench_dbqa | 5.608 | 5.518 | 5.531 | **5.427** | Wiki |
| lab_bench_protocolqa | 2.650 | 2.509 | 2.463 | **2.441** | Wiki |
| medmcqa | 3.045 | 3.013 | **2.851** | 2.895 | Sci |
| medqa_en | 2.889 | 2.746 | **2.543** | 2.712 | Sci |
| qasper_yesno | 1.951 | 1.942 | 1.956 | **1.693** | Wiki |
| sciriff_yesno | 2.153 | 2.089 | 2.288 | **1.931** | Wiki |

### MT MBPP - Multilingual Code (17 tasks)

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

## Key Findings

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

## Summary by Category

| Category | Total | Code | Science | Baseline | Wiki |
|----------|-------|------|---------|----------|------|
| Code tasks | 20 | **20** | 0 | 0 | 0 |
| Math | 8 | **6** | 0 | 1 | 1 |
| Reasoning | 7 | 0 | 2 | **3** | 2 |
| MMLU | 8 | 0 | **4** | 3 | 1 |
| Generative QA | 6 | 1 | **3** | 1 | 1 |
| Science/Medical | 6 | 0 | 2 | 0 | **4** |
| Basic Skills | 5 | **4** | 1 | 0 | 0 |
