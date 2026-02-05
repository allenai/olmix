# Training Duration Experiments

Tests how BPB metrics improve with longer training, holding data proportions constant.

## Configs

| Config | Chinchilla | Tokens | Runtime | Steps |
|--------|------------|--------|---------|-------|
| duration_0.5x.yaml | 0.5x | 140M | ~8 min | 1,061 |
| duration_2.5x.yaml | 2.5x | 700M | ~35 min | 5,301 |
| duration_5.0x.yaml | 5.0x | 1.4B | ~70 min | 10,602 |

## Results

60 BPB v2 tasks, lower is better.

### Core QA RC (7 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| hellaswag | 1.611 | 1.448 | **1.401** | -13.0% |
| arc_challenge_test | 2.046 | 1.890 | **1.782** | -12.9% |
| arc_easy_test | 2.019 | 1.871 | **1.694** | -16.1% |
| piqa_val | 1.981 | 1.839 | **1.780** | -10.1% |
| winogrande_val | 1.737 | 1.655 | **1.622** | -6.6% |
| socialiqa_val | 2.077 | 1.867 | **1.853** | -10.8% |
| csqa_val | 2.382 | **2.099** | 2.183 | -8.3% |

### MMLU Test RC (4 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| mmlu_humanities_test | 1.972 | 1.688 | **1.631** | -17.3% |
| mmlu_other_test | 2.420 | 2.248 | **2.174** | -10.2% |
| mmlu_social_sciences_test | 1.787 | 1.672 | **1.618** | -9.4% |
| mmlu_stem_test | 2.890 | 2.736 | **2.576** | -10.9% |

### MMLU Val RC (4 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| mmlu_humanities_val | 2.035 | 1.741 | **1.687** | -17.1% |
| mmlu_other_val | 2.517 | 2.303 | **2.257** | -10.3% |
| mmlu_social_sciences_val | 1.762 | 1.658 | **1.593** | -9.6% |
| mmlu_stem_val | 2.871 | 2.745 | **2.615** | -8.9% |

### Math - GSM8K & Minerva (8 tasks)

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

### Code (3 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| codex_humaneval | 3.213 | **2.660** | 2.682 | -16.5% |
| codex_mbpp | 3.749 | 3.091 | **3.066** | -18.2% |
| basic_skills_coding | 4.553 | 3.900 | **3.762** | -17.4% |

### Generative QA (6 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| coqa | 2.604 | 2.041 | **1.985** | -23.8% |
| drop | **5.208** | 5.634 | 5.466 | 4.9% |
| jeopardy | 2.731 | **2.447** | 2.452 | -10.2% |
| lambada | 2.721 | 2.248 | **2.153** | -20.9% |
| naturalqs | 2.707 | **2.602** | 2.625 | -3.0% |
| squad | 2.539 | 1.893 | **1.812** | -28.6% |

### Basic Skills (5 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| basic_skills_arithmetic | 2.763 | 2.737 | **2.661** | -3.7% |
| basic_skills_common_knowledge | 2.185 | 2.017 | **1.953** | -10.6% |
| basic_skills_logical_reasoning | 1.676 | 1.358 | **1.226** | -26.8% |
| basic_skills_pattern | 3.454 | 2.254 | **2.075** | -39.9% |
| basic_skills_string_operations | 4.270 | **3.970** | 3.987 | -6.6% |

### Science/Medical (6 tasks)

| Task | 0.5x (140M) | 2.5x (700M) | 5.0x (1.4B) | Improvement |
|------|-------------|-------------|-------------|-------------|
| lab_bench_dbqa | 5.764 | 5.871 | **5.698** | -1.1% |
| lab_bench_protocolqa | 2.675 | 2.400 | **2.336** | -12.7% |
| medmcqa | 3.084 | 2.883 | **2.810** | -8.9% |
| medqa_en | 2.861 | 2.652 | **2.591** | -9.4% |
| qasper_yesno | 2.071 | **0.743** | 0.911 | -56.0% |
| sciriff_yesno | 2.199 | **1.286** | 1.620 | -26.3% |

### MT MBPP - Multilingual Code (17 tasks)

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

## Key Findings

1. **Math tasks show largest improvements** (28-39% reduction in BPB)
   - precalculus: -38.7%
   - intermediate_algebra: -32.7%
   - algebra: -29.5%

2. **Pattern recognition improves dramatically**: -39.9%

3. **qasper_yesno shows unusual behavior**: -56% at 2.5x but regresses at 5.0x

4. **Most gains happen 0.5x -> 2.5x**, with diminishing returns from 2.5x -> 5.0x

5. **Average improvement across all 60 tasks**: -18.5% BPB reduction from 140M to 1.4B tokens
