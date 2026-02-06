"""Constants for olmix fit module.

This module defines:
1. WandbMetrics - In-loop BPB metrics logged to WandB during training (60 metrics)
2. OlmoEvalMetrics - Offline BPB metrics for olmo-cookbook-eval (109 tasks)
"""

from enum import Enum

BASE_METRICS_PATH = "ai2-llm/regmixer"


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


# List of all WandB metric values for convenience
ALL_WANDB_METRICS = [m.value for m in WandbMetrics]


class OlmoEvalMetrics(Enum):
    """Offline BPB metrics for olmo-cookbook-eval.

    Source: olmo3:dev:1b:bpb task group in olmo-cookbook (109 tasks).
    These task names are compatible with olmo-cookbook-eval CLI.
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

    # Science/medical (8)
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


# List of all OlmoEval metric values for convenience
ALL_OLMO_EVAL_METRICS = [m.value for m in OlmoEvalMetrics]
