"""Evaluator configurations for olmix experiments."""

from enum import Enum

# Default task list for in-loop evaluation (27 tasks that are verified to work)
# NOTE: Gen tasks (coqa, drop, etc.) and science/medical tasks are disabled due to
# package data loading issues when installing olmo_eval from git. Re-enable once
# olmo_eval is published to PyPI or the data loading issue is resolved.
DEFAULT_EVAL_TASKS: list[str] = [
    # Core QA RC (7)
    "arc_challenge_test_rc_5shot",
    "arc_easy_test_rc_5shot",
    "csqa_val_rc_5shot",
    "hellaswag_rc_5shot",
    "piqa_val_rc_5shot",
    "socialiqa_val_rc_5shot",
    "winogrande_val_rc_5shot",
    # MMLU RC (4)
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
    # Basic skills BPB (6)
    "basic_skills_arithmetic_bpb_5shot",
    "basic_skills_coding_bpb_5shot",
    "basic_skills_common_knowledge_bpb_5shot",
    "basic_skills_logical_reasoning_bpb_5shot",
    "basic_skills_pattern_bpb_5shot",
    "basic_skills_string_operations_bpb_5shot",
]


class CodeTasks(Enum):
    """Enum of code-related evaluation tasks."""

    BASIC_SKILLS_CODING_RC_5SHOT = "basic_skills_coding_rc_5shot"
    CODEX_HUMANEVAL = "codex_humaneval_gold_bpb_0shot"
    CODEX_MBPP = "codex_mbpp_gold_bpb_0shot"


class DownstreamEvaluatorsSmall(Enum):
    """
    Enum class enumerating in-loop evaluators for small models.

    This is the current set of evaluators used in OLMo-core + basic_skills.
    """

    ARC_CHALLENGE = "arc_challenge_test_rc_5shot"
    ARC_EASY = "arc_easy_test_rc_5shot"
    HELLASWAG = "hellaswag_rc_5shot"  # 1K subset of HellaSwag
    WINOGRANDE = "winogrande_val_rc_5shot"  # Helpful after 750M-5xC scale
    CSQA = "csqa_val_rc_5shot"
    PIQA = "piqa_val_rc_5shot"
    SOCIALIQA = "socialiqa_val_rc_5shot"

    # BASIC_SKILLS RC
    BASIC_SKILLS_ARITHMETIC_RC_5SHOT = "basic_skills_arithmetic_rc_5shot"
    BASIC_SKILLS_CODING_RC_5SHOT = "basic_skills_coding_rc_5shot"
    BASIC_SKILLS_COMMON_KNOWLEDGE_RC_5SHOT = "basic_skills_common_knowledge_rc_5shot"
    BASIC_SKILLS_LOGICAL_REASONING_RC_5SHOT = "basic_skills_logical_reasoning_rc_5shot"
    BASIC_SKILLS_PATTERN_RC_5SHOT = "basic_skills_pattern_rc_5shot"
    BASIC_SKILLS_STRING_OPERATIONS_RC_5SHOT = "basic_skills_string_operations_rc_5shot"

    # MMLU RC
    MMLU_STEM_VAL = "mmlu_stem_val_rc_5shot"
    MMLU_HUMANITIES_VAL = "mmlu_humanities_val_rc_5shot"
    MMLU_SOCIAL_SCIENCES_VAL = "mmlu_social_sciences_val_rc_5shot"
    MMLU_OTHER_VAL = "mmlu_other_val_rc_5shot"
    MMLU_STEM_TEST = "mmlu_stem_test_rc_5shot"
    MMLU_HUMANITIES_TEST = "mmlu_humanities_test_rc_5shot"
    MMLU_SOCIAL_SCIENCES_TEST = "mmlu_social_sciences_test_rc_5shot"
    MMLU_OTHER_TEST = "mmlu_other_test_rc_5shot"

    # Gen tasks BPB
    GSM8K = "gsm8k_gold_bpb_5shot"
    MINERVA_ALGEBRA = "minerva_math_algebra_gold_bpb_0shot"
    MINERVA_COUNTING = "minerva_math_counting_and_probability_gold_bpb_0shot"
    MINERVA_GEOMETRY = "minerva_math_geometry_gold_bpb_0shot"
    MINERVA_INTERMEDIATE = "minerva_math_intermediate_algebra_gold_bpb_0shot"
    MINERVA_NUMBER = "minerva_math_number_theory_gold_bpb_0shot"
    MINERVA_PREALGEBRA = "minerva_math_prealgebra_gold_bpb_0shot"
    MINERVA_PRECALCULUS = "minerva_math_precalculus_gold_bpb_0shot"
    CODEX_HUMANEVAL = "codex_humaneval_gold_bpb_0shot"
    CODEX_MBPP = "codex_mbpp_gold_bpb_0shot"

    # Sanity check for MCQA ability
    COPYCOLORS = "copycolors_10way"


class DownstreamEvaluators(Enum):
    """Enum class enumerating available in-loop evaluators."""

    PIQA = "piqa"
    HELLASWAG = "hellaswag"
    WINOGRANDE = "winogrande"
    OPENBOOK_QA = "openbook_qa"
    BOOLQ = "boolq"
    SCIQ = "sciq"
    ARC_EASY = "arc_easy"
    ARC_CHALLENGE = "arc_challenge"
    COPA = "copa"
    COMMONSENSE_QA = "commonsense_qa"
    SOCIAL_IQA = "social_iqa"
    MMLU_STEM_VAR = "mmlu_stem_var"
    MMLU_HUMANITIES_VAR = "mmlu_humanities_var"
    MMLU_SOCIAL_SCIENCES_VAR = "mmlu_social_sciences_var"
    MMLU_OTHER_VAR = "mmlu_other_var"
    MMLU_STEM_MC_5SHOT = "mmlu_stem_mc_5shot"
    MMLU_HUMANITIES_MC_5SHOT = "mmlu_humanities_mc_5shot"
    MMLU_SOCIAL_SCIENCES_MC_5SHOT = "mmlu_social_sciences_mc_5shot"
    MMLU_OTHER_MC_5SHOT = "mmlu_other_mc_5shot"
    MMLU_STEM_MC_5SHOT_TEST = "mmlu_stem_mc_5shot_test"
    MMLU_HUMANITIES_MC_5SHOT_TEST = "mmlu_humanities_mc_5shot_test"
    MMLU_SOCIAL_SCIENCES_MC_5SHOT_TEST = "mmlu_social_sciences_mc_5shot_test"
    MMLU_OTHER_MC_5SHOT_TEST = "mmlu_other_mc_5shot_test"
    BASIC_ARITHMETIC = "basic_arithmetic"
    TRIVIA_QA_WIKI_PPL = "trivia_qa_wiki_ppl"
    NATURAL_QS_OPEN_PPL = "natural_qs_open_ppl"
    ARC_EASY_PPL = "arc_easy_ppl"
    PIQA_RC_5SHOT = "piqa_rc_5shot"
    PIQA_RC_5SHOT_BPB = "piqa_rc_5shot_bpb"
    PIQA_MC_5SHOT = "piqa_mc_5shot"
    PIQA_MC_5SHOT_BPB = "piqa_mc_5shot_bpb"
    HELLASWAG_RC_5SHOT = "hellaswag_rc_5shot"
    HELLASWAG_RC_5SHOT_BPB = "hellaswag_rc_5shot_bpb"
    HELLASWAG_MC_5SHOT = "hellaswag_mc_5shot"
    HELLASWAG_MC_5SHOT_BPB = "hellaswag_mc_5shot_bpb"
    WINOGRANDE_RC_5SHOT = "winogrande_rc_5shot"
    WINOGRANDE_RC_5SHOT_BPB = "winogrande_rc_5shot_bpb"
    WINOGRANDE_MC_5SHOT = "winogrande_mc_5shot"
    WINOGRANDE_MC_5SHOT_BPB = "winogrande_mc_5shot_bpb"
    OPENBOOKQA_RC_5SHOT = "openbookqa_rc_5shot"
    OPENBOOKQA_RC_5SHOT_BPB = "openbookqa_rc_5shot_bpb"
    OPENBOOKQA_MC_5SHOT = "openbookqa_mc_5shot"
    OPENBOOKQA_MC_5SHOT_BPB = "openbookqa_mc_5shot_bpb"
    BOOLQ_RC_5SHOT = "boolq_rc_5shot"
    BOOLQ_RC_5SHOT_BPB = "boolq_rc_5shot_bpb"
    BOOLQ_MC_5SHOT = "boolq_mc_5shot"
    BOOLQ_MC_5SHOT_BPB = "boolq_mc_5shot_bpb"
    ARC_EASY_RC_5SHOT = "arc_easy_rc_5shot"
    ARC_EASY_RC_5SHOT_BPB = "arc_easy_rc_5shot_bpb"
    ARC_EASY_MC_5SHOT = "arc_easy_mc_5shot"
    ARC_EASY_MC_5SHOT_BPB = "arc_easy_mc_5shot_bpb"
    ARC_CHALLENGE_RC_5SHOT = "arc_challenge_rc_5shot"
    ARC_CHALLENGE_RC_5SHOT_BPB = "arc_challenge_rc_5shot_bpb"
    ARC_CHALLENGE_MC_5SHOT = "arc_challenge_mc_5shot"
    ARC_CHALLENGE_MC_5SHOT_BPB = "arc_challenge_mc_5shot_bpb"
    CSQA_RC_5SHOT = "csqa_rc_5shot"
    CSQA_RC_5SHOT_BPB = "csqa_rc_5shot_bpb"
    CSQA_MC_5SHOT = "csqa_mc_5shot"
    CSQA_MC_5SHOT_BPB = "csqa_mc_5shot_bpb"
    SOCIALIQA_RC_5SHOT = "socialiqa_rc_5shot"
    SOCIALIQA_RC_5SHOT_BPB = "socialiqa_rc_5shot_bpb"
    SOCIALIQA_MC_5SHOT = "socialiqa_mc_5shot"
    SOCIALIQA_MC_5SHOT_BPB = "socialiqa_mc_5shot_bpb"
    MMLU_STEM_VAR_BPB = "mmlu_stem_var_bpb"
    MMLU_HUMANITIES_VAR_BPB = "mmlu_humanities_var_bpb"
    MMLU_SOCIAL_SCIENCES_VAR_BPB = "mmlu_social_sciences_var_bpb"
    MMLU_OTHER_VAR_BPB = "mmlu_other_var_bpb"
    MMLU_STEM_BPB = "mmlu_stem_bpb"
    MMLU_HUMANITIES_BPB = "mmlu_humanities_bpb"
    MMLU_SOCIAL_SCIENCES_BPB = "mmlu_social_sciences_bpb"
    MMLU_OTHER_BPB = "mmlu_other_bpb"
