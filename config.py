"""
config.py – Example study configuration for the LLM survey simulation pipeline.

Edit this file to configure your own study. This example evaluates public
perception of emergency communication terms in English.

Study: Emergency Communication Term Perception Study
-------------------------------------------------------
Stimuli: 5 terms used in US/UK emergency communication
Questions:
  1. clarity    – How clear is the meaning? (1–7 Likert, -1 = don't know)
  2. urgency    – How urgent does this feel? (1–4 ordinal, -1 = don't know)
  3. likelihood – How likely is the event? (0–100%, -1 = don't know, -2 = can't say)
"""

from pipeline import StudyConfig, Question

# ---------------------------------------------------------------------------
# Stimuli
# The terms you want to evaluate. These are shown to the LLM in each prompt.
# ---------------------------------------------------------------------------
STIMULI = ["Watch", "Warning", "Advisory", "Alert", "Emergency"]

# ---------------------------------------------------------------------------
# Questions
# Each question defines:
#   id           – unique identifier, used in filenames and CSV columns
#   prompt_text  – the question as it appears in the prompt
#   label        – display label for plots and reports
#   scale_range  – (min, max) of valid integer responses (inclusive)
#   sentinels    – set of valid out-of-range codes (e.g. -1 = don't know)
#   compare_stat – "mean" or "median" for LLM-vs-human comparison
#   metric_type  – "categorical" → JSD only
#                  "continuous"  → JSD (binned) + Wasserstein
#   dist_categories – [(label_str, int_value), ...] for distribution mode
#                     If None, auto-derived from scale_range + sentinels
#   scale_labels – labels for the colorbar in heatmaps
#                  list[str] (one per tick) or dict[int, str]
# ---------------------------------------------------------------------------
QUESTIONS = [
    Question(
        id="clarity",
        prompt_text=(
            "How clear is the meaning of this term to you? "
            "1 = completely unclear, 7 = completely clear; "
            "-1 = don't know."
        ),
        label="Clarity (1–7; −1)",
        scale_range=(1, 7),
        sentinels={-1},
        compare_stat="mean",
        metric_type="categorical",
        # dist_categories auto-derived: ("1",1), ("2",2), ..., ("7",7), ("-1",-1)
        scale_labels=[
            "1 completely unclear", "2", "3", "4", "5", "6", "7 completely clear",
        ],
    ),
    Question(
        id="urgency",
        prompt_text=(
            "How urgent does this term feel to you? "
            "1 = not urgent at all, 2 = somewhat urgent, "
            "3 = urgent, 4 = extremely urgent; "
            "-1 = don't know."
        ),
        label="Urgency (1–4; −1)",
        scale_range=(1, 4),
        sentinels={-1},
        compare_stat="median",
        metric_type="categorical",
        scale_labels=[
            "1 not urgent", "2 somewhat", "3 urgent", "4 extremely urgent",
        ],
    ),
    Question(
        id="likelihood",
        prompt_text=(
            "In your opinion, how likely is it that the described event will occur? "
            "Answer as a percentage from 0 to 100. "
            "0 = not at all likely, 100 = certain to occur; "
            "-1 = don't know; -2 = this term says nothing about likelihood."
        ),
        label="Likelihood (0–100%; −1/−2)",
        scale_range=(0, 100),
        sentinels={-1, -2},
        compare_stat="mean",
        metric_type="continuous",
        # For continuous questions, define bins explicitly.
        # Each tuple is (label_shown_in_prompt, bin_midpoint_value).
        dist_categories=[
            ("0-20", 10),
            ("21-40", 30),
            ("41-60", 50),
            ("61-80", 70),
            ("81-100", 90),
            ("-1", -1),
            ("-2", -2),
        ],
        scale_labels={
            0: "0% not likely",
            25: "25%",
            50: "50%",
            75: "75%",
            100: "100% certain",
        },
    ),
]

# ---------------------------------------------------------------------------
# Prompt modes
# Each mode maps a name to the role/context text shown before the questions.
# The LLM sees: role_text + rules + stimuli list + questions.
#
# Three standard modes:
#   participant     – LLM answers as a single participant
#   population      – LLM estimates average population response (single value)
#   population_dist – LLM estimates the full distribution over the population
#                     (percent values per answer option; tends to work best)
# ---------------------------------------------------------------------------
PARTICIPANT_ROLE = (
    "We are conducting a study on how people perceive emergency communication terms "
    "used in weather and public safety alerts. We would like to know how you personally "
    "interpret the following terms. Please answer each question based on your own understanding."
)

POPULATION_ROLE = (
    "We are conducting a study on how people perceive emergency communication terms "
    "used in weather and public safety alerts. We would like to know how you personally "
    "interpret the following terms. "
    "When answering, please simulate the average response of the general adult population "
    "in the United States."
)

POPULATION_DIST_ROLE = (
    "We are conducting a study on how people perceive emergency communication terms "
    "used in weather and public safety alerts. We would like to know how you personally "
    "interpret the following terms. "
    "When answering, please simulate the average response of the general adult population "
    "in the United States."
)

PROMPT_MODES = {
    "participant": PARTICIPANT_ROLE,
    "population": POPULATION_ROLE,
    "population_dist": POPULATION_DIST_ROLE,
}

# ---------------------------------------------------------------------------
# Prompt rules
# These strings are appended after the role text.
# Use {n_stimuli} and {n_questions} as placeholders – they are filled in automatically.
# ---------------------------------------------------------------------------

# Rules for single-value modes (participant, population)
RULES_SINGLE = (
    "Rules: Use only the provided answer options. "
    "Reply with exactly {n_stimuli} lines in the format 'Term|Q1|Q2|Q3'. "
    "Only whole integers."
)

# Rules for distribution mode
RULES_DIST = (
    "Rules: Provide frequency distributions in percent as the general US population would answer. "
    "Percentages as whole integers; each list must sum to exactly 100. "
    "Reply with exactly {n_stimuli} lines in the format 'Term|Q1|Q2|Q3'. "
    "Each Q-value is a comma-separated list of percentages in the order of the answer options."
)

# Fallback rules (used on retry if the model did not follow format)
RULES_DIST_FALLBACK = (
    "Rules: Reply ONLY with {n_stimuli} lines in the format 'Term|Q1|Q2|Q3'. "
    "Each Q-value is a comma-separated list of whole percentages "
    "in the order of the answer options; each list must sum to exactly 100. "
    "No additional text, no tables, no markdown."
)

# Example format shown in distribution mode (not real data)
DIST_EXAMPLE = (
    "Example (format only, not real data):\n"
    "ExampleTerm|10,10,10,10,10,10,20,20|10,30,30,20,10|10,20,30,20,10,5,5"
)

# ---------------------------------------------------------------------------
# Models to use by default (can be overridden via --models on the CLI)
# Any model available on OpenRouter can be used.
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "openai/gpt-4o",
    "anthropic/claude-opus-4-5",
    "google/gemini-2.0-flash-001",
]

# ---------------------------------------------------------------------------
# Optional: Human data mapping
# Set these if you want to compare LLM responses against existing survey data.
# Leave as None if you only run LLM simulations.
#
# human_stimulus_codes: {int_code_in_csv: canonical_stimulus_name}
#   Maps the integer index used in your survey CSV columns to stimulus names.
#   Example: column "q1_3" means question prefix "q1", stimulus index 3.
#
# human_prefix_to_qid: {"column_prefix": "question_id"}
#   Maps the column prefix in your CSV to the question id defined above.
#   Example: column "q_clarity_1" has prefix "q_clarity".
#
# human_sentinels: {qid: {raw_code: mapped_value}}
#   Maps raw numeric codes in the human data to the sentinel values used
#   in your questions (e.g. 999 in the original data → -1 = don't know).
# ---------------------------------------------------------------------------

# Example mapping (uncomment and adapt if you have human survey data):
# human_stimulus_codes = {
#     1: "Watch",
#     2: "Warning",
#     3: "Advisory",
#     4: "Alert",
#     5: "Emergency",
# }
# human_prefix_to_qid = {
#     "q_clarity": "clarity",
#     "q_urgency": "urgency",
#     "q_likelihood": "likelihood",
# }
# human_sentinels = {
#     "clarity": {999: -1},
#     "urgency": {999: -1},
#     "likelihood": {998: -2, 999: -1},
# }

human_stimulus_codes = None
human_prefix_to_qid = None
human_sentinels = None

# ---------------------------------------------------------------------------
# Output directories and logo files
# logo_files maps provider prefixes to image filenames in logo_dir.
# Supported formats: PNG (preferred), SVG (requires cairosvg or ImageMagick).
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
FIGS_DIR = "figs"
LOGO_DIR = "Logos"
LOGO_FILES = {
    "anthropic": "Claude_AI_logo.png",
    "google": "Gemini_logo.png",
    "openai": "OpenAI_Logo.png",
}

# ---------------------------------------------------------------------------
# Build the StudyConfig object
# This is the object that run.py and all pipeline functions use.
# ---------------------------------------------------------------------------
config = StudyConfig(
    study_name="Emergency Communication Term Perception Study",
    stimuli=STIMULI,
    stimuli_label="Terms",
    questions=QUESTIONS,
    prompt_modes=PROMPT_MODES,
    dist_modes={"population_dist"},
    rules_single=RULES_SINGLE,
    rules_dist=RULES_DIST,
    rules_dist_fallback=RULES_DIST_FALLBACK,
    dist_example=DIST_EXAMPLE,
    default_models=DEFAULT_MODELS,
    dist_total=100,
    results_dir=RESULTS_DIR,
    figs_dir=FIGS_DIR,
    logo_dir=LOGO_DIR,
    logo_files=LOGO_FILES,
    human_stimulus_codes=human_stimulus_codes,
    human_prefix_to_qid=human_prefix_to_qid,
    human_sentinels=human_sentinels,
)
