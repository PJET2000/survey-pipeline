"""
config_warnphasen.py – Exact configuration for the original German DWD/WEXICOM IV study.

This replicates the study from:
  Toussaint, P., Dietze, S., & Marocko, S. (2026).
  LLM-Based Survey Simulation as a Pre-Study Tool. [preprint]

Usage:
  python run.py --config examples/warnphasen/config_warnphasen.py run
  python run.py --config examples/warnphasen/config_warnphasen.py analyze --inputs "results/*.csv"
  python run.py --config examples/warnphasen/config_warnphasen.py analyze \\
      --inputs "results/*.csv" --human-input data_human.csv

Human data (data_human.csv) is not included in this repository for privacy reasons.
See examples/warnphasen/README.md for details.
"""

import sys
import os

# Make sure pipeline.py is importable when running from a different working directory
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pipeline import StudyConfig, Question, _norm_stimulus

# ---------------------------------------------------------------------------
# German umlaut normalization
# The DWD warning terms contain no umlauts themselves, but model responses
# may use ae/oe/ue digraphs. This function normalizes both directions so that
# "Fruehwarnung" maps to the canonical "Frühwarnung".
# ---------------------------------------------------------------------------
def _norm_word(s: str) -> str:
    x = s.lower().replace(" ", "").replace("-", "")
    # Map digraphs to umlauts for matching
    x = x.replace("ae", "ä").replace("oe", "ö").replace("ue", "ü").replace("ss", "ß")
    return x


# ---------------------------------------------------------------------------
# Stimuli (original German DWD warning phase terms)
# ---------------------------------------------------------------------------
WORDS = [
    "Warnausblick", "Warnprognose", "Frühwarnung", "Vorwarnung",
    "Vorabinformation", "Warnung", "Akutwarnung", "Alarm", "Warnsignal",
]

# ---------------------------------------------------------------------------
# Questions (original German WEXICOM IV questions)
# ---------------------------------------------------------------------------
QUESTIONS = [
    Question(
        id="verstaendlichkeit",
        prompt_text=(
            "Wie verständlich beziehungsweise klar ist Ihnen, was mit dem jeweiligen Wort gemeint ist? "
            "1 = völlig unklar, 7 = völlig klar; -1 = weiß nicht."
        ),
        label="Verständlichkeit (1–7; −1)",
        scale_range=(1, 7),
        sentinels={-1},
        compare_stat="mean",
        metric_type="categorical",
        dist_categories=[
            ("1", 1), ("2", 2), ("3", 3), ("4", 4),
            ("5", 5), ("6", 6), ("7", 7), ("-1", -1),
        ],
        scale_labels=[
            "1 völlig unklar", "2", "3", "4", "5", "6", "7 völlig klar",
        ],
    ),
    Question(
        id="eintrittssicherheit",
        prompt_text=(
            "Wie sicher erscheint Ihnen das Eintreten des Gewitters, wenn Sie dieses Wort hören? "
            "Das Eintreten des Ereignisses ist von 0% (gar nicht sicher) bis 100% (völlig sicher); "
            "-1 = weiß nicht; -2 = Das Wort sagt nichts über die Sicherheit aus."
        ),
        label="Eintrittssicherheit (0–100; −1/−2)",
        scale_range=(0, 100),
        sentinels={-1, -2},
        compare_stat="mean",
        metric_type="continuous",
        dist_categories=[
            ("0-20", 10), ("21-40", 30), ("41-60", 50), ("61-80", 70), ("81-100", 90),
            ("-1", -1), ("-2", -2),
        ],
        scale_labels={
            0: "0% gar nicht sicher",
            25: "25%",
            50: "50%",
            75: "75%",
            100: "100% völlig sicher",
        },
    ),
    Question(
        id="eintrittszeitpunkt",
        prompt_text=(
            "Wann wird das Gewitter Ihrer Ansicht nach stattfinden? "
            "1 = sehr bald (in max. 1 Stunde), 2 = bald (in ca. 1 bis 6 Stunden), "
            "3 = später (in ca. 6 - 48 Stunden), 4 = viel später (mehr als 48 Stunden); "
            "-1 = weiß nicht; -2 = Das Wort sagt nichts über den Zeitpunkt aus."
        ),
        label="Eintrittszeitpunkt (1–4; −1/−2)",
        scale_range=(1, 4),
        sentinels={-1, -2},
        compare_stat="median",
        metric_type="categorical",
        dist_categories=[
            ("1", 1), ("2", 2), ("3", 3), ("4", 4), ("-1", -1), ("-2", -2),
        ],
        scale_labels=[
            "1 sehr bald (≤1h)",
            "2 bald (1–6h)",
            "3 später (6–48h)",
            "4 viel später (>48h)",
        ],
    ),
    Question(
        id="eintrittswahrscheinlichkeit",
        prompt_text=(
            "Welches Wort passt Ihrer Meinung nach am besten zu den folgenden Aussagen? "
            "1 = Gewitter sind möglich, 2 = Gewitter werden erwartet, 3 = Gewitter ist sichtbar; "
            "-1 = weiß nicht; -2 = Das Wort passt zu keiner der Aussagen."
        ),
        label="Eintrittswahrscheinlichkeit (1–3; −1/−2)",
        scale_range=(1, 3),
        sentinels={-1, -2},
        compare_stat="median",
        metric_type="categorical",
        dist_categories=[
            ("1", 1), ("2", 2), ("3", 3), ("-1", -1), ("-2", -2),
        ],
        scale_labels=[
            "1 möglich",
            "2 erwartet",
            "3 sichtbar",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Prompt roles (original German WEXICOM IV text)
# ---------------------------------------------------------------------------
_WEXICOM_INTRO = (
    "Die Katastrophenforschungsstelle (KFS) der Freien Universität Berlin führt im Rahmen "
    "des Forschungsprojekts WEXICOM IV eine Studie zur Verbesserung von Wetterwarnungen durch. "
    "Das Ziel ist es, Wetterwarnungen klarer und verständlicher zu gestalten und sie besser "
    "an die Bedürfnisse der Bevölkerung anzupassen. In dieser Umfrage möchten wir mehr über "
    "Ihre Erfahrungen mit Extremwetter, Ihr Vertrauen in Wetterwarnungen sowie Ihre Wünsche "
    "und Vorstellungen zu Warnzeitpunkten und Begriffen erfahren. Im Fokus stehen dabei "
    "insbesondere Gewitterwarnungen – wie sie wahrgenommen, verstanden und genutzt werden. "
    "Nun interessiert uns, was Sie über die folgenden Wörter denken und welche Bedeutung "
    "Sie ihnen zuschreiben."
)

_WEXICOM_POPULATION_SUFFIX = (
    "Simuliere beim Beantworten der untenstehenden Fragen die durchschnittlichen "
    "Einschätzungen der Bevölkerung in Deutschland."
)

PROMPT_ROLES = {
    "participant": _WEXICOM_INTRO,
    "population": (
        "Die Katastrophenforschungsstelle (KFS) der Freien Universität Berlin führt im Rahmen "
        "des Forschungsprojekts WEXICOM IV eine Studie zur Verbesserung von Wetterwarnungen durch. "
        "Das Ziel ist es, Wetterwarnungen klarer und verständlicher zu gestalten und sie besser "
        "an die Bedürfnisse der Bevölkerung anzupassen. In dieser Umfrage möchten wir mehr über "
        "Ihre Erfahrungen mit Extremwetter, Ihr Vertrauen in Wetterwarnungen sowie Ihre Wünsche "
        "und Vorstellungen zu Warnzeitpunkten und Begriffen erfahren. Im Fokus stehen dabei "
        "insbesondere Gewitterwarnungen – wie sie wahrgenommen, verstanden und genutzt werden. "
        + _WEXICOM_POPULATION_SUFFIX
    ),
    "population_dist": (
        "Die Katastrophenforschungsstelle (KFS) der Freien Universität Berlin führt im Rahmen "
        "des Forschungsprojekts WEXICOM IV eine Studie zur Verbesserung von Wetterwarnungen durch. "
        "Das Ziel ist es, Wetterwarnungen klarer und verständlicher zu gestalten und sie besser "
        "an die Bedürfnisse der Bevölkerung anzupassen. In dieser Umfrage möchten wir mehr über "
        "Ihre Erfahrungen mit Extremwetter, Ihr Vertrauen in Wetterwarnungen sowie Ihre Wünsche "
        "und Vorstellungen zu Warnzeitpunkten und Begriffen erfahren. Im Fokus stehen dabei "
        "insbesondere Gewitterwarnungen – wie sie wahrgenommen, verstanden und genutzt werden. "
        + _WEXICOM_POPULATION_SUFFIX
    ),
}

# ---------------------------------------------------------------------------
# Prompt rules (original German)
# ---------------------------------------------------------------------------
RULES_SINGLE = (
    "Regeln: Nutze zur Beantwortung bitte nur die vorgegebenen Antwortoptionen. "
    "Antworte mit genau {n_stimuli} Zeilen 'Wort|Q1|Q2|Q3|Q4'. Nur ganze Zahlen."
)

RULES_DIST = (
    "Regeln: Gib die Häufigkeitsverteilungen in Prozent so an, wie die deutsche Bevölkerung "
    "antworten würde. Prozentwerte als ganze Zahlen; jede Liste muss exakt 100 ergeben. "
    "Antworte mit genau {n_stimuli} Zeilen 'Wort|Q1|Q2|Q3|Q4'. Jede Q-Angabe ist eine kommaseparierte "
    "Liste der Prozentwerte in der Reihenfolge der Antwortoptionen."
)

RULES_DIST_FALLBACK = (
    "Regeln: Antworte NUR mit {n_stimuli} Zeilen 'Wort|Q1|Q2|Q3|Q4'. "
    "Jede Q-Angabe ist eine kommagetrennte Liste ganzer Prozentwerte "
    "in der Reihenfolge der Antwortoptionen; jede Liste muss exakt 100 ergeben. "
    "Keine weiteren Texte, keine Tabellen, kein Markdown."
)

DIST_EXAMPLE = (
    "Beispiel (nur Format, keine echten Daten):\n"
    "Beispielwort|10,10,10,10,10,10,20,20|10,10,10,10,10,25,25|10,20,30,30,5,5|20,30,30,10,10"
)

# ---------------------------------------------------------------------------
# Models (original study)
# ---------------------------------------------------------------------------
DEFAULT_MODELS = [
    "openai/gpt-5.2",
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-4.5",
]

# ---------------------------------------------------------------------------
# Human data mapping (WEXICOM IV survey data_human.csv)
# Column format: f25_1 ... f25_9 (Verstaendlichkeit per word)
#                f26_1 ... f26_9 (Eintrittssicherheit)
#                f27_1 ... f27_9 (Eintrittszeitpunkt)
#                f29_1 ... f29_9 (Eintrittswahrscheinlichkeit)
# Word index: 1=Vorabinformation, 2=Warnprognose, ..., 9=Warnausblick
# ---------------------------------------------------------------------------
HUMAN_WORD_CODES = {
    1: "Vorabinformation",
    2: "Warnprognose",
    3: "Akutwarnung",
    4: "Alarm",
    5: "Warnung",
    6: "Frühwarnung",
    7: "Warnsignal",
    8: "Vorwarnung",
    9: "Warnausblick",
}

HUMAN_PREFIX_TO_QID = {
    "f25": "verstaendlichkeit",
    "f26": "eintrittssicherheit",
    "f27": "eintrittszeitpunkt",
    "f29": "eintrittswahrscheinlichkeit",
}

HUMAN_SENTINELS = {
    "verstaendlichkeit": {999: -1},
    "eintrittssicherheit": {998: -2, 999: -1},
    "eintrittszeitpunkt": {998: -2, 999: -1},
    "eintrittswahrscheinlichkeit": {998: -2, 999: -1},
}

# ---------------------------------------------------------------------------
# Logo files (original study used PNG logos)
# ---------------------------------------------------------------------------
LOGO_FILES = {
    "anthropic": "Claude_AI_logo.png",
    "google": "Gemini_logo.png",
    "openai": "OpenAI_Logo.png",
}

# ---------------------------------------------------------------------------
# Build config
# The prompts in this study use "Wörter" (German for "words") as the stimuli label,
# and the question lines use "Frage {i} ({qid}): {text}" format.
# Since build_prompt() uses "Question {i}" by default, we override the stimuli_label
# and note that the prompt format matches the original by adjusting rules_single/dist
# which already embed the German word "Wort" in the format string.
# ---------------------------------------------------------------------------
config = StudyConfig(
    study_name="Warnphasen WEXICOM IV",
    stimuli=WORDS,
    stimuli_label="Wörter",
    questions=QUESTIONS,
    prompt_modes=PROMPT_ROLES,
    dist_modes={"population_dist"},
    rules_single=RULES_SINGLE,
    rules_dist=RULES_DIST,
    rules_dist_fallback=RULES_DIST_FALLBACK,
    dist_example=DIST_EXAMPLE,
    default_models=DEFAULT_MODELS,
    dist_total=100,
    results_dir="results",
    figs_dir="figs",
    logo_dir="Logos",
    logo_files=LOGO_FILES,
    human_stimulus_codes=HUMAN_WORD_CODES,
    human_prefix_to_qid=HUMAN_PREFIX_TO_QID,
    human_sentinels=HUMAN_SENTINELS,
)

# ---------------------------------------------------------------------------
# Override stimulus canonicalization with German umlaut normalization.
# This ensures that model responses like "Fruehwarnung" or "frühwarnung"
# are correctly matched to the canonical "Frühwarnung".
# ---------------------------------------------------------------------------
config._stimulus_canon = {_norm_word(w): w for w in WORDS}

# Also patch build_prompt to use "Frage" instead of "Question" for the
# question lines, matching the original German prompt format exactly.
# We do this by monkey-patching the config object with a custom build_prompt
# wrapper that the pipeline can optionally call via config.build_prompt_override.
# However, since the pipeline's build_prompt() function uses hardcoded "Question {i}",
# and the original study used "Frage {i}", we need to override this.
# The cleanest approach: store a custom prompt builder on the config.
import re as _re


def _build_prompt_german(stimuli, prompt_mode, config, variant="default"):
    """German variant of build_prompt using 'Frage' and 'Wörter'."""
    from pipeline import TEMPERATURE  # noqa – just verifying import works
    role = config.prompt_modes[prompt_mode]
    n_stimuli = len(stimuli)
    n_questions = len(config.questions)

    lines = [role]
    if prompt_mode in config.dist_modes:
        if variant == "fallback":
            rules = config.rules_dist_fallback.replace("{n_stimuli}", str(n_stimuli)).replace("{n_questions}", str(n_questions))
            lines.append(rules)
            if config.dist_example:
                lines.append(config.dist_example)
        else:
            rules = config.rules_dist.replace("{n_stimuli}", str(n_stimuli)).replace("{n_questions}", str(n_questions))
            lines.append(rules)
    else:
        rules = config.rules_single.replace("{n_stimuli}", str(n_stimuli)).replace("{n_questions}", str(n_questions))
        lines.append(rules)

    lines.extend([f"{config.stimuli_label}:", *[f"- {s}" for s in stimuli], ""])

    for i, q in enumerate(config.questions, 1):
        lines.append(f"Frage {i} ({q.id}): {q.prompt_text}")
        if prompt_mode in config.dist_modes:
            labels = ", ".join(config.dist_labels(q.id))
            lines.append(f"Antwortoptionen (Reihenfolge Prozentliste): {labels}")

    return "\n".join(lines)


# Attach the German prompt builder to the config so run.py can detect and use it
config._build_prompt_fn = _build_prompt_german
