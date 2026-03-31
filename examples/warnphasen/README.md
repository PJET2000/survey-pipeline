# Warnphasen – WEXICOM IV Example Configuration

This directory contains the exact configuration for the original German DWD warning phase study from:

> Toussaint, P., Dietze, S., & Marocko, S. (2026). LLM-Based Survey Simulation as a Pre-Study Tool. [preprint]

## What this study is

The study evaluates nine German weather warning terms used by the German Weather Service (Deutscher Wetterdienst, DWD) on four perception dimensions:

| Term | Translation |
|------|-------------|
| Warnausblick | Warning outlook |
| Warnprognose | Warning forecast |
| Frühwarnung | Early warning |
| Vorwarnung | Pre-warning |
| Vorabinformation | Advance information |
| Warnung | Warning |
| Akutwarnung | Acute warning |
| Alarm | Alarm |
| Warnsignal | Warning signal |

**Questions:**
1. **Verständlichkeit** (Clarity) – 1–7 Likert scale
2. **Eintrittssicherheit** (Probability of occurrence) – 0–100% continuous
3. **Eintrittszeitpunkt** (Expected timing) – 1–4 ordinal
4. **Eintrittswahrscheinlichkeit** (Probability category) – 1–3 categorical

The pipeline was applied to human reference data from the independently conducted WEXICOM IV project — a population survey by the Disaster Research Unit (Katastrophenforschungsstelle, KFS) at Freie Universität Berlin (Schulze, 2025). The WEXICOM IV data is external to this pipeline and not produced by the pipeline authors.

## Required data

The human survey data file (`data_human.csv`) is **not included** in this repository for privacy reasons. It contains individual-level survey responses from a German population sample collected as part of WEXICOM IV.

If you have access to the human data, place it at the root of the pipeline directory (or specify its path with `--human-input`) and run:

```bash
python run.py --config examples/warnphasen/config_warnphasen.py analyze \
    --inputs "results/*.csv" --human-input data_human.csv
```

## Running the LLM benchmark (no human data required)

```bash
# From the llm-survey-pipeline directory:
python run.py --config examples/warnphasen/config_warnphasen.py run --repeats 5
```

This will query the configured LLMs and produce heatmaps and violin plots in `figs/`.

## CSV format expected for human data

The human CSV should have columns named `f25_1` through `f25_9` (Verständlichkeit),
`f26_1` through `f26_9` (Eintrittssicherheit), `f27_1` through `f27_9` (Eintrittszeitpunkt),
and `f29_1` through `f29_9` (Eintrittswahrscheinlichkeit).

The index (1–9) maps to stimuli as follows:
1 = Vorabinformation, 2 = Warnprognose, 3 = Akutwarnung, 4 = Alarm, 5 = Warnung,
6 = Frühwarnung, 7 = Warnsignal, 8 = Vorwarnung, 9 = Warnausblick.

Special values: 999 = "weiß nicht" (don't know), 998 = "keine Aussage" (no statement).

## Notes on German umlaut handling

The config overrides the default stimulus canonicalization to handle umlauts:
model responses containing "ue", "ae", "oe" digraphs (e.g., "Fruehwarnung") are
automatically matched to the canonical umlaut form ("Frühwarnung").
