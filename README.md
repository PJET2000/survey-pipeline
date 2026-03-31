# LLM Survey Simulation Pipeline

A minimal, fully automated pipeline for simulating survey responses with LLMs and benchmarking them against human data. Originally developed for warning communication research, but designed to work with any structured survey study.

## What this is

You define a set of stimuli (words, images, scenarios) and a set of survey questions. The pipeline queries multiple LLMs via [OpenRouter](https://openrouter.ai), collects their responses, and produces heatmaps, violin plots, and quantitative metrics (Jensen-Shannon divergence, Wasserstein distance) comparing LLM distributions to human survey data.

Three prompt modes are supported:
- **participant** – the LLM answers as a single participant with a single value per question
- **population** – the LLM estimates the average response of the target population (single value)
- **population_dist** – the LLM estimates the full response distribution of the population as percent values per answer option


## Quick start

```bash
# 1. Clone and install
git clone https://github.com/your-org/llm-survey-pipeline.git
cd llm-survey-pipeline
pip install -r requirements.txt

# 2. Add your OpenRouter API key
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-...

# 3. Configure your study
# Edit config.py to define your stimuli, questions, and prompt roles

# 4. Run the benchmark
python run.py run --repeats 5

# 5. View results
# Plots are saved to figs/
# Raw data is saved to results/
```

## How it works

For each combination of (model, repeat, prompt_mode), the pipeline:
1. Shuffles the stimuli order (to reduce order effects)
2. Builds a prompt from the role text, rules, stimuli list, and questions
3. Calls the OpenRouter API and parses the structured response
4. Writes one row per (stimulus, question, repeat) to a CSV file

For `population_dist` mode, the model is asked to output percentage distributions. If the response is incomplete, an automatic fallback retry is triggered.

All raw prompts and responses are logged to `results/run_log.md` for transparency.

## Configuration guide

Edit `config.py` to configure your study. The key object is `StudyConfig`.

### Defining stimuli

```python
STIMULI = ["Term A", "Term B", "Term C"]
```

### Defining questions

**Categorical question (Likert scale):**

```python
Question(
    id="clarity",
    prompt_text="How clear is the meaning? 1=unclear, 7=clear; -1=don't know.",
    label="Clarity (1–7)",
    scale_range=(1, 7),
    sentinels={-1},           # valid out-of-range codes
    compare_stat="mean",       # "mean" or "median" for LLM-vs-human comparison
    metric_type="categorical", # uses JSD for quantitative comparison
    scale_labels=["1 unclear", "2", "3", "4", "5", "6", "7 clear"],
)
```

**Continuous question (probability estimate, 0–100):**

```python
Question(
    id="likelihood",
    prompt_text="How likely is the event? 0-100%; -1=don't know; -2=can't say.",
    label="Likelihood (0–100%)",
    scale_range=(0, 100),
    sentinels={-1, -2},
    compare_stat="mean",
    metric_type="continuous",  # uses JSD (binned) + Wasserstein
    dist_categories=[          # bin definitions for distribution mode
        ("0-20", 10), ("21-40", 30), ("41-60", 50),
        ("61-80", 70), ("81-100", 90),
        ("-1", -1), ("-2", -2),
    ],
    scale_labels={0: "0% not likely", 50: "50%", 100: "100% certain"},
)
```

### Prompt modes

```python
PROMPT_MODES = {
    "participant": "You are a study participant. Answer the following questions...",
    "population":  "Simulate the average response of the general population...",
    "population_dist": "Simulate the full response distribution of the general population...",
}
```

### Rules strings

Use `{n_stimuli}` and `{n_questions}` as placeholders:

```python
RULES_DIST = (
    "Reply with exactly {n_stimuli} lines 'Term|Q1|Q2|...|Q{n_questions}'. "
    "Each Q-value is a comma-separated list of percentages summing to 100."
)
```

## Output files

| File | Description |
|------|-------------|
| `results/<study>_<timestamp>.csv` | Raw LLM responses (one row per stimulus × question × repeat) |
| `results/run_log.md` | All prompts and raw model responses |
| `results/metrics_llm_vs_human.csv` | Quantitative metrics per (model, prompt_mode, q_id, stimulus) |
| `results/metrics_summary_by_q.csv` | Aggregated metrics per question |
| `results/metrics_summary_overall.csv` | Overall metrics per (model, prompt_mode) |
| `figs/heatmap_<q_id>.png` | Heatmap showing response distributions per stimulus |
| `figs/violin_<q_id>_<mode>_<model>.png` | Violin+boxplot distributions |
| `figs/qq_<q_id>_overlay.png` | Q-Q plot vs. human data (continuous questions) |
| `figs/metrics_*.png` | Metric comparison plots |

The CSV column for stimuli is `stimulus` (not `word` as in older versions).

## Human data comparison

To compare LLM responses against existing survey data, add three mappings to your `StudyConfig`:

```python
# Map integer codes in the human CSV to canonical stimulus names
human_stimulus_codes = {1: "Watch", 2: "Warning", 3: "Advisory"}

# Map column prefixes to question IDs
# Column "q_clarity_1" → question "clarity", stimulus code 1
human_prefix_to_qid = {"q_clarity": "clarity", "q_urgency": "urgency"}

# Map raw sentinel codes in the CSV to your question's sentinel values
human_sentinels = {
    "clarity": {999: -1},
    "urgency": {999: -1},
}
```

The human CSV should have columns like `prefix_index` (e.g., `q_clarity_1`, `q_clarity_2`), where `index` is the integer code for the stimulus.

Then run:

```bash
python run.py analyze --inputs "results/*.csv" --human-input data_human.csv
```

## Using the Warnphasen example

The `examples/warnphasen/` directory contains the complete configuration for the original German DWD warning phase study. To replicate that study:

```bash
python run.py --config examples/warnphasen/config_warnphasen.py run
python run.py --config examples/warnphasen/config_warnphasen.py analyze \
    --inputs "results/*.csv" --human-input data_human.csv
```

See `examples/warnphasen/README.md` for details on the required data.

## Supported models

Any model available via [OpenRouter](https://openrouter.ai/models) can be used. Specify models in `config.py` under `default_models`, or pass them on the command line:

```bash
python run.py run --models "openai/gpt-4o,anthropic/claude-opus-4-5,google/gemini-2.0-flash-001"
```

## Metrics

| Metric | Description |
|--------|-------------|
| **JSD** (Jensen-Shannon divergence) | Measures similarity between LLM and human response distributions. Range [0, 1]; lower is better. Computed for all questions; continuous questions are binned first. |
| **Wasserstein distance** | Earth mover's distance between LLM and human distributions on the raw continuous scale. Lower is better. Only for `metric_type="continuous"` questions. |
| **Range-normalized abs diff** | Absolute difference between preferred statistics (mean or median), divided by the question's range width. Makes scales comparable across questions. Lower is better. |
| **Spearman correlation** | Rank correlation of per-stimulus means/medians between LLM and humans across stimuli. Higher is better. |

## Citation

If you use this pipeline, please cite:

> Toussaint, P., & Marocko, S. (2026). LLM-Based Survey Simulation as a Pre-Study Tool. [preprint]
