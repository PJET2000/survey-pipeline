"""
pipeline.py – Core pipeline: StudyConfig, API calls, prompt building, parsing,
benchmark runner, and data loading.

All functions accept a StudyConfig object; no global constants are used.
"""

import os
import sys
import csv
import re
import glob
import json
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API defaults (overridable via .env)
# ---------------------------------------------------------------------------
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "1.0"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
MAX_TOKENS_DIST = 1000
MAX_TOKENS_DIST_OPENAI = 20000
TIMEOUT = 20
TIMEOUT_DIST = 45
MAX_WORKERS = int(os.getenv("LLM_MAX_WORKERS", "128"))
REASONING_EFFORT_DIST = "low"
DIST_RETRIES = 2

SENTINEL_COLORS = {-1: "#ffdd88", -2: "#f4a261"}

CSV_HEADER = [
    "timestamp", "run_id", "model", "repeat", "prompt_mode",
    "stimulus", "q_id", "question_text", "answer_int", "valid", "error",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Question:
    id: str
    prompt_text: str
    label: str
    scale_range: tuple                  # (min, max) inclusive
    sentinels: set                      # valid out-of-range values, e.g. {-1}
    compare_stat: str = "mean"          # "mean" or "median"
    metric_type: str = "categorical"    # "categorical" or "continuous"
    dist_categories: list = None        # [(label_str, int_value), ...]; auto-derived if None
    scale_labels: object = None         # list[str] or dict[int, str] for plots


@dataclass
class StudyConfig:
    study_name: str
    stimuli: list
    stimuli_label: str = "Stimuli"
    questions: list = field(default_factory=list)
    prompt_modes: dict = field(default_factory=dict)   # {mode_name: role_text}
    dist_modes: set = None
    rules_single: str = ""
    rules_dist: str = ""
    rules_dist_fallback: str = ""
    dist_example: str = ""
    default_models: list = field(default_factory=list)
    dist_total: int = 100
    results_dir: str = "results"
    figs_dir: str = "figs"
    logo_dir: str = "Logos"
    logo_files: dict = field(default_factory=dict)     # {provider_prefix: filename}
    human_stimulus_codes: dict = None   # {int_code: stimulus_name}
    human_prefix_to_qid: dict = None    # {"col_prefix": "question_id"}
    human_sentinels: dict = None        # {qid: {raw_code: mapped_value}}

    def __post_init__(self):
        if self.dist_modes is None:
            self.dist_modes = {"population_dist"}
        self._q_by_id = {q.id: q for q in self.questions}
        self._stimulus_canon = {_norm_stimulus(s): s for s in self.stimuli}

    def is_valid(self, q_id: str, v) -> bool:
        if v is None:
            return False
        q = self._q_by_id.get(q_id)
        if q is None:
            return False
        try:
            v = int(v)
        except (TypeError, ValueError):
            return False
        if v in q.sentinels:
            return True
        lo, hi = q.scale_range
        return lo <= v <= hi

    def dist_labels(self, q_id: str) -> list:
        """Return ordered category labels for distribution mode."""
        q = self._q_by_id[q_id]
        if q.dist_categories:
            return [lbl for lbl, _ in q.dist_categories]
        lo, hi = q.scale_range
        labels = [str(i) for i in range(lo, hi + 1)]
        for s in sorted(q.sentinels):
            labels.append(str(s))
        return labels

    def dist_values(self, q_id: str) -> list:
        """Return ordered category values for distribution mode."""
        q = self._q_by_id[q_id]
        if q.dist_categories:
            return [v for _, v in q.dist_categories]
        lo, hi = q.scale_range
        values = list(range(lo, hi + 1))
        for s in sorted(q.sentinels):
            values.append(s)
        return values


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_stimulus(s: str) -> str:
    """Default stimulus normalization. Override _stimulus_canon after init for custom logic."""
    return s.lower().strip().replace(" ", "").replace("-", "")


def _join_err(*parts) -> str:
    return ";".join([p for p in parts if p])


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def build_prompt(stimuli: list, prompt_mode: str, config: StudyConfig, variant: str = "default") -> str:
    """
    Build the full prompt string for the given stimuli and mode.

    If config._build_prompt_fn is set (e.g., for language-specific overrides),
    delegates to that function instead of the default implementation.
    """
    fn = getattr(config, "_build_prompt_fn", None)
    if fn is not None:
        return fn(stimuli, prompt_mode, config, variant)

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
        lines.append(f"Question {i} ({q.id}): {q.prompt_text}")
        if prompt_mode in config.dist_modes:
            labels = ", ".join(config.dist_labels(q.id))
            lines.append(f"Answer options (order of percent list): {labels}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_block(text: str, config: StudyConfig) -> dict:
    """Parse 'Stimulus|Q1|Q2|...' single-value block."""
    if not text:
        return {}
    raw = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lines = [ln for ln in raw if "|" in ln and not ln.startswith("```")]

    def to_int(tok):
        m = re.search(r"-?\d+", tok)
        return int(m.group(0)) if m else None

    out = {}
    exp = 1 + len(config.questions)
    for ln in lines:
        parts = [p.strip() for p in ln.strip("|").split("|") if p.strip()][:exp]
        if len(parts) < exp:
            continue
        s = config._stimulus_canon.get(_norm_stimulus(parts[0]))
        if not s or s in out:
            continue
        vals = [to_int(x) for x in parts[1:exp]]
        if any(v is None for v in vals):
            continue
        out[s] = vals
    return out


def parse_distribution_block(text: str, config: StudyConfig) -> dict:
    """Parse 'Stimulus|Q1|Q2|...' distribution block with percent lists."""
    if not text:
        return {}
    raw = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    lines = [ln for ln in raw if "|" in ln and not ln.startswith("```")]

    def to_ints(tok):
        return [int(x) for x in re.findall(r"-?\d+", tok)]

    out = {}
    exp = 1 + len(config.questions)
    for ln in lines:
        parts = [p.strip() for p in ln.strip("|").split("|") if p.strip()][:exp]
        if len(parts) < exp:
            continue
        s = config._stimulus_canon.get(_norm_stimulus(parts[0]))
        if not s or s in out:
            continue
        q_lists = []
        ok = True
        for i, part in enumerate(parts[1:exp]):
            q_id = config.questions[i].id
            nums = to_ints(part)
            need = len(config.dist_labels(q_id))
            if len(nums) < need:
                ok = False
                break
            q_lists.append(nums[:need])
        if ok:
            out[s] = q_lists
    return out


def _normalize_percent_list(values: list, target: int = 100):
    """Robustly normalize a percent list to sum to target. Returns (normalized, error_str)."""
    if not values:
        return None, "dist_empty"
    total = sum(values)
    if total <= 0:
        return None, "dist_sum_zero"
    if total == target:
        return values, ""
    scale = target / float(total)
    scaled = [int(round(v * scale)) for v in values]
    diff = target - sum(scaled)
    if diff != 0:
        idx = max(range(len(scaled)), key=lambda i: scaled[i])
        scaled[idx] = max(0, scaled[idx] + diff)
    return scaled, "dist_sum_norm"


def _needs_retry_dist(raw: str, parsed: dict, err: str, expected_count: int) -> bool:
    if err and err.startswith("request_failed"):
        return True
    if not raw or not raw.strip():
        return True
    if len(parsed) < expected_count:
        return True
    return False


# ---------------------------------------------------------------------------
# OpenRouter API
# ---------------------------------------------------------------------------

def call_openrouter(
    model: str,
    messages: list,
    max_tokens: int = None,
    timeout: int = None,
    temperature: float = None,
    return_raw: bool = False,
    extra_payload: dict = None,
):
    """Make a chat completion call to OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY not found. Add it to .env or set it as an environment variable.")
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE if temperature is None else temperature,
        "max_tokens": MAX_TOKENS if max_tokens is None else max_tokens,
    }
    if extra_payload:
        payload.update(extra_payload)
    r = requests.post(
        BASE_URL, headers=headers, json=payload,
        timeout=TIMEOUT if timeout is None else timeout,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}")
    resp = r.json()
    choice = (resp.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                for key in ("text", "output_text", "content"):
                    if part.get(key) is not None:
                        text_parts.append(str(part.get(key)))
                        break
            elif isinstance(part, str):
                text_parts.append(part)
        content = "".join(text_parts)
    if content is None:
        content = ""
    if not str(content).strip():
        for key in ("text", "output_text", "output"):
            val = msg.get(key)
            if isinstance(val, str) and val.strip():
                content = val
                break
        if not str(content).strip():
            val = choice.get("text")
            if isinstance(val, str) and val.strip():
                content = val
    out = str(content).strip()
    return (out, resp) if return_raw else out


def _dist_call_params(model: str):
    extra = None
    max_tokens = MAX_TOKENS_DIST
    if model.startswith("openai/"):
        extra = {"reasoning": {"effort": REASONING_EFFORT_DIST}}
        max_tokens = MAX_TOKENS_DIST_OPENAI
    return extra, max_tokens


def _call_dist_prompt(model: str, stimuli: list, prompt_mode: str, variant: str, config: StudyConfig):
    prompt_txt = build_prompt(stimuli, prompt_mode=prompt_mode, config=config, variant=variant)
    extra, max_tokens = _dist_call_params(model)
    try:
        raw, raw_resp = call_openrouter(
            model,
            [{"role": "user", "content": prompt_txt}],
            max_tokens=max_tokens,
            timeout=TIMEOUT_DIST,
            return_raw=True,
            extra_payload=extra,
        )
        parsed_try = parse_distribution_block(raw, config)
        err_try = "block_incomplete" if len(parsed_try) < len(stimuli) else ""
    except Exception as e:
        parsed_try, raw, err_try, raw_resp = {}, "", f"request_failed: {e}", None
    return prompt_txt, raw, parsed_try, err_try, raw_resp


# ---------------------------------------------------------------------------
# CSV row builders
# ---------------------------------------------------------------------------

def _rows_from_single(stimuli, parsed, ts, run_id, model, rep, pm, err, config: StudyConfig):
    rows = []
    for s in stimuli:
        vals = parsed.get(s, [None] * len(config.questions))
        for i, q in enumerate(config.questions):
            v = vals[i] if i < len(vals) else None
            valid = 1 if config.is_valid(q.id, v) else 0
            rows.append([
                ts, run_id, model, rep, pm, s, q.id, q.prompt_text,
                ("" if v is None else v), valid,
                _join_err(err, ("" if valid else "invalid_value")),
            ])
    return rows


def _rows_from_distribution(stimuli, parsed, ts, run_id, model, rep, pm, err, config: StudyConfig):
    rows = []
    for s in stimuli:
        dist = parsed.get(s)
        if not dist:
            miss = _join_err(err, "dist_missing_stimulus")
            for q in config.questions:
                rows.append([ts, run_id, model, rep, pm, s, q.id, q.prompt_text, "", 0, miss])
            continue

        for i, q in enumerate(config.questions):
            perc = dist[i] if i < len(dist) else None
            if not perc:
                miss = _join_err(err, "dist_missing_q")
                rows.append([ts, run_id, model, rep, pm, s, q.id, q.prompt_text, "", 0, miss])
                continue

            perc_norm, perr = _normalize_percent_list(perc, target=config.dist_total)
            if perc_norm is None:
                miss = _join_err(err, perr)
                rows.append([ts, run_id, model, rep, pm, s, q.id, q.prompt_text, "", 0, miss])
                continue

            vals = config.dist_values(q.id)
            err_full = _join_err(err, perr)
            for v, pct in zip(vals, perc_norm):
                if pct <= 0:
                    continue
                valid = 1 if config.is_valid(q.id, v) else 0
                row_err = _join_err(err_full, ("" if valid else "invalid_value"))
                for _ in range(pct):
                    rows.append([ts, run_id, model, rep, pm, s, q.id, q.prompt_text, v, valid, row_err])
    return rows


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(models: list, repeats: int, out_csv: str, config: StudyConfig, workers: int = None) -> str:
    """Run the full benchmark and write results to out_csv. Returns out_csv path."""
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figs_dir, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    log_path = os.path.join(config.results_dir, "run_log.md")
    prompts = list(config.prompt_modes.keys())

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(CSV_HEADER)
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"# Run Log\n\nStudy: {config.study_name}\nRun ID: {run_id}\n\n")

    jobs = [(m, r, p) for m in models for r in range(1, repeats + 1) for p in prompts]
    total = len(jobs)
    log_lock = Lock()

    def one_job(model, rep, pm):
        stimuli = config.stimuli[:]
        random.shuffle(stimuli)

        attempts = max(1, DIST_RETRIES if pm in config.dist_modes else 1)
        attempt_logs = []
        parsed = {}
        err = ""

        if pm in config.dist_modes:
            combined = {}
            last_err = ""
            for attempt in range(1, attempts + 1):
                variant = "fallback" if attempt > 1 else "default"
                prompt_txt, raw, parsed_try, err_try, raw_resp = _call_dist_prompt(
                    model, stimuli, pm, variant, config
                )
                for k, v in parsed_try.items():
                    if k not in combined:
                        combined[k] = v
                last_err = err_try
                attempt_logs.append((attempt, prompt_txt, raw, err_try, raw_resp))
                if not _needs_retry_dist(raw, combined, err_try, len(stimuli)):
                    break
            missing = [s for s in stimuli if s not in combined]
            if missing:
                prompt_txt, raw, parsed_try, err_try, raw_resp = _call_dist_prompt(
                    model, missing, pm, "fallback", config
                )
                for k, v in parsed_try.items():
                    if k not in combined:
                        combined[k] = v
                if err_try:
                    last_err = err_try
                attempt_logs.append(("fill-missing", prompt_txt, raw, err_try, raw_resp))
            parsed = combined
            err = "" if len(parsed) == len(stimuli) else _join_err(last_err, "block_incomplete")
        else:
            prompt_txt = build_prompt(stimuli, prompt_mode=pm, config=config)
            try:
                raw, raw_resp = call_openrouter(
                    model,
                    [{"role": "user", "content": prompt_txt}],
                    return_raw=True,
                )
                parsed = parse_block(raw, config)
                err = "block_incomplete" if len(parsed) < len(stimuli) else ""
            except Exception as e:
                parsed, raw, err, raw_resp = {}, "", f"request_failed: {e}", None
            attempt_logs.append((1, prompt_txt, raw, err, raw_resp))

        log_parts = [f"## {model} | repeat {rep} | {pm}\n\n"]
        for attempt, prompt_txt, raw, err_log, raw_resp in attempt_logs:
            suffix = "" if attempt == 1 else (f" (retry {attempt})" if isinstance(attempt, int) else f" ({attempt})")
            log_parts.append(f"### Prompt{suffix}\n\n```\n{prompt_txt}\n```\n\n")
            log_parts.append(f"### Response{suffix}\n\n```\n{raw}\n```\n\n")
            if err_log:
                log_parts.append(f"### Status{suffix}\n\n{err_log}\n\n")
            if (not raw or not raw.strip()) and raw_resp is not None:
                log_parts.append(f"### Raw Response{suffix}\n\n```json\n")
                log_parts.append(json.dumps(raw_resp, ensure_ascii=True, indent=2))
                log_parts.append("\n```\n\n")
        with log_lock:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("".join(log_parts))

        ts = datetime.now(timezone.utc).isoformat()
        if pm in config.dist_modes:
            return _rows_from_distribution(stimuli, parsed, ts, run_id, model, rep, pm, err, config)
        else:
            return _rows_from_single(stimuli, parsed, ts, run_id, model, rep, pm, err, config)

    if workers is None:
        cpu = os.cpu_count() or 4
        workers = min(MAX_WORKERS, max(1, cpu * 4))

    print(f"Starting {total} tasks with {workers} workers...")
    done = 0

    def _tick():
        pct = int(done * 100 / total) if total else 100
        sys.stdout.write(f"\rProgress: {done}/{total} ({pct}%)")
        sys.stdout.flush()

    _tick()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one_job, *job) for job in jobs]
        for fut in as_completed(futs):
            rows = fut.result()
            with open(out_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(rows)
            done += 1
            _tick()
    sys.stdout.write("\n")
    sys.stdout.flush()
    return out_csv


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _normalize_llm_frame(df: pd.DataFrame, config: StudyConfig) -> pd.DataFrame:
    df = df[df.get("valid", 0) == 1].copy() if "valid" in df.columns else df.copy()
    df["answer_int"] = pd.to_numeric(df["answer_int"], errors="coerce")
    stim_order = config.stimuli
    q_order = [q.id for q in config.questions]
    if "stimulus" in df.columns:
        df["stimulus"] = pd.Categorical(df["stimulus"], categories=stim_order, ordered=True)
    if "q_id" in df.columns:
        df["q_id"] = pd.Categorical(df["q_id"], categories=q_order, ordered=True)
    return df


def load_llm_data(pattern: str, config: StudyConfig) -> pd.DataFrame:
    """Load and normalize LLM result CSV files matching the glob pattern."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    return _normalize_llm_frame(df, config)


def load_human_data(
    path: str,
    config: StudyConfig,
    model: str = "human",
    prompt_mode: str = "participant",
) -> pd.DataFrame:
    """
    Load human survey data and convert to the same long format as LLM data.

    Requires config.human_stimulus_codes, config.human_prefix_to_qid,
    config.human_sentinels to be set. Raises ValueError if not configured.
    """
    if not config.human_stimulus_codes or not config.human_prefix_to_qid:
        raise ValueError(
            "Human data mapping is not configured. Set human_stimulus_codes, "
            "human_prefix_to_qid, and human_sentinels in your StudyConfig."
        )

    df = pd.read_csv(path)
    if "id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "id"})
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["id"] = df["id"].fillna(pd.Series(range(1, len(df) + 1), index=df.index))

    q_texts = {q.id: q.prompt_text for q in config.questions}
    sentinels = config.human_sentinels or {}

    parts = []
    for col in df.columns:
        m = re.match(r"([a-zA-Z]\w*)_([1-9]\d*)$", col)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        q_id = config.human_prefix_to_qid.get(prefix)
        stimulus = config.human_stimulus_codes.get(idx)
        if not q_id or not stimulus:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        series = series.replace(sentinels.get(q_id, {}))
        parts.append(pd.DataFrame({
            "timestamp": "",
            "run_id": "human",
            "model": model,
            "repeat": df["id"],
            "prompt_mode": prompt_mode,
            "stimulus": stimulus,
            "q_id": q_id,
            "question_text": q_texts.get(q_id, q_id),
            "answer_int": series,
        }))

    if not parts:
        raise ValueError(
            "No matching columns found. Check human_prefix_to_qid and column naming "
            "(expected pattern: prefix_index, e.g. f25_1)."
        )

    out = pd.concat(parts, ignore_index=True)
    out["valid"] = [
        1 if config.is_valid(q, v) else 0
        for q, v in zip(out["q_id"], out["answer_int"])
    ]
    out["error"] = np.where(out["valid"] == 1, "", "invalid_value")
    return _normalize_llm_frame(out, config)
