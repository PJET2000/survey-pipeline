#!/usr/bin/env python3
"""
Summarize results/metrics_llm_vs_human.csv into readable tables and plots.

Outputs (defaults):
- results/metrics_summary_by_q.csv
- results/metrics_summary_overall.csv
- results/metrics_report.md
- figs/metrics_jsd_heatmap.png
- figs/metrics_wasserstein.png
- figs/metrics_compare_abs_norm_range_heatmap.png

Usage:
  python metrics_report.py --metrics results/metrics_llm_vs_human.csv
  python metrics_report.py --metrics results/metrics_llm_vs_human.csv --config config.py
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


_LOGO_CACHE = {}

PROMPT_MARKER_LABELS = {
    "participant": "participant",
    "population": "population",
    "population_dist": "distribution",
}


# ---------------------------------------------------------------------------
# Config loading (optional)
# ---------------------------------------------------------------------------

def _load_config_module(path: str):
    """Dynamically load a config module and return its `config` object."""
    spec = importlib.util.spec_from_file_location("_study_config", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "config", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_model(model):
    return model.split("/")[-1] if isinstance(model, str) else str(model)


def _label(model, prompt_mode):
    return f"{_short_model(model)}|{prompt_mode}"


def _provider_from_model(model):
    m = (model or "").lower()
    if m.startswith("openai/"):
        return "openai"
    if m.startswith("anthropic/"):
        return "anthropic"
    if m.startswith("google/"):
        return "google"
    return "unknown"


def _ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)


def _load_logo_image(provider, logo_dir, logo_files):
    cache_key = (provider, logo_dir)
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]
    filename = logo_files.get(provider)
    if not filename:
        _LOGO_CACHE[cache_key] = None
        return None
    path = os.path.join(logo_dir, filename)
    if not os.path.exists(path):
        _LOGO_CACHE[cache_key] = None
        return None
    img = None
    png_path = os.path.splitext(path)[0] + ".png"
    if os.path.exists(png_path):
        _LOGO_CACHE[cache_key] = plt.imread(png_path)
        return _LOGO_CACHE[cache_key]
    if path.lower().endswith(".png"):
        img = plt.imread(path)
    else:
        try:
            import io
            import cairosvg
            from PIL import Image
            png_bytes = cairosvg.svg2png(url=path, output_width=64)
            img = Image.open(io.BytesIO(png_bytes))
        except Exception:
            if shutil.which("convert"):
                try:
                    subprocess.run(["convert", path, png_path], check=True)
                    img = plt.imread(png_path)
                except Exception:
                    img = None
            else:
                img = None
    _LOGO_CACHE[cache_key] = img
    return img


def _apply_logo_ylabels(ax, providers, y_positions, logo_dir, logo_files, zoom=0.14, x=-0.17):
    for prov, y in zip(providers, y_positions):
        img = _load_logo_image(prov, logo_dir, logo_files)
        if img is None:
            continue
        ab = AnnotationBbox(
            OffsetImage(img, zoom=zoom),
            (x, y),
            xycoords=ax.get_yaxis_transform(),
            frameon=False,
            box_alignment=(1.0, 0.5),
        )
        ab.set_clip_on(False)
        ax.add_artist(ab)


def _add_prompt_text(ax, prompt_modes, y_positions, x=-0.15):
    for mode, y in zip(prompt_modes, y_positions):
        ax.text(
            x, y,
            PROMPT_MARKER_LABELS.get(mode, mode),
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, clip_on=False,
        )


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _spearman(a, b):
    if len(a) < 2:
        return np.nan
    ra = pd.Series(a).rank(method="average")
    rb = pd.Series(b).rank(method="average")
    return ra.corr(rb)


def _rmse(a, b):
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return np.nan
    diff = a[mask] - b[mask]
    return float(np.sqrt((diff ** 2).mean()))


def _pairwise_agreement(a, b):
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    mask = a.notna() & b.notna()
    a = a[mask].to_numpy()
    b = b[mask].to_numpy()
    n = len(a)
    if n < 2:
        return np.nan
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = a[i] - a[j]
            db = b[i] - b[j]
            if da == 0 or db == 0:
                continue
            if da * db > 0:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return np.nan if total == 0 else concordant / total


def _rank_abs_diff(a, b):
    a = pd.Series(a, dtype="float64")
    b = pd.Series(b, dtype="float64")
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return np.nan
    ra = a[mask].rank(method="average")
    rb = b[mask].rank(method="average")
    return float((ra - rb).abs().mean())


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

def summarize_by_q(df, compare_stat_by_qid, q_ranges):
    rows = []
    stim_col = "stimulus" if "stimulus" in df.columns else "word"
    for (model, prompt_mode, q_id), g in df.groupby(["model", "prompt_mode", "q_id"]):
        mean_diff = g["mean_llm"] - g["mean_human"]
        median_diff = g["median_llm"] - g["median_human"]

        comp_stat = compare_stat_by_qid.get(q_id, "mean")
        if "compare_llm" in g.columns and "compare_human" in g.columns:
            comp_llm = g["compare_llm"]
            comp_hum = g["compare_human"]
        elif comp_stat == "median":
            comp_llm = g["median_llm"]
            comp_hum = g["median_human"]
        else:
            comp_llm = g["mean_llm"]
            comp_hum = g["mean_human"]
        comp_diff = comp_llm - comp_hum

        if "compare_diff_norm_range" in g.columns and "compare_abs_diff_norm_range" in g.columns:
            comp_diff_norm = g["compare_diff_norm_range"]
            comp_abs_diff_norm = g["compare_abs_diff_norm_range"]
        else:
            rng = q_ranges.get(q_id)
            width = float(rng[1] - rng[0]) if rng else np.nan
            if width and not np.isnan(width):
                comp_diff_norm = comp_diff / width
                comp_abs_diff_norm = comp_diff_norm.abs()
            else:
                comp_diff_norm = pd.Series(np.nan, index=g.index, dtype="float64")
                comp_abs_diff_norm = pd.Series(np.nan, index=g.index, dtype="float64")

        gap_diff = (g["mean_llm"] - g["median_llm"]) - (g["mean_human"] - g["median_human"])
        rows.append({
            "model": model,
            "prompt_mode": prompt_mode,
            "q_id": q_id,
            "n_stimuli": g[stim_col].nunique() if stim_col in g.columns else np.nan,
            "jsd_mean": g["js_divergence"].mean(skipna=True),
            "jsd_median": g["js_divergence"].median(skipna=True),
            "wasserstein_mean": g["wasserstein"].mean(skipna=True),
            "abs_mean_diff_mean": g["abs_mean_diff"].mean(skipna=True),
            "abs_median_diff_mean": g["abs_median_diff"].mean(skipna=True),
            "mean_diff_mean": mean_diff.mean(skipna=True),
            "median_diff_mean": median_diff.mean(skipna=True),
            "rmse_mean": _rmse(g["mean_llm"], g["mean_human"]),
            "rmse_median": _rmse(g["median_llm"], g["median_human"]),
            "spearman_mean": _spearman(g["mean_llm"], g["mean_human"]),
            "spearman_median": _spearman(g["median_llm"], g["median_human"]),
            "pairwise_agreement_mean": _pairwise_agreement(g["mean_llm"], g["mean_human"]),
            "pairwise_agreement_median": _pairwise_agreement(g["median_llm"], g["median_human"]),
            "rank_abs_diff_mean": _rank_abs_diff(g["mean_llm"], g["mean_human"]),
            "rank_abs_diff_median": _rank_abs_diff(g["median_llm"], g["median_human"]),
            "mean_median_gap_abs_mean": gap_diff.abs().mean(skipna=True),
            "compare_stat": comp_stat,
            "compare_diff_mean": comp_diff.mean(skipna=True),
            "compare_abs_diff_mean": comp_diff.abs().mean(skipna=True),
            "compare_diff_norm_range_mean": comp_diff_norm.mean(skipna=True),
            "compare_abs_diff_norm_range_mean": comp_abs_diff_norm.mean(skipna=True),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mean_bias_abs"] = out["mean_diff_mean"].abs()
    out["median_bias_abs"] = out["median_diff_mean"].abs()
    out["compare_bias_abs"] = out["compare_diff_mean"].abs()
    out["compare_bias_abs_norm_range"] = out["compare_diff_norm_range_mean"].abs()
    out["model_short"] = out["model"].apply(_short_model)
    out["label"] = out.apply(lambda r: _label(r["model"], r["prompt_mode"]), axis=1)
    return out


def summarize_overall(summary_by_q, jsd_qids, cont_qids):
    rows = []
    for (model, prompt_mode), g in summary_by_q.groupby(["model", "prompt_mode"]):
        g_jsd = g[g["q_id"].isin(jsd_qids)]
        g_cont = g[g["q_id"].isin(cont_qids)]
        rows.append({
            "model": model,
            "prompt_mode": prompt_mode,
            "jsd_mean": g_jsd["jsd_mean"].mean(skipna=True),
            "wasserstein_mean": g_cont["wasserstein_mean"].mean(skipna=True),
            "abs_mean_diff_mean": g["abs_mean_diff_mean"].mean(skipna=True),
            "abs_median_diff_mean": g["abs_median_diff_mean"].mean(skipna=True),
            "mean_diff_mean": g["mean_diff_mean"].mean(skipna=True),
            "median_diff_mean": g["median_diff_mean"].mean(skipna=True),
            "rmse_mean": g["rmse_mean"].mean(skipna=True),
            "rmse_median": g["rmse_median"].mean(skipna=True),
            "spearman_mean": g["spearman_mean"].mean(skipna=True),
            "spearman_median": g["spearman_median"].mean(skipna=True),
            "pairwise_agreement_mean": g["pairwise_agreement_mean"].mean(skipna=True),
            "pairwise_agreement_median": g["pairwise_agreement_median"].mean(skipna=True),
            "rank_abs_diff_mean": g["rank_abs_diff_mean"].mean(skipna=True),
            "rank_abs_diff_median": g["rank_abs_diff_median"].mean(skipna=True),
            "mean_median_gap_abs_mean": g["mean_median_gap_abs_mean"].mean(skipna=True),
            "compare_abs_diff_mean": g["compare_abs_diff_mean"].mean(skipna=True),
            "compare_diff_mean": g["compare_diff_mean"].mean(skipna=True),
            "compare_abs_diff_norm_range_mean": g["compare_abs_diff_norm_range_mean"].mean(skipna=True),
            "compare_diff_norm_range_mean": g["compare_diff_norm_range_mean"].mean(skipna=True),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["mean_bias_abs"] = out["mean_diff_mean"].abs()
    out["median_bias_abs"] = out["median_diff_mean"].abs()
    out["compare_bias_abs"] = out["compare_diff_mean"].abs()
    out["compare_bias_abs_norm_range"] = out["compare_diff_norm_range_mean"].abs()
    out["model_short"] = out["model"].apply(_short_model)
    out["label"] = out.apply(lambda r: _label(r["model"], r["prompt_mode"]), axis=1)
    return out


def _add_rank(frame, col, ascending=True, name=None):
    if col not in frame.columns:
        return
    rank = frame[col].rank(method="min", ascending=ascending, na_option="bottom")
    frame[name or f"rank_{col}"] = rank.astype("Int64")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_jsd_heatmap(summary_by_q, jsd_qids, q_labels, logo_dir, logo_files, out_path):
    cat = summary_by_q[summary_by_q["q_id"].isin(jsd_qids)].copy()
    if cat.empty:
        return
    order = cat.groupby("label")["jsd_mean"].mean().sort_values().index
    pivot = cat.pivot(index="label", columns="q_id", values="jsd_mean").reindex(order)
    pivot = pivot.reindex(columns=[q for q in jsd_qids if q in pivot.columns])

    fig_h = max(3.0, 0.4 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(7.0, fig_h), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(pivot.values), aspect="auto", cmap=cmap)

    ax.set_yticks(range(len(pivot.index)))
    label_meta = cat.drop_duplicates("label").set_index("label")[["model", "prompt_mode"]]
    ax.set_yticklabels([""] * len(pivot.index))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([q_labels.get(c, c) for c in pivot.columns], rotation=20, ha="right")
    ax.set_title("Mean JSD vs humans (lower is better)")
    fig.colorbar(im, ax=ax, label="Jensen-Shannon divergence")
    providers = [_provider_from_model(label_meta.loc[l, "model"]) for l in pivot.index]
    ax.tick_params(axis="y", pad=0, length=0)
    _apply_logo_ylabels(ax, providers, range(len(pivot.index)), logo_dir, logo_files)
    prompt_modes = [label_meta.loc[l, "prompt_mode"] for l in pivot.index]
    _add_prompt_text(ax, prompt_modes, range(len(pivot.index)))

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_metric_heatmap(
    summary_by_q,
    metric,
    all_qids,
    q_labels,
    logo_dir,
    logo_files,
    out_path,
    title,
    cmap="viridis",
    vmin=None,
    vmax=None,
    cbar_label=None,
    center_zero=False,
):
    if metric not in summary_by_q.columns:
        return
    pivot = summary_by_q.pivot(index="label", columns="q_id", values=metric)
    order = summary_by_q.groupby("label")[metric].mean().sort_values().index
    cols_present = [q for q in all_qids if q in pivot.columns]
    pivot = pivot.reindex(order).reindex(columns=cols_present)
    if pivot.empty:
        return

    values = pivot.values.astype(float)
    if center_zero and vmin is None and vmax is None:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax if vmax and not np.isnan(vmax) else None

    fig_h = max(3.0, 0.4 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(7.4, fig_h), constrained_layout=True)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#f0f0f0")
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)

    ax.set_yticks(range(len(pivot.index)))
    label_meta = summary_by_q.drop_duplicates("label").set_index("label")[["model", "prompt_mode"]]
    ax.set_yticklabels([""] * len(pivot.index))
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([q_labels.get(c, c) for c in pivot.columns], rotation=20, ha="right")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label or metric)
    providers = [_provider_from_model(label_meta.loc[l, "model"]) for l in pivot.index]
    ax.tick_params(axis="y", pad=0, length=0)
    _apply_logo_ylabels(ax, providers, range(len(pivot.index)), logo_dir, logo_files)
    prompt_modes = [label_meta.loc[l, "prompt_mode"] for l in pivot.index]
    _add_prompt_text(ax, prompt_modes, range(len(pivot.index)))

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_wasserstein(summary_by_q, cont_qids, out_path):
    sub = summary_by_q[summary_by_q["q_id"].isin(cont_qids)].copy()
    if sub.empty:
        return
    sub_agg = sub.groupby(["model", "prompt_mode"])["wasserstein_mean"].mean().reset_index()
    sub_agg = sub_agg.sort_values("wasserstein_mean")
    labels = [f"{_short_model(r['model'])}|{r['prompt_mode']}" for _, r in sub_agg.iterrows()]
    vals = sub_agg["wasserstein_mean"].tolist()

    fig_w = max(6.0, 0.6 * len(labels))
    fig, ax = plt.subplots(figsize=(fig_w, 4.0), constrained_layout=True)
    ax.bar(range(len(labels)), vals, color="#4c78a8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Wasserstein distance")
    ax.set_title("Continuous questions: mean Wasserstein vs humans (lower is better)")

    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _top_lines(df, metric, n=3, ascending=True, fmt=".3f"):
    sub = df.dropna(subset=[metric]).sort_values(metric, ascending=ascending).head(n)
    lines = []
    for _, r in sub.iterrows():
        lines.append(f"- {r['label']}: {r[metric]:{fmt}}")
    return lines


def write_report(summary_by_q, summary_overall, jsd_qids, all_qids, q_labels, out_path, metrics_path):
    summary_by_q = summary_by_q.copy()
    summary_overall = summary_overall.copy()
    if "compare_abs_diff_norm_range_mean" not in summary_by_q.columns and "compare_diff_norm_range_mean" in summary_by_q.columns:
        summary_by_q["compare_abs_diff_norm_range_mean"] = summary_by_q["compare_diff_norm_range_mean"].abs()
    if "compare_bias_abs_norm_range" not in summary_overall.columns and "compare_diff_norm_range_mean" in summary_overall.columns:
        summary_overall["compare_bias_abs_norm_range"] = summary_overall["compare_diff_norm_range_mean"].abs()

    lines = ["# Metrics report", "", f"Source: {metrics_path}", ""]
    lines.append("Lower is better for js_divergence, wasserstein, abs_mean_diff, abs_median_diff, rmse_mean, rmse_median, rank_abs_diff_mean, rank_abs_diff_median, mean_median_gap_abs_mean, compare_abs_diff_norm_range_mean.")
    lines.append("Higher is better for spearman and pairwise_agreement.")
    lines.append("Bias (mean_diff_mean / median_diff_mean) closer to 0 is better.")
    lines.append("Continuous questions are binned for JSD computation.")
    lines.append("Preferred stat comparison: see compare_stat column per question.")
    lines.append("Range-normalized values use (max-min) per question to make scales comparable.")
    lines.extend(["", "## Overall (preferred stat)", ""])
    lines.append("Top by preferred-stat abs diff (range-normalized):")
    lines.extend(_top_lines(summary_overall, "compare_abs_diff_norm_range_mean", ascending=True))
    lines.extend(["", "Top by preferred-stat abs diff:"])
    lines.extend(_top_lines(summary_overall, "compare_abs_diff_mean", ascending=True))
    lines.extend(["", "Top by js_divergence:"])
    lines.extend(_top_lines(summary_overall, "jsd_mean", ascending=True))
    lines.extend(["", "Top by wasserstein:"])
    lines.extend(_top_lines(summary_overall, "wasserstein_mean", ascending=True))
    lines.extend(["", "Closest preferred-stat bias (abs):"])
    lines.extend(_top_lines(summary_overall, "compare_bias_abs", ascending=True))
    lines.extend(["", "Closest preferred-stat bias (abs, range-normalized):"])
    lines.extend(_top_lines(summary_overall, "compare_bias_abs_norm_range", ascending=True))
    lines.extend(["", "## Diagnostics (raw mean/median across all questions)", ""])
    lines.append("Top by rmse_mean:")
    lines.extend(_top_lines(summary_overall, "rmse_mean", ascending=True))
    lines.extend(["", "Top by spearman (mean over questions):"])
    lines.extend(_top_lines(summary_overall, "spearman_mean", ascending=False))
    lines.extend(["", "Top by pairwise agreement (means):"])
    lines.extend(_top_lines(summary_overall, "pairwise_agreement_mean", ascending=False))
    lines.extend(["", "Closest mean bias (abs):"])
    lines.extend(_top_lines(summary_overall, "mean_bias_abs", ascending=True))
    lines.extend(["", "## Per question best", ""])
    for q_id in all_qids:
        metric = "jsd_mean"
        label = q_labels.get(q_id, q_id)
        lines.append(f"- {label} ({metric}):")
        sub = summary_by_q[summary_by_q["q_id"] == q_id]
        lines.extend(_top_lines(sub, metric, n=2, ascending=True))
    lines.extend(["", "## Per question best (preferred stat abs diff, range-normalized)", ""])
    for q_id in all_qids:
        label = q_labels.get(q_id, q_id)
        lines.append(f"- {label} (compare_abs_diff_norm_range_mean):")
        sub = summary_by_q[summary_by_q["q_id"] == q_id]
        lines.extend(_top_lines(sub, "compare_abs_diff_norm_range_mean", n=2, ascending=True))
    lines.extend(["", "## Per question best (preferred stat abs diff)", ""])
    for q_id in all_qids:
        label = q_labels.get(q_id, q_id)
        lines.append(f"- {label} (compare_abs_diff_mean):")
        sub = summary_by_q[summary_by_q["q_id"] == q_id]
        lines.extend(_top_lines(sub, "compare_abs_diff_mean", n=2, ascending=True))
    lines.extend(["", "## Diagnostics (raw mean/median, per question)", ""])
    for q_id in all_qids:
        label = q_labels.get(q_id, q_id)
        lines.append(f"- {label} (pairwise_agreement_mean):")
        sub = summary_by_q[summary_by_q["q_id"] == q_id]
        lines.extend(_top_lines(sub, "pairwise_agreement_mean", n=2, ascending=False))
    lines.extend(["", "## Output files"])
    lines.extend([
        "- results/metrics_summary_by_q.csv",
        "- results/metrics_summary_overall.csv",
        "- figs/metrics_jsd_heatmap.png",
        "- figs/metrics_wasserstein.png",
        "- figs/metrics_compare_abs_norm_range_heatmap.png",
    ])

    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Summarize metrics_llm_vs_human.csv")
    ap.add_argument("--metrics", default=os.path.join("results", "metrics_llm_vs_human.csv"))
    ap.add_argument("--out-dir", default="results", help="Output dir for summary CSV/MD")
    ap.add_argument("--fig-dir", default="figs", help="Output dir for plots")
    ap.add_argument(
        "--config", default="",
        help="Optional path to a config.py to get Q_LABELS display names"
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    stim_col = "stimulus" if "stimulus" in df.columns else "word"

    # Derive metadata from CSV (the metric_type and compare_stat columns are written by analyze.py)
    # With optional config override for Q_LABELS
    cfg = None
    if args.config:
        try:
            cfg = _load_config_module(args.config)
        except Exception as e:
            print(f"Warning: could not load config {args.config}: {e}", file=sys.stderr)

    all_qids = df["q_id"].dropna().unique().tolist()

    # Q_LABELS: from config if available, else use q_id as label
    if cfg is not None:
        q_labels = {q.id: q.label for q in cfg.questions}
        q_ranges = {q.id: q.scale_range for q in cfg.questions}
    else:
        q_labels = {q: q for q in all_qids}
        q_ranges = {}
        # Try to infer ranges from data
        for q_id in all_qids:
            sub = df[df["q_id"] == q_id]
            lo = sub[["mean_llm", "mean_human", "median_llm", "median_human"]].min().min()
            hi = sub[["mean_llm", "mean_human", "median_llm", "median_human"]].max().max()
            if not (np.isnan(lo) or np.isnan(hi)):
                q_ranges[q_id] = (lo, hi)

    # JSD qids = all qids (metric_type column tells us, but we compute JSD for all)
    jsd_qids = all_qids

    # Continuous qids = those with wasserstein data or metric_type=="continuous"
    if "metric_type" in df.columns:
        cont_qids = df[df["metric_type"] == "continuous"]["q_id"].dropna().unique().tolist()
    else:
        cont_qids = df[df["wasserstein"].notna()]["q_id"].dropna().unique().tolist()

    # Compare stat: from compare_stat column in CSV
    if "compare_stat" in df.columns:
        compare_stat_by_qid = (
            df.groupby("q_id")["compare_stat"].first().to_dict()
        )
    else:
        compare_stat_by_qid = {q: "mean" for q in all_qids}

    # Logo config (from study config or defaults)
    if cfg is not None:
        logo_dir = cfg.logo_dir
        logo_files = cfg.logo_files
    else:
        logo_dir = "Logos"
        logo_files = {
            "anthropic": "Claude_AI_logo.svg",
            "google": "Gemini_logo.svg",
            "openai": "OpenAI_Logo.svg",
        }

    df["abs_mean_diff"] = (df["mean_llm"] - df["mean_human"]).abs()
    df["abs_median_diff"] = (df["median_llm"] - df["median_human"]).abs()

    if "compare_stat" not in df.columns:
        df["compare_stat"] = df["q_id"].map(compare_stat_by_qid).fillna("mean")
    if "compare_llm" not in df.columns:
        df["compare_llm"] = np.where(df["compare_stat"] == "median", df["median_llm"], df["mean_llm"])
    if "compare_human" not in df.columns:
        df["compare_human"] = np.where(df["compare_stat"] == "median", df["median_human"], df["mean_human"])
    if "compare_diff" not in df.columns:
        df["compare_diff"] = df["compare_llm"] - df["compare_human"]
    if "compare_abs_diff" not in df.columns:
        df["compare_abs_diff"] = df["compare_diff"].abs()
    if "compare_diff_norm_range" not in df.columns or "compare_abs_diff_norm_range" not in df.columns:
        def _range_width(q_id):
            rng = q_ranges.get(q_id)
            if not rng:
                return np.nan
            return float(rng[1] - rng[0])
        widths = df["q_id"].map(_range_width)
        df["compare_diff_norm_range"] = df["compare_diff"] / widths
        df["compare_abs_diff_norm_range"] = df["compare_diff_norm_range"].abs()

    summary_by_q = summarize_by_q(df, compare_stat_by_qid, q_ranges)
    if summary_by_q.empty:
        print("No data to summarize.")
        return
    summary_overall = summarize_overall(summary_by_q, jsd_qids, cont_qids)

    _add_rank(summary_overall, "jsd_mean", ascending=True, name="rank_jsd")
    _add_rank(summary_overall, "wasserstein_mean", ascending=True, name="rank_wasserstein")
    _add_rank(summary_overall, "abs_mean_diff_mean", ascending=True, name="rank_abs_mean")
    _add_rank(summary_overall, "spearman_mean", ascending=False, name="rank_spearman")
    rank_cols = [c for c in ["rank_jsd", "rank_wasserstein", "rank_abs_mean", "rank_spearman"] if c in summary_overall.columns]
    if rank_cols:
        summary_overall["rank_sum"] = summary_overall[rank_cols].sum(axis=1, skipna=True)

    _ensure_dir(args.out_dir)
    summary_by_q.to_csv(os.path.join(args.out_dir, "metrics_summary_by_q.csv"), index=False)
    summary_overall.to_csv(os.path.join(args.out_dir, "metrics_summary_overall.csv"), index=False)

    write_report(
        summary_by_q, summary_overall,
        jsd_qids, all_qids, q_labels,
        os.path.join(args.out_dir, "metrics_report.md"),
        args.metrics,
    )

    plot_jsd_heatmap(
        summary_by_q, jsd_qids, q_labels, logo_dir, logo_files,
        os.path.join(args.fig_dir, "metrics_jsd_heatmap.png"),
    )
    plot_wasserstein(
        summary_by_q, cont_qids,
        os.path.join(args.fig_dir, "metrics_wasserstein.png"),
    )

    s_norm = summary_by_q["compare_abs_diff_norm_range_mean"].dropna()
    if s_norm.empty:
        vmax_norm = 0.3
    else:
        vmax_norm = float(s_norm.quantile(0.95))
        vmax_norm = min(0.3, vmax_norm)
        if vmax_norm <= 0:
            vmax_norm = float(s_norm.max())
        if vmax_norm <= 0:
            vmax_norm = 0.3

    plot_metric_heatmap(
        summary_by_q,
        "compare_abs_diff_norm_range_mean",
        all_qids,
        q_labels,
        logo_dir,
        logo_files,
        os.path.join(args.fig_dir, "metrics_compare_abs_norm_range_heatmap.png"),
        "Preferred-stat abs diff (range-normalized)",
        cmap="magma",
        vmin=0,
        vmax=vmax_norm,
        cbar_label="Range-normalized abs diff",
    )

    print("Wrote metrics summary and plots.")


if __name__ == "__main__":
    main()
