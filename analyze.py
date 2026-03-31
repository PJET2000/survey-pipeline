"""
analyze.py – Plotting and metrics export functions.

All functions accept a StudyConfig from pipeline.py. Import this module after
setting up your config.
"""

import os
import re
import io
import shutil
import subprocess
import zlib
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize, BoundaryNorm, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from pipeline import StudyConfig, SENTINEL_COLORS

_LOGO_CACHE = {}

PROMPT_MARKER_LABELS = {
    "participant": "participant",
    "population": "population",
    "population_dist": "distribution",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _in_range(q_id: str, series: pd.Series, config: StudyConfig) -> pd.Series:
    q = config._q_by_id[q_id]
    lo, hi = q.scale_range
    return series.where((series >= lo) & (series <= hi))


def _short_model(m):
    try:
        return m.split("/")[-1]
    except Exception:
        return str(m)


def _provider_from_model(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("openai/"):
        return "openai"
    if m.startswith("anthropic/"):
        return "anthropic"
    if m.startswith("google/"):
        return "google"
    return ""


def _load_logo_image(provider: str, config: StudyConfig):
    cache_key = (provider, config.logo_dir)
    if cache_key in _LOGO_CACHE:
        return _LOGO_CACHE[cache_key]
    filename = config.logo_files.get(provider)
    if not filename:
        _LOGO_CACHE[cache_key] = None
        return None
    path = os.path.join(config.logo_dir, filename)
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
            import cairosvg
            from PIL import Image
            png_bytes = cairosvg.svg2png(url=path, output_width=64)
            img = np.array(Image.open(io.BytesIO(png_bytes)))
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


def _apply_logo_rows(ax, providers, y_positions, config: StudyConfig, zoom=0.14, x=-0.095):
    for prov, y in zip(providers, y_positions):
        if not prov:
            continue
        img = _load_logo_image(prov, config)
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


def _add_prompt_text(ax, prompt_modes, y_positions, skip_rows=None, x=-0.09):
    skip = set(skip_rows or [])
    for idx, (pm, y) in enumerate(zip(prompt_modes, y_positions)):
        if idx in skip:
            continue
        ax.text(
            x, y,
            PROMPT_MARKER_LABELS.get(pm, pm),
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, clip_on=False,
        )


def _add_human_label(ax, y_positions, x=-0.09):
    for y in y_positions:
        ax.text(
            x, y, "humans",
            transform=ax.get_yaxis_transform(),
            ha="left", va="center", fontsize=9, clip_on=False,
        )


def _dist_bin_ranges(q_id: str, config: StudyConfig) -> dict:
    """Get numeric bin ranges from dist_categories for continuous questions."""
    q = config._q_by_id.get(q_id)
    if q is None or not q.dist_categories:
        return {}
    ranges = {}
    for lbl, v in q.dist_categories:
        m = re.match(r"^(\d+)\s*-\s*(\d+)$", str(lbl))
        if m:
            ranges[v] = (int(m.group(1)), int(m.group(2)))
    return ranges


def _jitter_in_bins(vals, bin_ranges: dict, seed_text: str) -> list:
    if not vals:
        return []
    seed = zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF
    rnd = random.Random(seed)
    out = []
    for v in vals:
        rng = bin_ranges.get(v)
        if rng:
            lo, hi = rng
            out.append(rnd.uniform(lo, hi))
        else:
            out.append(float(v))
    out.sort()
    return out


# ---------------------------------------------------------------------------
# Heatmaps
# ---------------------------------------------------------------------------

def plot_heatmaps(
    df: pd.DataFrame,
    config: StudyConfig,
    outdir: str,
    last_model=None,
    first_model=None,
    sort_cells=True,
    q_filter=None,
):
    """Plot one heatmap per question showing all (model, prompt_mode) × stimulus combinations."""
    models = sorted(df["model"].dropna().unique())
    if first_model and first_model in models:
        models = [first_model] + [m for m in models if m != first_model]
    elif last_model and last_model in models:
        models = [m for m in models if m != last_model] + [last_model]

    prompts = list(config.prompt_modes.keys())
    frame_color = "#6b6b6b"
    grid_lw = 2.0

    for q in config.questions:
        if q_filter and q.id not in q_filter:
            continue
        sub = df[df["q_id"] == q.id].copy()
        if sub.empty:
            continue
        sub["val"] = _in_range(q.id, sub["answer_int"], config)

        rows = []
        for m in models:
            for p in prompts:
                part = sub[(sub["model"] == m) & (sub["prompt_mode"] == p)]
                if not part.empty:
                    rows.append((m, p))
        if not rows:
            continue

        cols = config.stimuli
        fig, ax = plt.subplots(
            figsize=(max(10, 1.3 * len(cols)), max(4.0, 0.6 * len(rows))),
            constrained_layout=True,
        )

        lo, hi = q.scale_range
        if q.metric_type == "continuous":
            cmap = plt.colormaps.get_cmap("viridis")
            norm = Normalize(vmin=lo, vmax=hi)
            if q.scale_labels and isinstance(q.scale_labels, dict):
                cbar_ticks = sorted(q.scale_labels.keys())
            else:
                cbar_ticks = [lo, lo + (hi - lo) // 4, (lo + hi) // 2, lo + 3 * (hi - lo) // 4, hi]
        else:
            k = int(hi - lo + 1)
            base = plt.colormaps.get_cmap("viridis")
            colors = base(np.linspace(0.1, 0.9, k))
            cmap = ListedColormap(colors)
            boundaries = np.arange(lo - 0.5, hi + 1.5, 1)
            norm = BoundaryNorm(boundaries=boundaries, ncolors=cmap.N)
            cbar_ticks = list(range(lo, hi + 1))

        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([""] * len(rows))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right")
        ax.set_title(f"Heatmap – {q.label}")

        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, label="Value (in-range)")
        cbar.set_ticks(cbar_ticks)
        if q.scale_labels:
            if isinstance(q.scale_labels, dict):
                cbar.set_ticklabels([q.scale_labels.get(t, str(t)) for t in cbar_ticks])
            elif isinstance(q.scale_labels, list):
                cbar.set_ticklabels(q.scale_labels)

        dist_ranges = _dist_bin_ranges(q.id, config) if q.metric_type == "continuous" else {}

        for i, (m, p) in enumerate(rows):
            part = sub[(sub["model"] == m) & (sub["prompt_mode"] == p)]
            for j, w in enumerate(cols):
                cell = part[part["stimulus"] == w].sort_values("repeat")
                if cell.empty:
                    continue
                vals = cell["answer_int"].tolist()
                if q.metric_type == "continuous" and p in config.dist_modes:
                    vals_num = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().tolist()
                    sentinels_in_q = list(q.sentinels)
                    neg_vals = [v for v in vals_num if v in sentinels_in_q]
                    in_vals = [v for v in vals_num if v not in sentinels_in_q]
                    in_vals = _jitter_in_bins(in_vals, dist_ranges, f"{m}|{p}|{w}|{q.id}")
                    vals = neg_vals + in_vals
                elif sort_cells:
                    vals = sorted(vals)
                n = max(1, len(vals))
                left, bottom = j - 0.5, i - 0.5
                tile_w, tile_h = 1.0 / n, 1.0
                for k_idx, v in enumerate(vals):
                    x = left + k_idx * tile_w
                    if v in SENTINEL_COLORS:
                        face = SENTINEL_COLORS[v]
                    else:
                        face = cmap(norm(float(v)))
                    ax.add_patch(plt.Rectangle(
                        (x, bottom), tile_w, tile_h,
                        facecolor=face, edgecolor=None, linewidth=0,
                    ))

        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(-0.5, len(rows) - 0.5)
        ax.invert_yaxis()
        for rr in range(len(rows) + 1):
            ax.hlines(rr - 0.5, -0.5, len(cols) - 0.5, colors=frame_color, linewidth=grid_lw)
        for cc in range(len(cols) + 1):
            ax.vlines(cc - 0.5, -0.5, len(rows) - 0.5, colors=frame_color, linewidth=grid_lw)
        for spine in ax.spines.values():
            spine.set_color(frame_color)

        handles = [
            Patch(facecolor=SENTINEL_COLORS[-1], edgecolor="#8a7a42", label="-1 don't know"),
            Patch(facecolor=SENTINEL_COLORS[-2], edgecolor="#8a5b29", label="-2 no statement"),
        ]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=True, fontsize=8)

        providers = []
        human_rows = []
        prompt_modes_list = []
        for idx, (m, p) in enumerate(rows):
            if m == "human" and p == "participant":
                providers.append("")
                human_rows.append(idx)
            else:
                providers.append(_provider_from_model(m))
            prompt_modes_list.append(p)
        ax.tick_params(axis="y", pad=0, length=0)
        _apply_logo_rows(ax, providers, range(len(rows)), config)
        _add_prompt_text(ax, prompt_modes_list, range(len(rows)), skip_rows=human_rows)
        if human_rows:
            _add_human_label(ax, human_rows)

        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"heatmap_{q.id}.png"), dpi=180, bbox_inches="tight")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Violin plots
# ---------------------------------------------------------------------------

def plot_violins(
    df: pd.DataFrame,
    config: StudyConfig,
    outdir: str,
    human_model: str = "human",
    human_prompt_mode: str = "participant",
):
    """Plot violin+boxplot distributions per (model, prompt_mode, question)."""
    models = sorted(df["model"].dropna().unique())
    if human_model in models:
        models = [human_model] + [m for m in models if m != human_model]
    prompts = list(config.prompt_modes.keys())
    human_frame = df[
        (df["model"] == human_model) & (df["prompt_mode"] == human_prompt_mode)
    ].copy()

    stimuli = config.stimuli

    for model in models:
        for prompt_mode in prompts:
            for q in config.questions:
                sub = df[
                    (df["q_id"] == q.id) &
                    (df["prompt_mode"] == prompt_mode) &
                    (df["model"] == model)
                ].copy()
                if sub.empty:
                    continue
                vals = _in_range(q.id, sub["answer_int"], config)
                sub["val"] = vals
                data = [sub[sub["stimulus"] == s]["val"].dropna().values for s in stimuli]
                if all(len(x) == 0 for x in data):
                    continue
                data = [x if len(x) > 0 else np.array([np.nan]) for x in data]

                fig, ax = plt.subplots(figsize=(max(10, 1.3 * len(stimuli)), 6), constrained_layout=True)
                parts = ax.violinplot(data, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor("white")
                    pc.set_edgecolor("black")
                    pc.set_alpha(1)
                bp = ax.boxplot(
                    data, widths=0.15, vert=True, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#b22222", linewidth=1.8),
                )
                for box in bp["boxes"]:
                    box.set_facecolor("#f2f2f2")
                    box.set_edgecolor("#444")

                overlay_human = (
                    q.metric_type == "continuous"
                    and not human_frame.empty
                    and not (model == human_model and prompt_mode == human_prompt_mode)
                )
                if overlay_human:
                    hq = human_frame[human_frame["q_id"] == q.id]
                    if not hq.empty:
                        h_data = []
                        for s in stimuli:
                            h_vals = pd.to_numeric(hq[hq["stimulus"] == s]["answer_int"], errors="coerce")
                            h_vals = _in_range(q.id, h_vals, config).dropna().values
                            h_data.append(h_vals if len(h_vals) > 0 else np.array([np.nan]))
                        hb = ax.boxplot(
                            h_data, widths=0.10, vert=True, showfliers=False, patch_artist=True,
                            medianprops=dict(color="#1f77b4", linewidth=1.6),
                        )
                        for box in hb["boxes"]:
                            box.set_facecolor("#cfe3ff")
                            box.set_edgecolor("#1f77b4")
                            box.set_alpha(0.9)
                        for med in hb["medians"]:
                            med.set_color("#1f77b4")
                        for item in hb.get("whiskers", []):
                            item.set_color("#1f77b4")
                        for item in hb.get("caps", []):
                            item.set_color("#1f77b4")
                        for key in ("boxes", "medians", "whiskers", "caps"):
                            for item in hb.get(key, []):
                                item.set_zorder(3)

                ax.set_xticks(range(1, len(stimuli) + 1))
                ax.set_xticklabels(stimuli, rotation=30, ha="right")
                lo, hi = q.scale_range
                if q.metric_type == "continuous":
                    n_ticks = min(5, hi - lo + 1)
                    ax.set_yticks(np.linspace(lo, hi, n_ticks).astype(int))
                else:
                    ax.set_yticks(list(range(lo, hi + 1)))

                span = max(1.0, float(hi - lo))
                bar_h = max(0.2, 0.02 * span)
                base2 = lo - bar_h * 2.2
                base1 = lo - bar_h * 1.0
                ax.set_ylim(base2 - bar_h * 0.3, hi)
                ax.set_ylabel(q.label)
                ax.set_title(
                    f"Distribution – {q.label}\nModel: {_short_model(model)} | Prompt: {prompt_mode}"
                )

                shares_neg1, shares_neg2 = [], []
                for s in stimuli:
                    s_sub = sub[sub["stimulus"] == s]
                    tot = s_sub["answer_int"].notna().sum()
                    k1 = (s_sub["answer_int"] == -1).sum()
                    k2 = (s_sub["answer_int"] == -2).sum()
                    shares_neg1.append((k1 / tot) if tot > 0 else 0.0)
                    shares_neg2.append((k2 / tot) if tot > 0 else 0.0)

                for i, (p1, p2) in enumerate(zip(shares_neg1, shares_neg2), start=1):
                    w2 = 0.8 * p2
                    ax.add_patch(plt.Rectangle((i - w2 / 2, base2), w2, bar_h, color=SENTINEL_COLORS[-2], lw=0))
                    w1 = 0.8 * p1
                    ax.add_patch(plt.Rectangle((i - w1 / 2, base1), w1, bar_h, color=SENTINEL_COLORS[-1], lw=0, ec="#8a7a42"))

                handles = []
                if overlay_human:
                    handles.append(Patch(facecolor="#cfe3ff", edgecolor="#1f77b4", label="Humans (boxplot)"))
                handles.extend([
                    Patch(facecolor=SENTINEL_COLORS[-1], edgecolor="#8a7a42", label="-1 don't know"),
                    Patch(facecolor=SENTINEL_COLORS[-2], edgecolor="#8a5b29", label="-2 no statement"),
                ])
                ncol = 3 if overlay_human else 2
                fig.legend(
                    handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=True, fontsize=8,
                )

                os.makedirs(outdir, exist_ok=True)
                plt.savefig(
                    os.path.join(outdir, f"violin_{q.id}_{prompt_mode}_{_short_model(model)}.png"),
                    dpi=180, bbox_inches="tight",
                )
                plt.close(fig)


# ---------------------------------------------------------------------------
# Q-Q plot (for continuous questions)
# ---------------------------------------------------------------------------

def plot_qq(
    df: pd.DataFrame,
    config: StudyConfig,
    q_id: str,
    outdir: str,
    human_model: str = "human",
    human_prompt_mode: str = "participant",
):
    """Q-Q plot comparing LLM quantiles vs. human quantiles for a continuous question."""
    sub = df[df["q_id"] == q_id].copy()
    if sub.empty:
        return
    human = sub[(sub["model"] == human_model) & (sub["prompt_mode"] == human_prompt_mode)]
    if human.empty:
        return

    q = config._q_by_id[q_id]
    lo, hi = q.scale_range

    h_vals = pd.to_numeric(human["answer_int"], errors="coerce")
    h_vals = _in_range(q_id, h_vals, config).dropna()
    if len(h_vals) < 5:
        return

    qs = np.linspace(0.05, 0.95, 19)
    hq = np.quantile(h_vals, qs)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    ax.plot([lo, hi], [lo, hi], color="#777", linestyle="--", linewidth=1, label="y = x")

    models = sorted(sub["model"].dropna().unique())
    prompts = list(config.prompt_modes.keys())
    for m in models:
        for p in prompts:
            if m == human_model and p == human_prompt_mode:
                continue
            part = sub[(sub["model"] == m) & (sub["prompt_mode"] == p)]
            if part.empty:
                continue
            vals = pd.to_numeric(part["answer_int"], errors="coerce")
            vals = _in_range(q_id, vals, config).dropna()
            if len(vals) < 5:
                continue
            q_quantiles = np.quantile(vals, qs)
            ax.plot(hq, q_quantiles, marker="o", markersize=2.5, linewidth=1, alpha=0.85,
                    label=f"{_short_model(m)}-{p}")

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Human quantiles ({lo}–{hi})")
    ax.set_ylabel(f"LLM quantiles ({lo}–{hi})")
    ax.set_title(f"Q-Q Plot – {q.label} (LLM vs. Humans)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True, fontsize=8)

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, f"qq_{q_id}_overlay.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Metrics computation helpers
# ---------------------------------------------------------------------------

def _bin_continuous(series: pd.Series, q_id: str, config: StudyConfig) -> pd.Series:
    """
    Map continuous values to bin midpoints (from dist_categories).
    Sentinel values are passed through unchanged.
    """
    q = config._q_by_id[q_id]
    s = pd.to_numeric(series, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype="float64")

    sentinel_vals = q.sentinels
    sent_mask = s.isin(sentinel_vals)
    out[sent_mask] = s[sent_mask]

    if q.dist_categories:
        for lbl, midpoint in q.dist_categories:
            m_obj = re.match(r"^(\d+)\s*-\s*(\d+)$", str(lbl))
            if m_obj:
                lo_b, hi_b = int(m_obj.group(1)), int(m_obj.group(2))
                mask = (s >= lo_b) & (s <= hi_b)
                out[mask] = midpoint
    return out


def _dist_probs(series: pd.Series, categories: list) -> np.ndarray:
    counts = np.array([(series == c).sum() for c in categories], dtype=float)
    total = counts.sum()
    if total <= 0:
        return None
    return counts / total


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _wasserstein_1d(x, y) -> float:
    x = np.sort(np.array(x, dtype=float))
    y = np.sort(np.array(y, dtype=float))
    if len(x) == 0 or len(y) == 0:
        return np.nan
    wx = np.full(len(x), 1.0 / len(x))
    wy = np.full(len(y), 1.0 / len(y))
    vals = np.unique(np.concatenate([x, y]))
    i = j = 0
    cdf_x = cdf_y = 0.0
    dist = 0.0
    prev = vals[0]
    for v in vals:
        dist += abs(cdf_x - cdf_y) * (v - prev)
        while i < len(x) and x[i] == v:
            cdf_x += wx[i]
            i += 1
        while j < len(y) and y[j] == v:
            cdf_y += wy[j]
            j += 1
        prev = v
    return dist


# ---------------------------------------------------------------------------
# Metrics export
# ---------------------------------------------------------------------------

def export_metrics(
    df: pd.DataFrame,
    config: StudyConfig,
    out_csv: str,
    human_model: str = "human",
    human_prompt_mode: str = "participant",
) -> str:
    """
    Compute per-(model, prompt_mode, q_id, stimulus) metrics and write to CSV.
    Returns out_csv path or "" if nothing to export.

    Metrics:
    - JSD: computed for all questions (continuous ones are binned first)
    - Wasserstein: computed for continuous questions only
    - compare_stat: mean or median per question
    - metric_type: "categorical" or "continuous" (written to CSV for metrics_report.py)
    """
    df = df[df.get("valid", 0) == 1].copy() if "valid" in df.columns else df.copy()
    human = df[(df["model"] == human_model) & (df["prompt_mode"] == human_prompt_mode)]
    if human.empty:
        return ""

    rows = []
    for m in sorted(df["model"].dropna().unique()):
        if m == human_model:
            continue
        m_frame = df[df["model"] == m]
        for pm in sorted(m_frame["prompt_mode"].dropna().unique()):
            for q in config.questions:
                q_id = q.id
                dist_vals = config.dist_values(q_id)
                for stimulus in config.stimuli:
                    llm = m_frame[
                        (m_frame["prompt_mode"] == pm) &
                        (m_frame["q_id"] == q_id) &
                        (m_frame["stimulus"] == stimulus)
                    ]
                    hum = human[
                        (human["q_id"] == q_id) &
                        (human["stimulus"] == stimulus)
                    ]
                    if llm.empty or hum.empty:
                        continue

                    llm_vals = pd.to_numeric(llm["answer_int"], errors="coerce").dropna()
                    hum_vals = pd.to_numeric(hum["answer_int"], errors="coerce").dropna()
                    llm_in = _in_range(q_id, llm_vals, config).dropna()
                    hum_in = _in_range(q_id, hum_vals, config).dropna()

                    mean_llm = float(llm_in.mean()) if not llm_in.empty else np.nan
                    mean_hum = float(hum_in.mean()) if not hum_in.empty else np.nan
                    median_llm = float(llm_in.median()) if not llm_in.empty else np.nan
                    median_hum = float(hum_in.median()) if not hum_in.empty else np.nan

                    comp_stat = q.compare_stat
                    if comp_stat == "median":
                        comp_llm = median_llm
                        comp_hum = median_hum
                    else:
                        comp_llm = mean_llm
                        comp_hum = mean_hum

                    comp_diff = (
                        comp_llm - comp_hum
                        if (not np.isnan(comp_llm) and not np.isnan(comp_hum))
                        else np.nan
                    )
                    rng = float(q.scale_range[1] - q.scale_range[0])
                    if not np.isnan(comp_diff) and rng > 0:
                        comp_diff_norm_range = comp_diff / rng
                        comp_abs_diff_norm_range = abs(comp_diff_norm_range)
                    else:
                        comp_diff_norm_range = np.nan
                        comp_abs_diff_norm_range = np.nan

                    # JSD (all questions; continuous questions binned)
                    jsd = np.nan
                    if q.metric_type == "continuous":
                        llm_cat = _bin_continuous(llm_vals, q_id, config).dropna()
                        hum_cat = _bin_continuous(hum_vals, q_id, config).dropna()
                        p_dist = _dist_probs(llm_cat, dist_vals)
                        q_dist = _dist_probs(hum_cat, dist_vals)
                    else:
                        p_dist = _dist_probs(llm_vals, dist_vals)
                        q_dist = _dist_probs(hum_vals, dist_vals)
                    if p_dist is not None and q_dist is not None:
                        jsd = _js_divergence(p_dist, q_dist)

                    # Wasserstein (continuous questions only)
                    wass = np.nan
                    if q.metric_type == "continuous":
                        if not llm_in.empty and not hum_in.empty:
                            wass = _wasserstein_1d(llm_in.values, hum_in.values)

                    rows.append([
                        m, pm, q_id, stimulus,
                        len(llm_vals), len(hum_vals),
                        mean_llm, mean_hum, median_llm, median_hum,
                        comp_stat, comp_llm, comp_hum, comp_diff,
                        comp_diff_norm_range, comp_abs_diff_norm_range,
                        jsd, wass,
                        q.metric_type,
                    ])

    if not rows:
        return ""

    out = pd.DataFrame(rows, columns=[
        "model", "prompt_mode", "q_id", "stimulus",
        "n_llm", "n_human",
        "mean_llm", "mean_human", "median_llm", "median_human",
        "compare_stat", "compare_llm", "compare_human", "compare_diff",
        "compare_diff_norm_range", "compare_abs_diff_norm_range",
        "js_divergence", "wasserstein",
        "metric_type",
    ])
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out_csv
