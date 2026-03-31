#!/usr/bin/env python3
"""
run.py – CLI entry point for the LLM survey simulation pipeline.

Edit config.py to configure your study, then run:
  python run.py run
  python run.py analyze --inputs "results/*.csv"
  python run.py analyze --inputs "results/*.csv" --human-input data.csv
  python run.py analyze-human --input data.csv
  python run.py report --metrics results/metrics_llm_vs_human.csv

To use a different config file:
  python run.py --config examples/warnphasen/config_warnphasen.py run
  python run.py --config examples/warnphasen/config_warnphasen.py analyze --inputs "results/*.csv"
"""

import argparse
import importlib.util
import os
import sys
from datetime import datetime, timezone

import pandas as pd


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str):
    """Dynamically load a config module and return its `config` object."""
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        sys.exit(f"Config file not found: {abs_path}")
    spec = importlib.util.spec_from_file_location("_study_config", abs_path)
    mod = importlib.util.module_from_spec(spec)
    # Add the config file's directory to sys.path so relative imports work
    config_dir = os.path.dirname(abs_path)
    if config_dir not in sys.path:
        sys.path.insert(0, config_dir)
    spec.loader.exec_module(mod)
    cfg = getattr(mod, "config", None)
    if cfg is None:
        sys.exit(f"Config file {abs_path} does not define a `config` variable.")
    return cfg


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_run(args, config):
    from pipeline import run_benchmark, load_llm_data
    from analyze import plot_heatmaps, plot_violins

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figs_dir, exist_ok=True)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = config.default_models

    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    study_slug = config.study_name.lower().replace(" ", "_")[:30]
    out_csv = os.path.join(config.results_dir, f"{study_slug}_{ts}.csv")

    run_benchmark(models, args.repeats, out_csv, config, workers=args.workers or None)
    print(f"Done. CSV: {out_csv}")

    df = load_llm_data(out_csv, config)
    q_filter = _parse_q_filter(args.heatmap_q, config) if args.heatmap_q else None
    plot_heatmaps(df, config, config.figs_dir, q_filter=q_filter)
    if not args.no_violins:
        plot_violins(df, config, config.figs_dir)
    print(f"Plots in: {config.figs_dir}")


def cmd_analyze(args, config):
    from pipeline import load_llm_data, load_human_data
    from analyze import plot_heatmaps, plot_violins, plot_qq, export_metrics

    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(config.figs_dir, exist_ok=True)

    df = load_llm_data(args.inputs, config)
    first_model = None

    if args.human_input:
        df_h = load_human_data(
            args.human_input,
            config,
            model=args.human_model,
            prompt_mode=args.human_prompt_mode,
        )
        df = pd.concat([df, df_h], ignore_index=True)
        first_model = args.human_model

    q_filter = _parse_q_filter(args.heatmap_q, config) if args.heatmap_q else None
    plot_heatmaps(df, config, config.figs_dir, first_model=first_model, q_filter=q_filter)

    if not args.no_violins:
        plot_violins(df, config, config.figs_dir,
                     human_model=args.human_model,
                     human_prompt_mode=args.human_prompt_mode)

    if args.human_input:
        for q in config.questions:
            if q.metric_type == "continuous":
                plot_qq(df, config, q.id, config.figs_dir,
                        human_model=args.human_model,
                        human_prompt_mode=args.human_prompt_mode)
        metrics_path = os.path.join(config.results_dir, "metrics_llm_vs_human.csv")
        out = export_metrics(df, config, metrics_path,
                             human_model=args.human_model,
                             human_prompt_mode=args.human_prompt_mode)
        if out:
            print(f"Metrics: {out}")

    print(f"Done. Plots in: {config.figs_dir}")


def cmd_analyze_human(args, config):
    from pipeline import load_human_data
    from analyze import plot_heatmaps, plot_violins

    os.makedirs(config.figs_dir, exist_ok=True)

    df = load_human_data(args.input, config, model=args.model, prompt_mode=args.prompt_mode)

    if args.export_long:
        out_dir = os.path.dirname(args.export_long)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        df.to_csv(args.export_long, index=False)
        print(f"Human long-format: {args.export_long}")

    q_filter = _parse_q_filter(args.heatmap_q, config) if args.heatmap_q else None
    plot_heatmaps(df, config, config.figs_dir, first_model=args.model, q_filter=q_filter)

    if not args.no_violins:
        plot_violins(df, config, config.figs_dir,
                     human_model=args.model,
                     human_prompt_mode=args.prompt_mode)

    print(f"Done. Plots in: {config.figs_dir}")


def cmd_report(args, config):
    import subprocess
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "metrics_report.py"),
        "--metrics", args.metrics,
        "--out-dir", config.results_dir,
        "--fig-dir", config.figs_dir,
    ]
    if args.config_path:
        cmd += ["--config", args.config_path]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_q_filter(arg: str, config):
    if not arg:
        return None
    items = [x.strip() for x in arg.split(",") if x.strip()]
    q_ids = {q.id for q in config.questions}
    unknown = [x for x in items if x not in q_ids]
    if unknown:
        sys.exit(f"Unknown q_id(s): {', '.join(unknown)}")
    return items


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="LLM survey simulation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config", default="config.py",
        help="Path to config.py (default: config.py in current directory)",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # run
    pr = sub.add_parser("run", help="Run LLM benchmark → CSV (+ plots)")
    pr.add_argument("--models", default="", help="Comma-separated model IDs (default: from config)")
    pr.add_argument("--repeats", type=int, default=5, help="Repetitions per model")
    pr.add_argument("--workers", type=int, default=0, help="Parallel workers (0 = auto)")
    pr.add_argument("--heatmap-q", default="", help="Only plot these questions in heatmaps (comma-separated q_ids)")
    pr.add_argument("--no-violins", action="store_true", help="Skip violin plots")
    pr.set_defaults(func=cmd_run)

    # analyze
    pa = sub.add_parser("analyze", help="Generate plots from CSV(s)")
    pa.add_argument("--inputs", default=os.path.join("results", "*.csv"), help="CSV file or glob pattern")
    pa.add_argument("--human-input", default="", help="Optional: path to human survey CSV")
    pa.add_argument("--human-model", default="human", help="Label for human rows")
    pa.add_argument("--human-prompt-mode", default="participant", help="Prompt mode label for human rows")
    pa.add_argument("--heatmap-q", default="", help="Only plot these questions in heatmaps")
    pa.add_argument("--no-violins", action="store_true", help="Skip violin plots")
    pa.set_defaults(func=cmd_analyze)

    # analyze-human
    ph = sub.add_parser("analyze-human", help="Plots from human survey CSV only")
    ph.add_argument("--input", default="data_human.csv", help="Human survey CSV")
    ph.add_argument("--model", default="human", help="Label for the model column")
    ph.add_argument("--prompt-mode", default="participant", help="Prompt mode label")
    ph.add_argument("--export-long", default="", help="Optional: save human CSV in LLM long format")
    ph.add_argument("--heatmap-q", default="", help="Only plot these questions in heatmaps")
    ph.add_argument("--no-violins", action="store_true", help="Skip violin plots")
    ph.set_defaults(func=cmd_analyze_human)

    # report
    prp = sub.add_parser("report", help="Summarize metrics CSV into tables and plots")
    prp.add_argument(
        "--metrics",
        default=os.path.join("results", "metrics_llm_vs_human.csv"),
        help="Path to metrics CSV",
    )
    prp.set_defaults(func=cmd_report)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Store config path for the report subcommand
    args.config_path = args.config if args.config != "config.py" else ""

    args.func(args, config)


if __name__ == "__main__":
    main()
