import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _read_csv_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_metadata(path: Path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _detect_strategy(run_dir: Path, metadata, summary_rows):
    if summary_rows and "strategy" in summary_rows[0] and summary_rows[0]["strategy"]:
        return summary_rows[0]["strategy"]
    if "strategy" in metadata and metadata["strategy"]:
        return str(metadata["strategy"])

    name = run_dir.name
    if "_v" in name:
        parts = name.split("_")
        for part in parts:
            if part.startswith("v"):
                return part
    return "unknown"


def _discover_run_dirs(input_dir: Path):
    if (input_dir / "summary.csv").exists():
        return [input_dir]

    run_dirs = []
    for child in input_dir.iterdir():
        if child.is_dir() and (child / "summary.csv").exists():
            run_dirs.append(child)
    return sorted(run_dirs)


def _compute_strategy_summary(summary_rows):
    by_strategy = {}
    for row in summary_rows:
        strategy = row.get("strategy", "unknown")
        by_strategy.setdefault(strategy, []).append(row)

    baseline_elapsed = None
    if "v0" in by_strategy:
        v0_elapsed = [_to_float(r.get("elapsed_seconds")) for r in by_strategy["v0"]]
        v0_elapsed = [v for v in v0_elapsed if v is not None]
        if v0_elapsed:
            baseline_elapsed = float(np.mean(v0_elapsed))

    table = []
    for strategy in sorted(by_strategy.keys()):
        rows = by_strategy[strategy]
        best_vals = [_to_float(r.get("best_fitness")) for r in rows]
        elapsed_vals = [_to_float(r.get("elapsed_seconds")) for r in rows]
        eval_vals = [_to_float(r.get("fitness_eval_seconds")) for r in rows]
        update_vals = [_to_float(r.get("state_update_seconds")) for r in rows]
        workers_vals = [_to_int(r.get("workers")) for r in rows]

        best_vals = [v for v in best_vals if v is not None]
        elapsed_vals = [v for v in elapsed_vals if v is not None]
        eval_vals = [v for v in eval_vals if v is not None]
        update_vals = [v for v in update_vals if v is not None]
        workers_vals = [v for v in workers_vals if v is not None]

        mean_best = float(np.mean(best_vals)) if best_vals else np.nan
        mean_elapsed = float(np.mean(elapsed_vals)) if elapsed_vals else np.nan
        mean_eval = float(np.mean(eval_vals)) if eval_vals else np.nan
        mean_update = float(np.mean(update_vals)) if update_vals else np.nan

        if elapsed_vals and eval_vals and update_vals:
            overhead_vals = [
                e - fe - su for e, fe, su in zip(elapsed_vals, eval_vals, update_vals)
            ]
            mean_overhead = float(np.mean(overhead_vals))
        else:
            mean_overhead = np.nan

        if np.isfinite(mean_overhead) and np.isfinite(mean_elapsed) and mean_elapsed > 0.0:
            overhead_ratio = mean_overhead / mean_elapsed
        else:
            overhead_ratio = np.nan

        if baseline_elapsed is not None and np.isfinite(mean_elapsed) and mean_elapsed > 0.0:
            speedup = baseline_elapsed / mean_elapsed
        else:
            speedup = np.nan

        mean_workers = float(np.mean(workers_vals)) if workers_vals else np.nan
        if np.isfinite(speedup) and np.isfinite(mean_workers) and mean_workers > 0.0:
            efficiency = speedup / mean_workers
        else:
            efficiency = np.nan

        table.append(
            {
                "strategy": strategy,
                "n_runs": len(rows),
                "mean_best_fitness": mean_best,
                "mean_elapsed_seconds": mean_elapsed,
                "mean_fitness_eval_seconds": mean_eval,
                "mean_state_update_seconds": mean_update,
                "mean_overhead_seconds": mean_overhead,
                "overhead_ratio": overhead_ratio,
                "speedup_vs_v0": speedup,
                "efficiency": efficiency,
            }
        )

    return table


def _write_summary_csv(table, output_path: Path):
    headers = [
        "strategy",
        "n_runs",
        "mean_best_fitness",
        "mean_elapsed_seconds",
        "mean_fitness_eval_seconds",
        "mean_state_update_seconds",
        "mean_overhead_seconds",
        "overhead_ratio",
        "speedup_vs_v0",
        "efficiency",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(table)


def _plot_times(table, output_path: Path):
    labels = [row["strategy"] for row in table]
    values = [row["mean_elapsed_seconds"] for row in table]
    values = [v if np.isfinite(v) else 0.0 for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="tab:blue")
    ax.set_ylabel("Mean elapsed seconds")
    ax.set_xlabel("Strategy")
    ax.set_title("Mean total time by strategy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def _plot_convergence(history_rows, output_path: Path):
    by_strategy = {}
    for row in history_rows:
        strategy = row.get("strategy", "unknown")
        iteration = _to_int(row.get("iteration"))
        best_f = _to_float(row.get("best_fitness"))
        if iteration is None or best_f is None:
            continue
        by_strategy.setdefault(strategy, {}).setdefault(iteration, []).append(best_f)

    if not by_strategy:
        return False

    fig, ax = plt.subplots(figsize=(7, 4))
    plotted = False
    for strategy in sorted(by_strategy.keys()):
        per_iter = by_strategy[strategy]
        iters = sorted(per_iter.keys())
        means = [float(np.mean(per_iter[i])) for i in iters]
        ax.plot(iters, means, label=strategy, linewidth=2)
        plotted = True

    if not plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean best fitness")
    ax.set_title("Convergence by strategy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def _detect_final_fitness_column(summary_rows):
    if not summary_rows:
        return None
    candidates = ["best_fitness", "best_f", "final_fitness", "fitness"]
    sample_keys = summary_rows[0].keys()
    for name in candidates:
        if name in sample_keys:
            return name
    return None


def _plot_fitness_boxplot_by_strategy(summary_rows, fitness_col, output_path: Path):
    by_strategy = {}
    for row in summary_rows:
        strategy = row.get("strategy", "unknown")
        value = _to_float(row.get(fitness_col))
        if value is None:
            continue
        by_strategy.setdefault(strategy, []).append(value)

    labels = sorted(by_strategy.keys())
    data = [by_strategy[label] for label in labels if by_strategy[label]]
    labels = [label for label in labels if by_strategy[label]]

    if not data:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=True)
    ax.set_xlabel("Strategy")
    ax.set_ylabel("Final fitness")
    ax.set_title("Final fitness distribution by strategy")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Analyze PSO results from summary/history/metadata files.")
    parser.add_argument("--input-dir", required=True, help="Directory with one run or multiple run folders")
    parser.add_argument("--output-dir", default=None, help="Directory to save analysis outputs")
    parser.add_argument("--no-plots", action="store_true", help="Disable plot generation")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_run_dirs(input_dir)
    if not run_dirs:
        raise FileNotFoundError("No summary.csv found in input directory or its direct subfolders")

    summary_rows = []
    history_rows = []

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.csv"
        history_path = run_dir / "history.csv"
        metadata_path = run_dir / "metadata.json"

        run_summary = _read_csv_rows(summary_path)
        metadata = _read_metadata(metadata_path)
        strategy = _detect_strategy(run_dir, metadata, run_summary)

        for row in run_summary:
            row = dict(row)
            row["strategy"] = row.get("strategy") or strategy
            summary_rows.append(row)

        if history_path.exists():
            run_history = _read_csv_rows(history_path)
            for row in run_history:
                row = dict(row)
                row["strategy"] = strategy
                history_rows.append(row)
        else:
            logging.info("history.csv not found in %s (continuing without convergence for this run)", run_dir)

    if not summary_rows:
        raise RuntimeError("No summary rows were loaded")

    table = _compute_strategy_summary(summary_rows)
    summary_out = output_dir / "analysis_summary.csv"
    _write_summary_csv(table, summary_out)

    logging.info("Loaded %d run folder(s)", len(run_dirs))
    logging.info("Loaded %d summary row(s)", len(summary_rows))
    logging.info("Loaded %d history row(s)", len(history_rows))
    logging.info("Analysis summary saved in: %s", summary_out)

    # Print a simple console table.
    logging.info("Strategy summary:")
    for row in table:
        logging.info(
            "  %s | n=%d | best=%.6e | elapsed=%.6f | eval=%.6f | update=%.6f | speedup=%.3f | eff=%.3f | overhead=%.6f",
            row["strategy"],
            row["n_runs"],
            row["mean_best_fitness"],
            row["mean_elapsed_seconds"],
            row["mean_fitness_eval_seconds"],
            row["mean_state_update_seconds"],
            row["speedup_vs_v0"] if np.isfinite(row["speedup_vs_v0"]) else float("nan"),
            row["efficiency"] if np.isfinite(row["efficiency"]) else float("nan"),
            row["mean_overhead_seconds"] if np.isfinite(row["mean_overhead_seconds"]) else float("nan"),
        )

    if not args.no_plots:
        times_plot = output_dir / "analysis_times_by_strategy.png"
        _plot_times(table, times_plot)
        logging.info("Time comparison plot saved in: %s", times_plot)

        fitness_col = _detect_final_fitness_column(summary_rows)
        if fitness_col is None:
            logging.warning(
                "Final fitness column not found in summary rows. Expected one of: best_fitness, best_f, final_fitness, fitness"
            )
        else:
            fitness_plot = output_dir / "analysis_fitness_boxplot_by_strategy.png"
            fitness_plotted = _plot_fitness_boxplot_by_strategy(summary_rows, fitness_col, fitness_plot)
            if fitness_plotted:
                logging.info("Final fitness boxplot saved in: %s", fitness_plot)
            else:
                logging.warning("Final fitness boxplot was skipped (no usable fitness data)")

        conv_plot = output_dir / "analysis_convergence_by_strategy.png"
        plotted = _plot_convergence(history_rows, conv_plot)
        if plotted:
            logging.info("Convergence plot saved in: %s", conv_plot)
        else:
            logging.info("Convergence plot was skipped (no usable history data)")


if __name__ == "__main__":
    main()
