import argparse
import csv
import itertools
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from core import PSO, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
from objectives import BENCHMARKS, make_async_objective
from parallel.async_evaluator import AsyncEvaluator
from parallel.process_evaluator import ProcessPoolEvaluator
from parallel.thread_evaluator import ThreadPoolEvaluator


def _parse_int_list(text):
    """Convert a comma-separated string into a list of integers."""
    text = text.strip()
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",")]


def _parse_float_list(text):
    """Convert a comma-separated string into a list of floats."""
    text = text.strip()
    if not text:
        return []
    return [float(part.strip()) for part in text.split(",")]


def _write_csv(path, rows, headers):
    """Write rows into a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _run_one(objective_name, objective_spec, dim, seed, cfg, strategy, workers, batch_size, async_latency):
    """Run one grid-search configuration and return its result."""
    objective = objective_spec["func"]
    low = objective_spec["low"] * np.ones(dim)
    high = objective_spec["high"] * np.ones(dim)

    bounds = BoxBounds(low=low, high=high)

    # Select evaluator depending on the execution strategy.
    if strategy == "v1":
        evaluator = ThreadPoolEvaluator(objective, workers=workers)
    elif strategy == "v2":
        evaluator = ProcessPoolEvaluator(objective, workers=workers, batch_size=batch_size)
    elif strategy == "v3":
        async_objective = make_async_objective(objective, latency=async_latency)
        evaluator = AsyncEvaluator(async_objective, workers=workers)
    else:
        evaluator = SequentialEvaluator(objective)

    params = PSOParams(
        n_particles=cfg["n_particles"],
        n_iters=cfg["n_iters"],
        w=cfg["w"],
        c1=cfg["c1"],
        c2=cfg["c2"],
        vmax_frac=cfg["vmax_frac"],
        seed=seed,
        stop_tol=cfg["stop_tol"],
        stagnation_iters=cfg["stagnation_iters"],
        target_fitness=cfg["target_fitness"],
        log_every_iters=cfg["log_every_iters"],
    )

    pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())

    try:
        best_x, best_f, history = pso.run()
        timing = pso.last_run_stats or {}
    finally:
        # Close worker pools if needed.
        close_fn = getattr(evaluator, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "strategy": strategy,
        "workers": workers,
        "batch_size": batch_size,
        "objective": objective_name,
        "dim": dim,
        "seed": seed,
        "w": cfg["w"],
        "c1": cfg["c1"],
        "c2": cfg["c2"],
        "n_particles": cfg["n_particles"],
        "n_iters": cfg["n_iters"],
        "best_fitness": float(best_f),
        "iterations_run": len(history) - 1,
        "elapsed_seconds": float(timing.get("total_run_seconds", 0.0)),
        "fitness_eval_seconds": float(timing.get("fitness_eval_seconds", 0.0)),
        "state_update_seconds": float(timing.get("state_update_seconds", 0.0)),
        "best_x": best_x.tolist(),
        "history": [float(v) for v in history],
    }
def main():
    """Run the PSO grid search and save the results."""
    parser = argparse.ArgumentParser(description="Run PSO grid search (V0 sequential / V1 threads / V2 processes / V3 asyncio).")
    parser.add_argument("--dims", default="2,10,30", help="Comma-separated dimensions")
    parser.add_argument("--seeds", default="7,19", help="Comma-separated seeds")
    parser.add_argument("--strategy", choices=["v0", "v1", "v2", "v3"], default="v0", help="Execution strategy")
    parser.add_argument("--workers", type=int, default=0, help="Workers for v1/v2/v3")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for v2")
    parser.add_argument("--async-latency", type=float, default=0.01, help="Artificial latency for v3 objective")
    parser.add_argument("--w-grid", default="0.72", help="Comma-separated w values")
    parser.add_argument("--c1-grid", default="1.49", help="Comma-separated c1 values")
    parser.add_argument("--c2-grid", default="1.49", help="Comma-separated c2 values")
    parser.add_argument("--n-particles-grid", default="40", help="Comma-separated n_particles values")
    parser.add_argument("--n-iters-grid", default="100", help="Comma-separated n_iters values")
    parser.add_argument("--vmax-frac", type=float, default=0.2)
    parser.add_argument("--stop-tol", type=float, default=1e-10)
    parser.add_argument("--stagnation-iters", type=int, default=50)
    parser.add_argument("--target-fitness", type=float, default=None)
    parser.add_argument("--log-every", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    dims = _parse_int_list(args.dims)
    seeds = _parse_int_list(args.seeds)
    w_grid = _parse_float_list(args.w_grid)
    c1_grid = _parse_float_list(args.c1_grid)
    c2_grid = _parse_float_list(args.c2_grid)
    n_particles_grid = _parse_int_list(args.n_particles_grid)
    n_iters_grid = _parse_int_list(args.n_iters_grid)

    if not dims:
        raise ValueError("dims cannot be empty")
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if not w_grid:
        raise ValueError("w-grid cannot be empty")
    if not c1_grid:
        raise ValueError("c1-grid cannot be empty")
    if not c2_grid:
        raise ValueError("c2-grid cannot be empty")
    if not n_particles_grid:
        raise ValueError("n-particles-grid cannot be empty")
    if not n_iters_grid:
        raise ValueError("n-iters-grid cannot be empty")
    if args.strategy in {"v1", "v2", "v3"} and args.workers <= 0:
        raise ValueError("workers must be > 0 when strategy is v1, v2, or v3")
    if args.strategy == "v2" and args.batch_size <= 0:
        raise ValueError("batch-size must be > 0 when strategy=v2")
    if args.strategy == "v3" and args.async_latency < 0.0:
        raise ValueError("async-latency must be >= 0 when strategy=v3")

    effective_workers = args.workers if args.strategy in {"v1", "v2", "v3"} else 0
    effective_batch_size = args.batch_size if args.strategy == "v2" else 0

    # Build all hyperparameter combinations to test.
    hyper_grid = list(itertools.product(w_grid, c1_grid, c2_grid, n_particles_grid, n_iters_grid))
    total_runs = len(BENCHMARKS) * len(dims) * len(seeds) * len(hyper_grid)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"grid_{args.strategy}_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "strategy": args.strategy,
        "workers": effective_workers,
        "batch_size": effective_batch_size,
        "async_latency": args.async_latency,
        "timestamp": run_id,
        "benchmarks": list(BENCHMARKS.keys()),
        "dims": dims,
        "seeds": seeds,
        "grid": {
            "w": w_grid,
            "c1": c1_grid,
            "c2": c2_grid,
            "n_particles": n_particles_grid,
            "n_iters": n_iters_grid,
        },
        "base_params": {
            "vmax_frac": args.vmax_frac,
            "stop_tol": args.stop_tol,
            "stagnation_iters": args.stagnation_iters,
            "target_fitness": args.target_fitness,
            "log_every_iters": args.log_every,
        },
        "total_runs": total_runs,
        "timing_metrics": [
            "elapsed_seconds",
            "fitness_eval_seconds",
            "state_update_seconds",
        ],
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_rows = []
    run_counter = 0

    # Run every benchmark with every dimension, seed, and hyperparameter combination.
    for objective_name, objective_spec in BENCHMARKS.items():
        for dim in dims:
            for seed in seeds:
                for w, c1, c2, n_particles, n_iters in hyper_grid:
                    run_counter += 1
                    cfg = {
                        "w": w,
                        "c1": c1,
                        "c2": c2,
                        "n_particles": n_particles,
                        "n_iters": n_iters,
                        "vmax_frac": args.vmax_frac,
                        "stop_tol": args.stop_tol,
                        "stagnation_iters": args.stagnation_iters,
                        "target_fitness": args.target_fitness,
                        "log_every_iters": args.log_every,
                    }

                    result = _run_one(
                        objective_name,
                        objective_spec,
                        dim,
                        seed,
                        cfg,
                        strategy=args.strategy,
                        workers=effective_workers,
                        batch_size=effective_batch_size,
                        async_latency=args.async_latency,
                    )

                    summary_rows.append(
                        {
                            "strategy": result["strategy"],
                            "workers": result["workers"],
                            "batch_size": result["batch_size"],
                            "objective": result["objective"],
                            "dim": result["dim"],
                            "seed": result["seed"],
                            "w": result["w"],
                            "c1": result["c1"],
                            "c2": result["c2"],
                            "n_particles": result["n_particles"],
                            "n_iters": result["n_iters"],
                            "best_fitness": result["best_fitness"],
                            "iterations_run": result["iterations_run"],
                            "elapsed_seconds": result["elapsed_seconds"],
                            "fitness_eval_seconds": result["fitness_eval_seconds"],
                            "state_update_seconds": result["state_update_seconds"],
                        }
                    )

                    logging.info(
                        f"[{run_counter}/{total_runs}] "
                        f"{result['strategy']} w={result['workers']} bs={result['batch_size']} "
                        f"{result['objective']} d={result['dim']} seed={result['seed']} "
                        f"w={result['w']} c1={result['c1']} c2={result['c2']} "
                        f"np={result['n_particles']} iters={result['n_iters']} "
                        f"best={result['best_fitness']:.6e} "
                        f"time={result['elapsed_seconds']:.4f}s"
                    )

    _write_csv(
        output_dir / "summary.csv",
        summary_rows,
        headers=[
            "strategy",
            "workers",
            "batch_size",
            "objective",
            "dim",
            "seed",
            "w",
            "c1",
            "c2",
            "n_particles",
            "n_iters",
            "best_fitness",
            "iterations_run",
            "elapsed_seconds",
            "fitness_eval_seconds",
            "state_update_seconds",
        ],
    )

    logging.info("Results saved in: %s", output_dir)


if __name__ == "__main__":
    main()
