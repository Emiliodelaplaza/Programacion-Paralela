import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from core import PSO, PSOVectorized, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
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


def _run_one(objective_name, objective_spec, dim, seed, pso_params, strategy, workers, batch_size, async_latency):
    """Run one benchmark configuration and return its results."""
    objective = objective_spec["func"]
    low = objective_spec["low"] * np.ones(dim)
    high = objective_spec["high"] * np.ones(dim)

    bounds = BoxBounds(low=low, high=high)
    params = PSOParams(
        n_particles=pso_params["n_particles"],
        n_iters=pso_params["n_iters"],
        w=pso_params["w"],
        c1=pso_params["c1"],
        c2=pso_params["c2"],
        vmax_frac=pso_params["vmax_frac"],
        seed=seed,
        stop_tol=pso_params["stop_tol"],
        stagnation_iters=pso_params["stagnation_iters"],
        target_fitness=pso_params["target_fitness"],
        log_every_iters=pso_params["log_every_iters"],
    )

    # Select evaluator depending on the execution strategy.
    if strategy == "v1":
        evaluator = ThreadPoolEvaluator(objective, workers=workers)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif strategy == "v2":
        evaluator = ProcessPoolEvaluator(objective, workers=workers, batch_size=batch_size)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif strategy == "v3":
        async_objective = make_async_objective(objective, latency=async_latency)
        evaluator = AsyncEvaluator(async_objective, workers=workers)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif strategy == "v4":
        evaluator = None
        pso = PSOVectorized(bounds=bounds, objective=objective, params=params, bounds_policy=ClampPolicy())
    else:
        evaluator = SequentialEvaluator(objective)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())

    try:
        best_x, best_f, history = pso.run()
        timing = pso.last_run_stats or {}
        elapsed = float(timing.get("total_run_seconds", 0.0))
        fitness_eval_seconds = float(timing.get("fitness_eval_seconds", 0.0))
        state_update_seconds = float(timing.get("state_update_seconds", 0.0))
    finally:
        # Close worker pools if the evaluator uses them.
        close_fn = getattr(evaluator, "close", None)
        if callable(close_fn):
            close_fn()

    return {
        "objective": objective_name,
        "dim": dim,
        "seed": seed,
        "strategy": strategy,
        "workers": workers,
        "batch_size": batch_size,
        "best_fitness": float(best_f),
        "first_fitness": float(history[0]),
        "iterations_run": len(history) - 1,
        "elapsed_seconds": elapsed,
        "fitness_eval_seconds": fitness_eval_seconds,
        "state_update_seconds": state_update_seconds,
        "best_x": best_x.tolist(),
        "history": [float(v) for v in history],
    }


def _write_csv(path, rows, headers):
    """Write rows into a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def main():
    """Run the full benchmark suite and save the results."""
    parser = argparse.ArgumentParser(description="Run PSO benchmark suite (V0 sequential / V1 threads / V2 processes / V3 asyncio / V4 vectorized).")
    parser.add_argument("--dims", default="2,10,30", help="Comma-separated dimensions")
    parser.add_argument("--seeds", default="7,19,31,43,59", help="Comma-separated seeds")
    parser.add_argument("--strategy", choices=["v0", "v1", "v2", "v3", "v4"], default="v0", help="Execution strategy")
    parser.add_argument("--workers", type=int, default=0, help="Workers for v1/v2/v3")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for v2")
    parser.add_argument("--async-latency", type=float, default=0.01, help="Artificial latency for v3 objective")
    parser.add_argument("--n-particles", type=int, default=40)
    parser.add_argument("--n-iters", type=int, default=300)
    parser.add_argument("--w", type=float, default=0.72)
    parser.add_argument("--c1", type=float, default=1.49)
    parser.add_argument("--c2", type=float, default=1.49)
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

    if not dims:
        raise ValueError("dims cannot be empty")
    if not seeds:
        raise ValueError("seeds cannot be empty")
    if args.strategy in {"v1", "v2", "v3"} and args.workers <= 0:
        raise ValueError("workers must be > 0 when strategy is v1, v2, or v3")
    if args.strategy == "v2" and args.batch_size <= 0:
        raise ValueError("batch-size must be > 0 when strategy=v2")
    if args.strategy == "v3" and args.async_latency < 0.0:
        raise ValueError("async-latency must be >= 0 when strategy=v3")

    effective_workers = args.workers if args.strategy in {"v1", "v2", "v3"} else 0
    effective_batch_size = args.batch_size if args.strategy == "v2" else 0

    pso_params = {
        "n_particles": args.n_particles,
        "n_iters": args.n_iters,
        "w": args.w,
        "c1": args.c1,
        "c2": args.c2,
        "vmax_frac": args.vmax_frac,
        "stop_tol": args.stop_tol,
        "stagnation_iters": args.stagnation_iters,
        "target_fitness": args.target_fitness,
        "log_every_iters": args.log_every,
    }

    # Create an output folder for this run.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"benchmarks_{args.strategy}_{run_id}"
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
        "pso_params": pso_params,
        "timing_metrics": [
            "elapsed_seconds",
            "fitness_eval_seconds",
            "state_update_seconds",
        ],
        "logging": {
            "log_every": args.log_every,
            "log_level": args.log_level.upper(),
        },
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    summary_rows = []
    history_rows = []

    total_runs = len(BENCHMARKS) * len(dims) * len(seeds)
    run_counter = 0

    # Run all combinations of benchmark, dimension, and seed.
    for objective_name, objective_spec in BENCHMARKS.items():
        for dim in dims:
            for seed in seeds:
                run_counter += 1
                result = _run_one(
                    objective_name,
                    objective_spec,
                    dim,
                    seed,
                    pso_params,
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
                        "best_fitness": result["best_fitness"],
                        "first_fitness": result["first_fitness"],
                        "iterations_run": result["iterations_run"],
                        "elapsed_seconds": result["elapsed_seconds"],
                        "fitness_eval_seconds": result["fitness_eval_seconds"],
                        "state_update_seconds": result["state_update_seconds"],
                    }
                )

                # Save convergence history for each run.
                for iteration, best_val in enumerate(result["history"]):
                    history_rows.append(
                        {
                            "objective": result["objective"],
                            "dim": result["dim"],
                            "seed": result["seed"],
                            "iteration": iteration,
                            "best_fitness": best_val,
                        }
                    )

                logging.info(
                    f"[{run_counter}/{total_runs}] "
                    f"{result['strategy']} w={result['workers']} bs={result['batch_size']} "
                    f"{result['objective']} d={result['dim']} seed={result['seed']} "
                    f"best={result['best_fitness']:.6e} "
                    f"time={result['elapsed_seconds']:.4f}s "
                    f"eval={result['fitness_eval_seconds']:.4f}s "
                    f"update={result['state_update_seconds']:.4f}s"
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
            "best_fitness",
            "first_fitness",
            "iterations_run",
            "elapsed_seconds",
            "fitness_eval_seconds",
            "state_update_seconds",
        ],
    )

    _write_csv(
        output_dir / "history.csv",
        history_rows,
        headers=["objective", "dim", "seed", "iteration", "best_fitness"],
    )

    logging.info("Results saved in: %s", output_dir)


if __name__ == "__main__":
    main()
