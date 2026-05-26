import argparse
import logging

import numpy as np

from core import PSO, PSOVectorized, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
from objectives import BENCHMARKS, make_async_objective
from parallel.async_evaluator import AsyncEvaluator
from parallel.process_evaluator import ProcessPoolEvaluator
from parallel.thread_evaluator import ThreadPoolEvaluator


def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run one PSO execution (V0/V1/V2/V3/V4).")
    parser.add_argument("--strategy", choices=["v0", "v1", "v2", "v3", "v4"], default="v0", help="Execution strategy")
    parser.add_argument("--workers", type=int, default=0, help="Workers for v1/v2/v3")
    parser.add_argument("--batch-size", type=int, default=0, help="Batch size for v2")
    parser.add_argument("--async-latency", type=float, default=0.01, help="Artificial latency for v3 objective")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument("--objective", choices=list(BENCHMARKS.keys()), default="sphere")
    args = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Select the objective function and create bounds for 10 dimensions.
    spec = BENCHMARKS[args.objective]
    d = 10
    bounds = BoxBounds(low=spec["low"] * np.ones(d), high=spec["high"] * np.ones(d))

    if args.strategy in {"v1", "v2", "v3"} and args.workers <= 0:
        raise ValueError("workers must be > 0 when strategy is v1, v2, or v3")
    if args.strategy == "v2" and args.batch_size <= 0:
        raise ValueError("batch-size must be > 0 when strategy=v2")

    # Define PSO parameters for a single execution.
    params = PSOParams(
        n_particles=40,
        n_iters=300,
        seed=7,
        stop_tol=1e-10,
        stagnation_iters=50,
        log_every_iters=args.log_every,
    )

    if args.strategy == "v1":
        evaluator = ThreadPoolEvaluator(spec["func"], workers=args.workers)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif args.strategy == "v2":
        evaluator = ProcessPoolEvaluator(spec["func"], workers=args.workers, batch_size=args.batch_size)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif args.strategy == "v3":
        async_objective = make_async_objective(spec["func"], latency=args.async_latency)
        evaluator = AsyncEvaluator(async_objective, workers=args.workers)
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    elif args.strategy == "v4":
        evaluator = None
        pso = PSOVectorized(bounds=bounds, objective=spec["func"], params=params, bounds_policy=ClampPolicy())
    else:
        evaluator = SequentialEvaluator(spec["func"])
        pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())

    try:
        best_x, best_f, history = pso.run()
        timing = pso.last_run_stats or {}
    finally:
        close_fn = getattr(evaluator, "close", None)
        if callable(close_fn):
            close_fn()

    # Log the final results.
    logging.info("Strategy: %s", args.strategy)
    logging.info("Objective: %s", args.objective)
    logging.info("Dim: %s", d)
    logging.info("Best fitness: %s", best_f)
    logging.info("Best x (first 5 dims): %s", best_x[:5])
    logging.info("First best: %s", history[0])
    logging.info("Last best: %s", history[-1])
    logging.info("Iterations run: %s", len(history) - 1)
    logging.info("Total run seconds: %s", timing.get("total_run_seconds"))
    logging.info("Fitness eval seconds: %s", timing.get("fitness_eval_seconds"))
    logging.info("State update seconds: %s", timing.get("state_update_seconds"))


if __name__ == "__main__":
    main()
