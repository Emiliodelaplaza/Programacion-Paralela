import argparse
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import PSO, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
from objectives import BENCHMARKS


def _make_2d_contour(objective, low, high, resolution=120):
    # Build a grid to draw the objective function in 2D.
    x0 = np.linspace(low[0], high[0], resolution)
    x1 = np.linspace(low[1], high[1], resolution)
    xx, yy = np.meshgrid(x0, x1)
    flat = np.column_stack([xx.ravel(), yy.ravel()])
    zz = np.array([objective(p) for p in flat], dtype=float).reshape(xx.shape)
    return xx, yy, zz


def _plot_convergence(history, out_path):
    # Save the best-fitness curve over the iterations.
    fig, ax = plt.subplots(figsize=(7, 4))
    iters = np.arange(len(history))
    ax.plot(iters, history, color="tab:blue", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness")
    ax.set_title("Convergence (best fitness vs iteration)")
    ax.grid(True, alpha=0.3)

    # Use log scale only when all values are positive.
    if np.all(np.array(history) > 0):
        ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _make_frames_2d(frames_dir, objective_name, positions_history, gbest_history, position_iters, history, low, high):
    # Generate one frame per stored iteration in 2D.
    objective = BENCHMARKS[objective_name]["func"]
    xx, yy, zz = _make_2d_contour(objective, low, high)

    for idx, (positions, gbest, iter_idx) in enumerate(zip(positions_history, gbest_history, position_iters)):
        fig, ax = plt.subplots(figsize=(7, 5))
        contour = ax.contourf(xx, yy, zz, levels=28, cmap="viridis")
        fig.colorbar(contour, ax=ax, shrink=0.85, label="f(x)")

        # Current swarm positions.
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=30,
            c="white",
            edgecolors="black",
            linewidths=0.6,
            label="Swarm",
        )

        # Current global best position.
        ax.scatter(gbest[0], gbest[1], s=120, c="red", marker="*", label="Global best")

        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_title(f"{objective_name} d=2 | iter={iter_idx} | best={history[iter_idx]:.3e}")
        ax.legend(loc="upper right")

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)


def _make_frames_3d(frames_dir, objective_name, positions_history, gbest_history, position_iters, history, low, high):
    # Generate one frame per stored iteration in 3D.
    for idx, (positions, gbest, iter_idx) in enumerate(zip(positions_history, gbest_history, position_iters)):
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20, c="tab:blue", alpha=0.9, label="Swarm")
        ax.scatter(gbest[0], gbest[1], gbest[2], s=120, c="red", marker="*", label="Global best")

        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")
        ax.set_title(f"{objective_name} d=3 | iter={iter_idx} | best={history[iter_idx]:.3e}")
        ax.legend(loc="upper right")

        frame_path = frames_dir / f"frame_{idx:04d}.png"
        fig.tight_layout()
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Generate simple PSO visualization frames for V0.")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=2, help="Dimension: 2 or 3")
    parser.add_argument("--objective", choices=list(BENCHMARKS.keys()), default="sphere")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-iters", type=int, default=60)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--track-every", type=int, default=1)
    parser.add_argument("--output-dir", default="results", help="Directory for output files")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Create the search space from the selected benchmark.
    spec = BENCHMARKS[args.objective]
    low = spec["low"] * np.ones(args.dim)
    high = spec["high"] * np.ones(args.dim)

    bounds = BoxBounds(low=low, high=high)
    evaluator = SequentialEvaluator(spec["func"])
    params = PSOParams(
        n_particles=args.n_particles,
        n_iters=args.n_iters,
        w=0.72,
        c1=1.49,
        c2=1.49,
        vmax_frac=0.2,
        seed=args.seed,
        stop_tol=0.0,
        stagnation_iters=None,
        target_fitness=None,
        log_every_iters=None,
    )

    # Run PSO while storing positions for the visualization.
    pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    _, _, history = pso.run(track_positions=True, track_every=args.track_every)

    positions_history = pso.last_positions_history or []
    gbest_history = pso.last_gbest_history or []
    position_iters = pso.last_positions_iters or []

    if not positions_history or not gbest_history or not position_iters:
        raise RuntimeError("Tracking data is empty. Run with track_positions enabled.")

    # Create the output folder for this visualization run.
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"viz_v0_{run_id}"
    dim_dir = run_dir / f"d{args.dim}"
    dim_dir.mkdir(parents=True, exist_ok=True)

    # Generate 2D or 3D frames depending on the chosen dimension.
    if args.dim == 2:
        _make_frames_2d(
            frames_dir=dim_dir,
            objective_name=args.objective,
            positions_history=positions_history,
            gbest_history=gbest_history,
            position_iters=position_iters,
            history=history,
            low=low,
            high=high,
        )
    else:
        _make_frames_3d(
            frames_dir=dim_dir,
            objective_name=args.objective,
            positions_history=positions_history,
            gbest_history=gbest_history,
            position_iters=position_iters,
            history=history,
            low=low,
            high=high,
        )

    # Save also the convergence curve.
    _plot_convergence(history, dim_dir / "convergence.png")
    logging.info("Visualization saved in: %s", dim_dir)


if __name__ == "__main__":
    main()
