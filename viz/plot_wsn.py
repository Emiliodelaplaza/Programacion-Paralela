import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import PSO, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
from objectives import create_wsn_objective, make_wsn_bounds


def _coverage_field(width, height, grid_size, alpha, sensors):
    xs = np.linspace(0.0, width, grid_size, dtype=float)
    ys = np.linspace(0.0, height, grid_size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    diff = points[:, None, :] - sensors[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2, dtype=float)
    p = np.exp(-alpha * dist_sq)
    coverage_point = 1.0 - np.prod(1.0 - p, axis=1)
    coverage_map = coverage_point.reshape(xx.shape)
    return xx, yy, coverage_map


def main():
    parser = argparse.ArgumentParser(description="Run a small PSO for WSN and plot final 2D coverage.")
    parser.add_argument("--num-sensors", type=int, default=5)
    parser.add_argument("--width", type=float, default=100.0)
    parser.add_argument("--height", type=float, default=60.0)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=80)
    parser.add_argument("--output", default="results/viz/wsn_coverage.png")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    low, high = make_wsn_bounds(args.num_sensors, args.width, args.height)
    bounds = BoxBounds(low=low, high=high)

    objective = create_wsn_objective(
        num_sensors=args.num_sensors,
        width=args.width,
        height=args.height,
        grid_size=args.grid_size,
        alpha=args.alpha,
        seed=args.seed,
        vectorized=True,
    )

    evaluator = SequentialEvaluator(objective)
    params = PSOParams(
        n_particles=args.n_particles,
        n_iters=args.n_iters,
        w=0.72,
        c1=1.49,
        c2=1.49,
        vmax_frac=0.2,
        seed=args.seed,
        stop_tol=1e-10,
        stagnation_iters=50,
        log_every_iters=None,
    )

    pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    best_x, best_f, _ = pso.run()

    sensors = best_x.reshape(args.num_sensors, 2)
    xx, yy, coverage_map = _coverage_field(args.width, args.height, args.grid_size, args.alpha, sensors)
    coverage_mean = float(np.mean(coverage_map))

    fig, ax = plt.subplots(figsize=(8, 5))
    contour = ax.contourf(xx, yy, coverage_map, levels=30, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(contour, ax=ax, label="Coverage probability")

    ax.scatter(sensors[:, 0], sensors[:, 1], c="red", s=70, edgecolors="black", linewidths=0.7, label="Sensors")
    ax.set_xlim(0.0, args.width)
    ax.set_ylim(0.0, args.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"WSN coverage | fitness={best_f:.6f} | mean coverage~{coverage_mean:.6f}")
    ax.legend(loc="upper right")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    logging.info("WSN visualization saved in: %s", out_path)


if __name__ == "__main__":
    main()
