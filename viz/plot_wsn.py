import argparse
import logging
from pathlib import Path

import matplotlib

# Use a non-interactive backend so the plot can be generated from the terminal
# without opening a graphical window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import PSO, PSOParams, BoxBounds, ClampPolicy, SequentialEvaluator
from objectives import create_wsn_objective, make_wsn_bounds


def _coverage_field(width, height, grid_size, alpha, sensors):
    # Build a regular 2D grid over the rectangular area to evaluate coverage.
    xs = np.linspace(0.0, width, grid_size, dtype=float)
    ys = np.linspace(0.0, height, grid_size, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute the squared distance from every grid point to every sensor.
    diff = points[:, None, :] - sensors[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2, dtype=float)

    # Detection probability decreases with distance.
    p = np.exp(-alpha * dist_sq)

    # Probability that each point is covered by at least one sensor.
    coverage_point = 1.0 - np.prod(1.0 - p, axis=1)

    # Reshape the flat coverage vector back into a 2D map for plotting.
    coverage_map = coverage_point.reshape(xx.shape)
    return xx, yy, coverage_map


def main():
    parser = argparse.ArgumentParser(description="Run a small PSO for WSN and plot final 2D coverage.")

    # WSN problem configuration.
    parser.add_argument("--num-sensors", type=int, default=5)
    parser.add_argument("--width", type=float, default=100.0)
    parser.add_argument("--height", type=float, default=60.0)
    parser.add_argument("--grid-size", type=int, default=30)
    parser.add_argument("--alpha", type=float, default=0.01)

    # PSO configuration.
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-particles", type=int, default=30)
    parser.add_argument("--n-iters", type=int, default=80)

    # Output and logging options.
    parser.add_argument("--output", default="results/viz/wsn_coverage.png")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # Create bounds for the sensor coordinates:
    # x coordinates are limited by [0, width] and y coordinates by [0, height].
    low, high = make_wsn_bounds(args.num_sensors, args.width, args.height)
    bounds = BoxBounds(low=low, high=high)

    # Create the WSN objective that PSO will minimize.
    # Since fitness = 1 - mean_coverage, lower values mean better coverage.
    objective = create_wsn_objective(
        num_sensors=args.num_sensors,
        width=args.width,
        height=args.height,
        grid_size=args.grid_size,
        alpha=args.alpha,
        seed=args.seed,
        vectorized=True,
    )

    # Use sequential evaluation for the visualization script.
    # The goal here is to generate a clear plot, not to benchmark strategies.
    evaluator = SequentialEvaluator(objective)

    # PSO parameters used for this small visualization run.
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

    # Run PSO and get the best sensor layout found.
    pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    best_x, best_f, _ = pso.run()

    # Convert the flat vector [x1, y1, ..., xM, yM] into sensor coordinates.
    sensors = best_x.reshape(args.num_sensors, 2)

    # Recompute the final coverage field so it can be visualized as a 2D map.
    xx, yy, coverage_map = _coverage_field(args.width, args.height, args.grid_size, args.alpha, sensors)
    coverage_mean = float(np.mean(coverage_map))

    # Create the coverage plot.
    fig, ax = plt.subplots(figsize=(8, 5))

    # Draw the coverage probability as a filled contour map.
    contour = ax.contourf(xx, yy, coverage_map, levels=30, cmap="viridis", vmin=0.0, vmax=1.0)
    fig.colorbar(contour, ax=ax, label="Coverage probability")

    # Draw final sensor positions on top of the coverage map.
    ax.scatter(sensors[:, 0], sensors[:, 1], c="red", s=70, edgecolors="black", linewidths=0.7, label="Sensors")

    # Configure plot axes and labels.
    ax.set_xlim(0.0, args.width)
    ax.set_ylim(0.0, args.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"WSN coverage | fitness={best_f:.6f} | mean coverage~{coverage_mean:.6f}")
    ax.legend(loc="upper right")

    # Save the plot to disk, creating the output directory if needed.
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    logging.info("WSN visualization saved in: %s", out_path)


if __name__ == "__main__":
    main()