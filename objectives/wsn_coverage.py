import numpy as np


def make_wsn_bounds(num_sensors, width, height):
    """Return axis-aligned bounds for [x1, y1, ..., xM, yM]."""
    num_sensors = int(num_sensors)
    width = float(width)
    height = float(height)

    if num_sensors <= 0:
        raise ValueError("num_sensors must be > 0")
    if width <= 0.0:
        raise ValueError("width must be > 0")
    if height <= 0.0:
        raise ValueError("height must be > 0")

    low = np.empty(2 * num_sensors, dtype=float)
    high = np.empty(2 * num_sensors, dtype=float)
    low[0::2] = 0.0
    low[1::2] = 0.0
    high[0::2] = width
    high[1::2] = height
    return low, high


def _build_grid_points(width, height, grid_size, seed):
    n_points = int(grid_size) ** 2
    if seed is None:
        xs = np.linspace(0.0, width, int(grid_size), dtype=float)
        ys = np.linspace(0.0, height, int(grid_size), dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    rng = np.random.default_rng(int(seed))
    x_rand = rng.uniform(0.0, width, size=n_points)
    y_rand = rng.uniform(0.0, height, size=n_points)
    return np.column_stack([x_rand, y_rand])


def create_wsn_objective(
    num_sensors=None,
    width=100.0,
    height=60.0,
    grid_size=20,
    alpha=0.01,
    seed=None,
    vectorized=True,
):
    """Create a WSN area-coverage objective function for PSO minimization."""
    width = float(width)
    height = float(height)
    grid_size = int(grid_size)
    alpha = float(alpha)

    if num_sensors is not None and int(num_sensors) <= 0:
        raise ValueError("num_sensors must be > 0 when provided")
    if width <= 0.0:
        raise ValueError("width must be > 0")
    if height <= 0.0:
        raise ValueError("height must be > 0")
    if grid_size <= 1:
        raise ValueError("grid_size must be > 1")
    if alpha <= 0.0:
        raise ValueError("alpha must be > 0")

    grid_points = _build_grid_points(width=width, height=height, grid_size=grid_size, seed=seed)

    def wsn_coverage_objective(position):
        x = np.asarray(position, dtype=float)
        if x.ndim != 1:
            raise ValueError("position must be a 1D array")

        if num_sensors is None:
            if x.size % 2 != 0:
                raise ValueError("position length must be even: [x1, y1, ..., xM, yM]")
            sensors_count = x.size // 2
        else:
            sensors_count = int(num_sensors)
            expected = 2 * sensors_count
            if x.size != expected:
                raise ValueError(f"position length must be {expected} for num_sensors={sensors_count}")

        coords = x.reshape(sensors_count, 2)

        # If bounds are bypassed, penalize invalid sensor coordinates.
        violation_x = np.maximum(0.0, -coords[:, 0]) + np.maximum(0.0, coords[:, 0] - width)
        violation_y = np.maximum(0.0, -coords[:, 1]) + np.maximum(0.0, coords[:, 1] - height)
        violation = float(np.sum(violation_x + violation_y))
        if violation > 0.0:
            return float(1.0 + violation)

        if vectorized:
            diff = grid_points[:, None, :] - coords[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=2, dtype=float)
            p = np.exp(-alpha * dist_sq)
            coverage_point = 1.0 - np.prod(1.0 - p, axis=1)
            coverage_mean = float(np.mean(coverage_point))
            return float(1.0 - coverage_mean)

        coverage_values = []
        for point in grid_points:
            p_values = []
            for sensor in coords:
                dx = point[0] - sensor[0]
                dy = point[1] - sensor[1]
                p_values.append(np.exp(-alpha * (dx * dx + dy * dy)))
            p_values = np.asarray(p_values, dtype=float)
            coverage_values.append(1.0 - np.prod(1.0 - p_values))
        coverage_mean = float(np.mean(coverage_values))
        return float(1.0 - coverage_mean)

    return wsn_coverage_objective
