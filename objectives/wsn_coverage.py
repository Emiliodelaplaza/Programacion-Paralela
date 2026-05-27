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

    # The vector alternates x and y coordinates:
    # x coordinates are bounded by [0, width], y coordinates by [0, height].
    low = np.empty(2 * num_sensors, dtype=float)
    high = np.empty(2 * num_sensors, dtype=float)
    low[0::2] = 0.0
    low[1::2] = 0.0
    high[0::2] = width
    high[1::2] = height
    return low, high


def _build_grid_points(width, height, grid_size, seed):
    # The grid represents the area where sensor coverage is evaluated.
    n_points = int(grid_size) ** 2

    if seed is None:
        # Deterministic regular grid over the rectangular region.
        xs = np.linspace(0.0, width, int(grid_size), dtype=float)
        ys = np.linspace(0.0, height, int(grid_size), dtype=float)
        xx, yy = np.meshgrid(xs, ys)
        return np.column_stack([xx.ravel(), yy.ravel()])

    # Optional seeded random sampling of points, useful for reproducible stochastic grids.
    rng = np.random.default_rng(int(seed))
    x_rand = rng.uniform(0.0, width, size=n_points)
    y_rand = rng.uniform(0.0, height, size=n_points)
    return np.column_stack([x_rand, y_rand])


class WSNCoverageObjective:
    """Pickle-friendly callable objective for WSN area coverage."""

    def __init__(
        self,
        num_sensors=None,
        width=100.0,
        height=60.0,
        grid_size=20,
        alpha=0.01,
        seed=None,
        vectorized=True,
    ):
        # num_sensors can be fixed or inferred later from the input vector length.
        self.num_sensors = None if num_sensors is None else int(num_sensors)
        self.width = float(width)
        self.height = float(height)
        self.grid_size = int(grid_size)
        self.alpha = float(alpha)
        self.seed = None if seed is None else int(seed)
        self.vectorized = bool(vectorized)

        if self.num_sensors is not None and self.num_sensors <= 0:
            raise ValueError("num_sensors must be > 0 when provided")
        if self.width <= 0.0:
            raise ValueError("width must be > 0")
        if self.height <= 0.0:
            raise ValueError("height must be > 0")
        if self.grid_size <= 1:
            raise ValueError("grid_size must be > 1")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be > 0")

        # Precompute the evaluation points once so every fitness call uses the same grid.
        self.grid_points = _build_grid_points(
            width=self.width,
            height=self.height,
            grid_size=self.grid_size,
            seed=self.seed,
        )

    def __call__(self, position):
        # Convert input to a 1D NumPy vector to keep the objective contract consistent.
        x = np.asarray(position, dtype=float)
        if x.ndim != 1:
            raise ValueError("position must be a 1D array")

        if self.num_sensors is None:
            # Infer the number of sensors from [x1, y1, ..., xM, yM].
            if x.size % 2 != 0:
                raise ValueError("position length must be even: [x1, y1, ..., xM, yM]")
            sensors_count = x.size // 2
        else:
            # If num_sensors is fixed, the input vector must match exactly.
            sensors_count = int(self.num_sensors)
            expected = 2 * sensors_count
            if x.size != expected:
                raise ValueError(f"position length must be {expected} for num_sensors={sensors_count}")

        coords = x.reshape(sensors_count, 2)

        # If bounds are bypassed, penalize invalid sensor coordinates.
        violation_x = np.maximum(0.0, -coords[:, 0]) + np.maximum(0.0, coords[:, 0] - self.width)
        violation_y = np.maximum(0.0, -coords[:, 1]) + np.maximum(0.0, coords[:, 1] - self.height)
        violation = float(np.sum(violation_x + violation_y))
        if violation > 0.0:
            return float(1.0 + violation)

        if self.vectorized:
            # Compute squared distances from every grid point to every sensor at once.
            diff = self.grid_points[:, None, :] - coords[None, :, :]
            dist_sq = np.sum(diff ** 2, axis=2, dtype=float)

            # Detection probability decreases with squared distance.
            p = np.exp(-self.alpha * dist_sq)

            # Combined coverage: probability that at least one sensor detects the point.
            coverage_point = 1.0 - np.prod(1.0 - p, axis=1)
            coverage_mean = float(np.mean(coverage_point))

            # PSO minimizes, so high coverage becomes low fitness.
            return float(1.0 - coverage_mean)

        # Non-vectorized fallback: easier to read and useful for comparison/debugging.
        coverage_values = []
        for point in self.grid_points:
            p_values = []
            for sensor in coords:
                dx = point[0] - sensor[0]
                dy = point[1] - sensor[1]
                p_values.append(np.exp(-self.alpha * (dx * dx + dy * dy)))

            p_values = np.asarray(p_values, dtype=float)
            coverage_values.append(1.0 - np.prod(1.0 - p_values))

        coverage_mean = float(np.mean(coverage_values))
        return float(1.0 - coverage_mean)


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
    return WSNCoverageObjective(
        num_sensors=num_sensors,
        width=width,
        height=height,
        grid_size=grid_size,
        alpha=alpha,
        seed=seed,
        vectorized=vectorized,
    )