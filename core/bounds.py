import numpy as np


class BoxBounds:
    """Axis-aligned bounds for each dimension."""

    def __init__(self, low, high):
        self.low = np.asarray(low, dtype=float)
        self.high = np.asarray(high, dtype=float)

        if self.low.ndim != 1 or self.high.ndim != 1:
            raise ValueError("low/high must be 1D arrays")
        if self.low.shape != self.high.shape:
            raise ValueError("low/high must have the same shape")
        if np.any(self.low >= self.high):
            raise ValueError("each low must be < high")

        self.dim = int(self.low.size)

    def sample_uniform(self, n, rng):
        """Sample n random positions uniformly inside the bounds."""
        return rng.uniform(self.low, self.high, size=(n, self.dim))


class ClampPolicy:
    """Clamp particles to the search space bounds."""

    def apply(self, x, v, bounds):
        """Clamp positions and cancel velocity where a bound is hit."""
        x_new = np.clip(x, bounds.low, bounds.high)

        # If a coordinate goes outside the bounds, its velocity is reset.
        # This prevents the particle from repeatedly leaving the search space.
        hit_mask = (x < bounds.low) | (x > bounds.high)
        v_new = np.asarray(v, dtype=float).copy()
        v_new[hit_mask] = 0.0
        return x_new, v_new