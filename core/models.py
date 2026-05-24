import numpy as np


class Particle:
    """Container for the state of one particle in PSO."""

    def __init__(self, x, v, best_x=None, best_f=None):
        # Current position of the particle.
        self.x = np.asarray(x, dtype=float).copy()

        # Current velocity of the particle.
        self.v = np.asarray(v, dtype=float).copy()

        # Best position found so far by this particle.
        self.best_x = (
            self.x.copy()
            if best_x is None
            else np.asarray(best_x, dtype=float).copy()
        )

        # Best fitness value found so far by this particle.
        if best_f is None:
            raise ValueError("best_f is required")
        self.best_f = float(best_f)