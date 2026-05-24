from __future__ import annotations
from typing import Callable
import numpy as np

# Function that evaluates one particle position.
ObjectiveFn = Callable[[np.ndarray], float]


class SequentialEvaluator:
    """V0: sequential evaluation."""

    def __init__(self, objective: ObjectiveFn):
        self.objective = objective

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        """Evaluate all particles one by one."""
        return np.array([float(self.objective(x)) for x in X], dtype=float)