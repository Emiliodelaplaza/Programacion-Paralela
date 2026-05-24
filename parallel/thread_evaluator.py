from concurrent.futures import ThreadPoolExecutor

import numpy as np


class ThreadPoolEvaluator:
    """V1 evaluator: parallel fitness evaluation using threads."""

    def __init__(self, objective, workers):
        self.objective = objective
        self.workers = int(workers)

        if self.workers <= 0:
            raise ValueError("workers must be > 0 for ThreadPoolEvaluator")

        # Create the pool of worker threads.
        self._executor = ThreadPoolExecutor(max_workers=self.workers)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        # Prevent evaluation after the executor has been closed.
        if self._executor is None:
            raise RuntimeError("ThreadPoolEvaluator is closed")

        # Evaluate all particles in parallel with threads.
        values = list(self._executor.map(self.objective, X))
        return np.asarray(values, dtype=float)

    def close(self):
        # Close the thread pool explicitly.
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self):
        # Safety cleanup in case close() was not called manually.
        try:
            if getattr(self, "_executor", None) is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
        except Exception:
            pass