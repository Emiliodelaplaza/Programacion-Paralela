from concurrent.futures import ProcessPoolExecutor

import numpy as np


def _eval_batch_worker(objective, X_batch):
    # Evaluate one batch of particles inside a worker process.
    return [float(objective(x)) for x in X_batch]


def _iter_batches(X, batch_size):
    # Split the population matrix into smaller batches.
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield X[start:end]


class ProcessPoolEvaluator:
    """V2 evaluator: parallel fitness evaluation with processes and batching."""

    def __init__(self, objective, workers, batch_size):
        self.objective = objective
        self.workers = int(workers)
        self.batch_size = int(batch_size)

        if self.workers <= 0:
            raise ValueError("workers must be > 0 for ProcessPoolEvaluator")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0 for ProcessPoolEvaluator")

        # Create the pool of worker processes.
        self._executor = ProcessPoolExecutor(max_workers=self.workers)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        # Prevent evaluation after the executor has been closed.
        if self._executor is None:
            raise RuntimeError("ProcessPoolEvaluator is closed")

        # Return an empty array if there are no particles to evaluate.
        if X.size == 0:
            return np.asarray([], dtype=float)

        # Submit one task per batch.
        futures = [
            self._executor.submit(_eval_batch_worker, self.objective, X_batch)
            for X_batch in _iter_batches(X, self.batch_size)
        ]

        # Collect all results from the worker processes.
        values = []
        for fut in futures:
            values.extend(fut.result())

        return np.asarray(values, dtype=float)

    def close(self):
        # Close the process pool explicitly.
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    def __del__(self):
        # Safety cleanup in case close() was not called manually.
        try:
            if getattr(self, "_executor", None) is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
        except Exception:
            pass