import asyncio

import numpy as np


class AsyncEvaluator:
    """V3 evaluator: asynchronous fitness evaluation with asyncio."""

    def __init__(self, objective_async, workers):
        self.objective_async = objective_async
        self.workers = int(workers)
        self._closed = False

        if self.workers <= 0:
            raise ValueError("workers must be > 0 for AsyncEvaluator")

    async def _evaluate_one(self, x, semaphore):
        async with semaphore:
            return float(await self.objective_async(x))

    async def _evaluate_all(self, X):
        semaphore = asyncio.Semaphore(self.workers)
        tasks = [asyncio.create_task(self._evaluate_one(x, semaphore)) for x in X]
        values = await asyncio.gather(*tasks)
        return np.asarray(values, dtype=float)

    def evaluate_batch(self, X: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("AsyncEvaluator is closed")

        if X.size == 0:
            return np.asarray([], dtype=float)

        return asyncio.run(self._evaluate_all(X))

    def close(self):
        self._closed = True

