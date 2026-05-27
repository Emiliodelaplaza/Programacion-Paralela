import asyncio
import time

import numpy as np

from parallel.async_evaluator import AsyncEvaluator


def test_async_evaluator_returns_expected_shape_and_dtype():
    # Async objective equivalent to Sphere for a single particle.
    async def objective(x):
        await asyncio.sleep(0.0)
        return float(np.sum(x ** 2))

    evaluator = AsyncEvaluator(objective, workers=2)
    X = np.array([[1.0, 2.0], [3.0, 4.0], [0.5, -0.5]], dtype=float)
    y = evaluator.evaluate_batch(X)

    # The evaluator must return one fitness value per input particle.
    assert y.shape == (3,)
    assert y.dtype == float
    assert np.allclose(y, [5.0, 25.0, 0.5])


def test_async_evaluator_preserves_input_order():
    # Different sleeps force tasks to finish out of order.
    # The returned array should still match the original input order.
    async def objective(x):
        await asyncio.sleep(0.03 * float(x[0]))
        return float(x[0])

    evaluator = AsyncEvaluator(objective, workers=3)
    X = np.array([[3.0], [1.0], [2.0], [0.0]], dtype=float)
    y = evaluator.evaluate_batch(X)

    assert np.allclose(y, [3.0, 1.0, 2.0, 0.0])


def test_async_evaluator_applies_configured_latency():
    # This test checks that async evaluation really waits for the awaited objective.
    async def objective(x):
        await asyncio.sleep(0.02)
        return float(x[0] + 1.0)

    evaluator = AsyncEvaluator(objective, workers=2)
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)

    t0 = time.perf_counter()
    y = evaluator.evaluate_batch(X)
    elapsed = time.perf_counter() - t0

    assert np.allclose(y, [1.0, 2.0, 3.0, 4.0])
    assert elapsed >= 0.02
    assert elapsed < 1.0