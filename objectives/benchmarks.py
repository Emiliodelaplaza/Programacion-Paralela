import asyncio

import numpy as np


def sphere(x):
    return float(np.sum(x ** 2))


def rosenbrock(x):
    x0 = x[:-1]
    x1 = x[1:]
    return float(np.sum(100.0 * (x1 - x0 ** 2) ** 2 + (1.0 - x0) ** 2))


def rastrigin(x):
    n = x.size
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x):
    n = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return float(term1 + term2 + 20.0 + np.e)


def make_async_objective(objective, latency):
    async def _objective(x):
        await asyncio.sleep(float(latency))
        return float(objective(x))

    return _objective


BENCHMARKS = {
    "sphere": {
        "func": sphere,
        "low": -5.12,
        "high": 5.12,
    },
    "rosenbrock": {
        "func": rosenbrock,
        "low": -2.048,
        "high": 2.048,
    },
    "rastrigin": {
        "func": rastrigin,
        "low": -5.12,
        "high": 5.12,
    },
    "ackley": {
        "func": ackley,
        "low": -32.768,
        "high": 32.768,
    },
}
