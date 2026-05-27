import asyncio

import numpy as np
from .wsn_coverage import create_wsn_objective, make_wsn_bounds


def sphere(x):
    # Sphere is a simple convex benchmark: the optimum is at x = 0.
    return float(np.sum(x ** 2))


def rosenbrock(x):
    # Rosenbrock has a narrow curved valley, so it is harder than Sphere.
    x0 = x[:-1]
    x1 = x[1:]
    return float(np.sum(100.0 * (x1 - x0 ** 2) ** 2 + (1.0 - x0) ** 2))


def rastrigin(x):
    # Rastrigin is multimodal: many local minima make it useful for testing PSO.
    n = x.size
    return float(10.0 * n + np.sum(x ** 2 - 10.0 * np.cos(2.0 * np.pi * x)))


def ackley(x):
    # Ackley combines an almost flat outer region with many local variations.
    n = x.size
    sum_sq = np.sum(x ** 2)
    sum_cos = np.sum(np.cos(2.0 * np.pi * x))
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
    term2 = -np.exp(sum_cos / n)
    return float(term1 + term2 + 20.0 + np.e)


def make_async_objective(objective, latency):
    # Wrap a normal objective into an async objective for the V3 asyncio strategy.
    async def _objective(x):
        # Simulate I/O or service latency so asyncio has a meaningful workload.
        await asyncio.sleep(float(latency))
        return float(objective(x))

    return _objective


def _wsn_bounds_from_dim(dim):
    # WSN encodes each sensor with two variables: x and y.
    dim = int(dim)
    if dim <= 0 or dim % 2 != 0:
        raise ValueError("wsn objective requires an even dimension (2 * num_sensors)")

    num_sensors = dim // 2
    return make_wsn_bounds(num_sensors=num_sensors, width=100.0, height=60.0)


# Central registry used by the experiment scripts.
# Each entry provides the objective and the default search bounds.
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
    "wsn": {
        # num_sensors=None allows the number of sensors to be inferred from dim.
        "func": create_wsn_objective(
            num_sensors=None,
            width=100.0,
            height=60.0,
            grid_size=20,
            alpha=0.01,
            seed=None,
            vectorized=True,
        ),
        # low/high are kept for compatibility, but bounds_builder is the real WSN bounds source.
        "low": 0.0,
        "high": 100.0,
        "bounds_builder": _wsn_bounds_from_dim,
    },
}
