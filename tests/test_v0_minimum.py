import numpy as np

from core import BoxBounds, ClampPolicy, PSO, PSOParams, SequentialEvaluator
from objectives import sphere


def _run_sphere(seed: int, dim: int = 10):
    # Create the search space for the Sphere function.
    bounds = BoxBounds(low=-5.0 * np.ones(dim), high=5.0 * np.ones(dim))
    evaluator = SequentialEvaluator(sphere)

    # Standard PSO configuration used in the tests.
    params = PSOParams(
        n_particles=40,
        n_iters=300,
        w=0.72,
        c1=1.49,
        c2=1.49,
        vmax_frac=0.2,
        seed=seed,
        stop_tol=1e-10,
        stagnation_iters=50,
    )

    pso = PSO(bounds=bounds, evaluator=evaluator, params=params, bounds_policy=ClampPolicy())
    return pso.run()


def test_seed_reproducibility_sphere():
    # The same seed should produce the same final result.
    _, best_f_1, _ = _run_sphere(seed=7)
    _, best_f_2, _ = _run_sphere(seed=7)
    assert np.isclose(best_f_1, best_f_2, rtol=0.0, atol=1e-15)


def test_clamp_policy_keeps_positions_in_bounds():
    bounds = BoxBounds(low=np.array([-1.0, -2.0]), high=np.array([1.0, 2.0]))
    policy = ClampPolicy()

    # Both coordinates start outside the valid range.
    x = np.array([1.5, -3.0])
    v = np.array([0.2, -0.4])

    x_new, _ = policy.apply(x, v, bounds)
    assert np.all(x_new >= bounds.low)
    assert np.all(x_new <= bounds.high)


def test_global_best_history_is_monotonic_non_increasing():
    # The global best value should never get worse.
    _, _, history = _run_sphere(seed=19)
    diffs = np.diff(np.asarray(history, dtype=float))
    assert np.all(diffs <= 1e-12)


def test_sphere_converges_below_threshold_in_d10():
    # Check that PSO reaches a small error in 10 dimensions.
    _, best_f, _ = _run_sphere(seed=7, dim=10)
    assert best_f < 1e-3