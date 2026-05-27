import numpy as np

from core import BoxBounds, ClampPolicy, PSOParams, PSOVectorized
from objectives import sphere


def _run_sphere_v4(seed: int, dim: int = 10):
    # Helper used by the tests to keep the V4 configuration identical.
    bounds = BoxBounds(low=-5.0 * np.ones(dim), high=5.0 * np.ones(dim))
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

    pso = PSOVectorized(bounds=bounds, objective=sphere, params=params, bounds_policy=ClampPolicy())
    return pso.run()


def test_v4_run_output_shape_and_types():
    best_x, best_f, history = _run_sphere_v4(seed=7, dim=10)

    # The optimizer should return a solution vector, scalar fitness and convergence history.
    assert best_x.shape == (10,)
    assert isinstance(best_f, float)
    assert isinstance(history, list)
    assert len(history) >= 2


def test_v4_seed_reproducibility_sphere():
    # Running with the same seed should produce the same final best fitness.
    _, best_f_1, _ = _run_sphere_v4(seed=7, dim=10)
    _, best_f_2, _ = _run_sphere_v4(seed=7, dim=10)

    assert np.isclose(best_f_1, best_f_2, rtol=0.0, atol=1e-15)


def test_v4_sphere_converges_reasonably():
    # Sphere has optimum 0, so V4 should get very close with this configuration.
    _, best_f, _ = _run_sphere_v4(seed=7, dim=10)

    assert best_f < 1e-3