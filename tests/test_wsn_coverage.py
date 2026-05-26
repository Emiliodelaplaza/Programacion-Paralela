import numpy as np
import pytest

from objectives import create_wsn_objective, make_wsn_bounds


def test_wsn_objective_returns_float_and_valid_range():
    objective = create_wsn_objective(num_sensors=3, width=100.0, height=60.0, grid_size=12, alpha=0.01)
    position = np.array([10.0, 10.0, 50.0, 30.0, 90.0, 50.0], dtype=float)
    fitness = objective(position)

    assert isinstance(fitness, float)
    assert 0.0 <= fitness <= 1.0


def test_wsn_objective_reproducibility_with_seeded_grid():
    objective_1 = create_wsn_objective(
        num_sensors=2,
        width=80.0,
        height=40.0,
        grid_size=10,
        alpha=0.015,
        seed=123,
    )
    objective_2 = create_wsn_objective(
        num_sensors=2,
        width=80.0,
        height=40.0,
        grid_size=10,
        alpha=0.015,
        seed=123,
    )
    position = np.array([10.0, 10.0, 70.0, 30.0], dtype=float)

    f1 = objective_1(position)
    f2 = objective_2(position)
    assert np.isclose(f1, f2, rtol=0.0, atol=1e-15)


def test_wsn_bounds_length_matches_num_sensors():
    low, high = make_wsn_bounds(num_sensors=5, width=100.0, height=60.0)

    assert low.shape == (10,)
    assert high.shape == (10,)


def test_wsn_reasonable_sensor_layout_does_not_fail():
    objective = create_wsn_objective(num_sensors=4, width=100.0, height=60.0, grid_size=14, alpha=0.01)
    position = np.array([10.0, 10.0, 90.0, 10.0, 10.0, 50.0, 90.0, 50.0], dtype=float)
    fitness = objective(position)

    assert isinstance(fitness, float)


def test_wsn_raises_for_wrong_position_size():
    objective = create_wsn_objective(num_sensors=3, width=100.0, height=60.0)
    with pytest.raises(ValueError):
        objective(np.array([1.0, 2.0, 3.0], dtype=float))
