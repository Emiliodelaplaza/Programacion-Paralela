import logging
from time import perf_counter

import numpy as np

from .bounds import ClampPolicy
from .pso import PSOParams

logger = logging.getLogger(__name__)


class PSOVectorized:
    """V4 PSO: vectorized update and vectorized fitness when possible."""

    def __init__(self, bounds, objective, params=None, bounds_policy=None):
        if params is None:
            params = PSOParams()

        self.bounds = bounds
        self.objective = objective
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.bounds_policy = bounds_policy or ClampPolicy()

        span = self.bounds.high - self.bounds.low
        self.vmax = self.p.vmax_frac * span

        self.X = None
        self.V = None
        self.PbestX = None
        self.PbestF = None
        self.gbest_x = None
        self.gbest_f = None
        self.last_run_stats = None
        self.last_positions_history = None
        self.last_positions_iters = None
        self.last_gbest_history = None
        self._timing_eval_seconds = 0.0
        self._timing_update_seconds = 0.0

    def _evaluate_batch_vectorized(self, X):
        name = getattr(self.objective, "__name__", "")

        if name == "sphere":
            return np.sum(X ** 2, axis=1, dtype=float)

        if name == "rosenbrock":
            x0 = X[:, :-1]
            x1 = X[:, 1:]
            return np.sum(100.0 * (x1 - x0 ** 2) ** 2 + (1.0 - x0) ** 2, axis=1, dtype=float)

        if name == "rastrigin":
            n = X.shape[1]
            return 10.0 * n + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1, dtype=float)

        if name == "ackley":
            n = X.shape[1]
            sum_sq = np.sum(X ** 2, axis=1, dtype=float)
            sum_cos = np.sum(np.cos(2.0 * np.pi * X), axis=1, dtype=float)
            term1 = -20.0 * np.exp(-0.2 * np.sqrt(sum_sq / n))
            term2 = -np.exp(sum_cos / n)
            return term1 + term2 + 20.0 + np.e

        return np.array([float(self.objective(x)) for x in X], dtype=float)

    def _evaluate_batch_timed(self, X):
        start = perf_counter()
        F = self._evaluate_batch_vectorized(X)
        self._timing_eval_seconds += perf_counter() - start
        return np.asarray(F, dtype=float)

    def _init_swarm(self):
        self.X = self.bounds.sample_uniform(self.p.n_particles, self.rng)
        self.V = self.rng.uniform(-self.vmax, self.vmax, size=self.X.shape)
        F = self._evaluate_batch_timed(self.X)

        self.PbestX = self.X.copy()
        self.PbestF = F.copy()
        idx = int(np.argmin(F))
        self.gbest_f = float(F[idx])
        self.gbest_x = self.X[idx].copy()

    def run(self, track_positions=False, track_every=1):
        if track_every <= 0:
            raise ValueError("track_every must be > 0")

        run_start = perf_counter()
        self._timing_eval_seconds = 0.0
        self._timing_update_seconds = 0.0
        self.last_run_stats = None
        self.last_positions_history = None
        self.last_positions_iters = None
        self.last_gbest_history = None
        self._init_swarm()
        assert self.gbest_x is not None and self.gbest_f is not None

        history = [float(self.gbest_f)]
        no_improve_count = 0
        positions_history = [] if track_positions else None
        positions_iters = [] if track_positions else None
        gbest_history = [] if track_positions else None

        if track_positions:
            positions_history.append(self.X.copy())
            positions_iters.append(0)
            gbest_history.append(self.gbest_x.copy())

        for iter_idx in range(1, self.p.n_iters + 1):
            best_before = float(self.gbest_f)

            update_start = perf_counter()
            r1 = self.rng.random((self.p.n_particles, self.bounds.dim))
            r2 = self.rng.random((self.p.n_particles, self.bounds.dim))

            self.V = (
                self.p.w * self.V
                + self.p.c1 * r1 * (self.PbestX - self.X)
                + self.p.c2 * r2 * (self.gbest_x - self.X)
            )
            self.V = np.clip(self.V, -self.vmax, self.vmax)

            self.X = self.X + self.V
            self.X, self.V = self.bounds_policy.apply(self.X, self.V, self.bounds)
            self._timing_update_seconds += perf_counter() - update_start

            F = self._evaluate_batch_timed(self.X)

            improved_mask = F < self.PbestF
            if np.any(improved_mask):
                self.PbestF[improved_mask] = F[improved_mask]
                self.PbestX[improved_mask] = self.X[improved_mask]

            best_idx = int(np.argmin(F))
            best_now = float(F[best_idx])
            if best_now < self.gbest_f:
                self.gbest_f = best_now
                self.gbest_x = self.X[best_idx].copy()

            history.append(float(self.gbest_f))
            improvement = best_before - float(self.gbest_f)

            if improvement > self.p.stop_tol:
                no_improve_count = 0
            else:
                no_improve_count += 1

            stop_due_target = self.p.target_fitness is not None and self.gbest_f <= self.p.target_fitness
            stop_due_stagnation = (
                self.p.stagnation_iters is not None and no_improve_count >= self.p.stagnation_iters
            )

            if track_positions:
                should_store_positions = (
                    (iter_idx % track_every == 0)
                    or stop_due_target
                    or stop_due_stagnation
                    or (iter_idx == self.p.n_iters)
                )
                if should_store_positions:
                    positions_history.append(self.X.copy())
                    positions_iters.append(iter_idx)
                    gbest_history.append(self.gbest_x.copy())

            if self.p.log_every_iters is not None:
                should_log = (iter_idx % self.p.log_every_iters == 0) or stop_due_target or stop_due_stagnation
                if should_log:
                    logger.info(
                        "iter=%d best_f=%.6e elapsed_seconds=%.6f",
                        iter_idx,
                        float(self.gbest_f),
                        perf_counter() - run_start,
                    )

            if stop_due_target or stop_due_stagnation:
                break

        total_run_seconds = perf_counter() - run_start
        self.last_run_stats = {
            "total_run_seconds": float(total_run_seconds),
            "fitness_eval_seconds": float(self._timing_eval_seconds),
            "state_update_seconds": float(self._timing_update_seconds),
        }

        if track_positions:
            self.last_positions_history = positions_history
            self.last_positions_iters = positions_iters
            self.last_gbest_history = gbest_history

        return self.gbest_x.copy(), float(self.gbest_f), history
