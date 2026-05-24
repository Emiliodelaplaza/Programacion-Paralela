import logging
from time import perf_counter

import numpy as np

from .models import Particle
from .bounds import ClampPolicy

logger = logging.getLogger(__name__)


class PSOParams:
    """Parameters for the baseline PSO implementation (V0)."""

    def __init__(
        self,
        n_particles=30,
        n_iters=200,
        w=0.72,
        c1=1.49,
        c2=1.49,
        vmax_frac=0.2,
        seed=123,
        stop_tol=1e-12,
        stagnation_iters=None,
        target_fitness=None,
        log_every_iters=10,
    ):
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.vmax_frac = float(vmax_frac)
        self.seed = int(seed)
        self.stop_tol = float(stop_tol)
        self.stagnation_iters = None if stagnation_iters is None else int(stagnation_iters)
        self.target_fitness = None if target_fitness is None else float(target_fitness)
        self.log_every_iters = None if log_every_iters is None else int(log_every_iters)

        if self.n_particles <= 0:
            raise ValueError("n_particles must be > 0")
        if self.n_iters <= 0:
            raise ValueError("n_iters must be > 0")
        if self.vmax_frac <= 0:
            raise ValueError("vmax_frac must be > 0")
        if self.stop_tol < 0:
            raise ValueError("stop_tol must be >= 0")
        if self.stagnation_iters is not None and self.stagnation_iters <= 0:
            raise ValueError("stagnation_iters must be > 0 when provided")
        if self.log_every_iters is not None and self.log_every_iters <= 0:
            raise ValueError("log_every_iters must be > 0 when provided")


class PSO:
    """Baseline PSO: sequential evaluator and global-best update."""

    def __init__(self, bounds, evaluator, params=None, bounds_policy=None):
        if params is None:
            params = PSOParams()

        self.bounds = bounds
        self.evaluator = evaluator
        self.p = params
        self.rng = np.random.default_rng(self.p.seed)
        self.bounds_policy = bounds_policy or ClampPolicy()

        # Maximum velocity is set as a fraction of the search-space size.
        span = self.bounds.high - self.bounds.low
        self.vmax = self.p.vmax_frac * span

        self.swarm = []
        self.gbest_x = None
        self.gbest_f = None
        self.last_run_stats = None
        self.last_positions_history = None
        self.last_positions_iters = None
        self.last_gbest_history = None
        self._timing_eval_seconds = 0.0
        self._timing_update_seconds = 0.0

    def _evaluate_batch_timed(self, X):
        """Evaluate particles and accumulate evaluation time."""
        start = perf_counter()
        F = self.evaluator.evaluate_batch(X)
        self._timing_eval_seconds += perf_counter() - start
        return F

    def _init_swarm(self):
        """Create the initial swarm and set the first global best."""
        X = self.bounds.sample_uniform(self.p.n_particles, self.rng)
        V = self.rng.uniform(-self.vmax, self.vmax, size=X.shape)
        F = self._evaluate_batch_timed(X)

        self.swarm = []
        self.gbest_x = None
        self.gbest_f = None

        for i in range(self.p.n_particles):
            x = X[i].copy()
            v = V[i].copy()
            f = float(F[i])
            part = Particle(x=x, v=v, best_x=x.copy(), best_f=f)
            self.swarm.append(part)

            if self.gbest_f is None or f < self.gbest_f:
                self.gbest_f = f
                self.gbest_x = x.copy()

    def run(self, track_positions=False, track_every=1):
        """Run PSO and return best position, best fitness, and history."""
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
            initial_X = np.array([part.x.copy() for part in self.swarm], dtype=float)
            positions_history.append(initial_X)
            positions_iters.append(0)
            gbest_history.append(self.gbest_x.copy())

        for iter_idx in range(1, self.p.n_iters + 1):
            best_before = float(self.gbest_f)
            X = np.empty((self.p.n_particles, self.bounds.dim), dtype=float)

            update_start = perf_counter()
            for i, part in enumerate(self.swarm):
                r1 = self.rng.random(self.bounds.dim)
                r2 = self.rng.random(self.bounds.dim)

                # Standard PSO velocity update.
                part.v = (
                    self.p.w * part.v
                    + self.p.c1 * r1 * (part.best_x - part.x)
                    + self.p.c2 * r2 * (self.gbest_x - part.x)
                )
                part.v = np.clip(part.v, -self.vmax, self.vmax)

                # Move particle and apply bounds.
                part.x = part.x + part.v
                part.x, part.v = self.bounds_policy.apply(part.x, part.v, self.bounds)
                X[i] = part.x
            self._timing_update_seconds += perf_counter() - update_start

            F = self._evaluate_batch_timed(X)

            for i, part in enumerate(self.swarm):
                f = float(F[i])

                # Update personal best.
                if f < part.best_f:
                    part.best_f = f
                    part.best_x = part.x.copy()

                # Update global best.
                if f < self.gbest_f:
                    self.gbest_f = f
                    self.gbest_x = part.x.copy()

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
                    positions_history.append(X.copy())
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