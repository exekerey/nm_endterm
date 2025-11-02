from pathlib import Path

import numpy as np
import pytest

from config import Config, InitConfig, LearningConfig, ProblemConfig, RunConfig, StabilityConfig
from learners.rls import RLSState, rls_update
from runner.loop import run_loop
from solver.lp_solver import solve_lp
from utils.projection import project_simplex


def test_project_simplex_properties():
    vec = np.array([0.5, -0.2, 1.7])
    projected = project_simplex(vec)
    np.testing.assert_allclose(projected.sum(), 1.0, atol=1e-6)
    assert np.all(projected >= -1e-6)


def test_solver_respects_trust_region():
    c = np.array([1.0, 1.5])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.2, 0.9, 0.6])
    x_prev = np.array([0.2, 0.2])
    delta = 0.3

    solution = solve_lp(c, A, b, x_prev=x_prev, delta=delta)
    assert np.all(A @ solution.x <= b + 1e-6)
    assert np.linalg.norm(solution.x - x_prev, ord=1) <= delta + 1e-6


def test_rls_update_reduces_error():
    state = RLSState(theta=np.zeros(2), P=10.0 * np.eye(2))
    features = np.array([1.0, 0.5])
    feedback = 1.2

    state_next, projected, err = rls_update(
        state,
        features,
        feedback,
        lam=0.99,
        project=project_simplex,
    )
    assert abs(err) < abs(feedback)
    assert projected.sum() == pytest.approx(1.0, rel=1e-6, abs=1e-6)
    assert np.all(projected >= -1e-6)
    # Expect updated theta to move in the direction of feedback.
    assert state_next.theta[0] > 0


class _LinearEnv:
    def __init__(self, weights):
        self.weights = weights
        self.n = self.p = len(weights)

    def observe(self, x):
        return float(self.weights @ x)

    def features(self, x):
        return x

    def extra_constraints(self, iteration):
        return None


def test_run_loop_recovers_linear_weights():
    weights = np.array([0.6, 0.4])
    A = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    b = np.array([1.0, 0.7, 0.6])

    cfg = Config(
        problem=ProblemConfig(
            n=2,
            p=2,
            A=A,
            b=b,
            feature_mode="identity",
            env="time_alloc",
        ),
        learning=LearningConfig(
            algo="rls",
            eta=0.1,
            lam=0.99,
            projection="simplex",
            exploration="none",
            epsilon=0.0,
            k=1,
        ),
        stability=StabilityConfig(
            trust_region=False,
            Delta0=1.0,
            Delta_min=0.1,
            Delta_max=2.0,
            up_factor=1.0,
            down_factor=1.0,
            err_small=0.05,
        ),
        run=RunConfig(
            T_max=80,
            tol_c=1e-3,
            tol_err=1e-2,
            seed=0,
            logging=False,
        ),
        init=InitConfig(
            c=np.array([0.5, 0.5]),
            theta=np.array([0.5, 0.5]),
            P=50.0 * np.eye(2),
        ),
        base_path=Path("."),
        environment={},
    )
    env = _LinearEnv(weights)
    history = run_loop(cfg, env)
    assert history, "history should not be empty"
    final_c = history[-1].c
    np.testing.assert_allclose(final_c, weights / weights.sum(), atol=0.1)
