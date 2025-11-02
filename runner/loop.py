"""Main orchestration loop for adaptive LP solving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple

import numpy as np

from config import Config
from env import Environment
from learners.rls import RLSState, rls_update
from learners.sgd import SGDState, sgd_update
from solver.lp_solver import LPSolverError, LPSolution, solve_lp
from utils.trust_region import adjust_delta


@dataclass
class RoundRecord:
    t: int
    x: np.ndarray
    Z: float
    Zhat: float
    err: float
    c: np.ndarray
    Delta: float
    objective_status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "x": self.x.tolist(),
            "Z": self.Z,
            "Zhat": self.Zhat,
            "err": self.err,
            "c": self.c.tolist(),
            "Delta": self.Delta,
            "lp_status": self.objective_status,
        }


def run_loop(cfg: Config, env: Environment) -> list[RoundRecord]:
    rng = np.random.default_rng(cfg.run.seed)
    c = cfg.init.c.astype(float, copy=True)
    learner_state: Optional[object]
    if cfg.learning.algo == "sgd":
        learner_state = SGDState(coefficients=c.copy())
    elif cfg.learning.algo == "rls":
        learner_state = RLSState(theta=cfg.init.theta.copy(), P=cfg.init.P.copy())
        c = cfg.init.theta.copy()
    else:
        raise ValueError(f"unsupported learning algo: {cfg.learning.algo}")

    delta = cfg.stability.Delta0
    x_prev = np.zeros(cfg.problem.n, dtype=float)
    history: list[RoundRecord] = []

    for t in range(cfg.run.T_max):
        extra = env.extra_constraints(t)
        solve_objective = c.copy()

        if cfg.learning.exploration == "epsilon":
            if rng.random() < cfg.learning.epsilon:
                j = t % cfg.problem.p
                solve_objective = _unit_vector(cfg.problem.p, j)
        elif cfg.learning.exploration == "cycle":
            if cfg.learning.k > 0 and (t % cfg.learning.k) == 0:
                j = (t // cfg.learning.k) % cfg.problem.p
                solve_objective = _unit_vector(cfg.problem.p, j)

        try:
            solution = _solve_with_trust_region(
                solve_objective,
                cfg,
                env,
                x_prev,
                delta,
                extra,
            )
        except LPSolverError as exc:
            raise RuntimeError(f"LP infeasible at iteration {t}: {exc}") from exc

        phi_x = env.features(solution.x)
        Zhat = float(c @ phi_x)
        Z = float(env.observe(solution.x))

        prev_record = history[-1] if history else None
        if cfg.learning.algo == "sgd":
            assert isinstance(learner_state, SGDState)
            learner_state, err = sgd_update(
                learner_state,
                phi_x,
                Z,
                Zhat,
                eta=cfg.learning.eta,
                project=cfg.projector,
            )
            c = learner_state.coefficients.copy()
        else:
            assert isinstance(learner_state, RLSState)
            learner_state, projected, err = rls_update(
                learner_state,
                phi_x,
                Z,
                lam=cfg.learning.lam,
                project=cfg.projector,
            )
            c = projected.copy()

        if cfg.stability.trust_region:
            delta = adjust_delta(
                delta,
                err,
                small_thresh=cfg.stability.err_small,
                up=cfg.stability.up_factor,
                down=cfg.stability.down_factor,
                delta_min=cfg.stability.Delta_min,
                delta_max=cfg.stability.Delta_max,
            )

        history.append(
            RoundRecord(
                t=t,
                x=solution.x.copy(),
                Z=Z,
                Zhat=Zhat,
                err=err,
                c=c.copy(),
                Delta=delta,
                objective_status=solution.status,
            )
        )

        if prev_record is not None:
            delta_c = np.linalg.norm(c - prev_record.c, ord=1)
            if delta_c < cfg.run.tol_c and abs(err) < cfg.run.tol_err:
                break

        x_prev = solution.x.copy()

    return history


def _solve_with_trust_region(
    objective: np.ndarray,
    cfg: Config,
    env: Environment,
    x_prev: np.ndarray,
    delta: float,
    extra: Optional[Tuple[np.ndarray, np.ndarray]],
) -> LPSolution:
    if cfg.stability.trust_region:
        return solve_lp(
            objective,
            cfg.problem.A,
            cfg.problem.b,
            x_prev=x_prev,
            delta=delta,
            extra=extra,
        )
    return solve_lp(
        objective,
        cfg.problem.A,
        cfg.problem.b,
        x_prev=None,
        delta=None,
        extra=extra,
    )


def _unit_vector(dim: int, idx: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=float)
    vec[idx % dim] = 1.0
    return vec
