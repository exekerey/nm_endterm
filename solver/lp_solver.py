"""Linear program solver wrapper built on SciPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class LPSolution:
    x: np.ndarray
    status: str
    objective: float


class LPSolverError(RuntimeError):
    pass


def _augment_constraints(
    A: np.ndarray,
    b: np.ndarray,
    n: int,
    *,
    extra: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    A_list = [A]
    b_list = [b]
    if extra is not None and extra[0] is not None and extra[1] is not None:
        A_extra, b_extra = extra
        if A_extra.size and b_extra.size:
            if A_extra.shape[1] != n:
                raise ValueError("extra constraint matrix must have n columns")
            A_list.append(A_extra)
            b_list.append(b_extra)
    A_ub = np.vstack(A_list) if len(A_list) > 1 else A_list[0]
    b_ub = np.concatenate(b_list) if len(b_list) > 1 else b_list[0]
    return A_ub, b_ub


def solve_lp(
    objective: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    x_prev: Optional[np.ndarray] = None,
    delta: Optional[float] = None,
    extra: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> LPSolution:
    """Solve the primal LP with optional l1 trust region."""
    c = np.asarray(objective, dtype=float)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = c.size

    if A.ndim != 2 or A.shape[1] != n:
        raise ValueError("A must be a 2-D array with n columns")
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be a 1-D array matching the number of rows in A")

    bounds = [(0, None)] * n
    if x_prev is None or delta is None:
        A_ub, b_ub = _augment_constraints(A, b, n, extra=extra)
        res = linprog(
            -c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise LPSolverError(res.message)
        return LPSolution(res.x, res.status, float(c @ res.x))

    x_prev = np.asarray(x_prev, dtype=float).reshape(n)
    if delta <= 0:
        raise ValueError("delta must be positive for trust region constraints")

    zeros = np.zeros((A.shape[0], n), dtype=float)
    A_trust = []
    b_trust = []

    identity = np.eye(n)
    A_trust.append(np.hstack([identity, -identity]))
    b_trust.append(x_prev)
    A_trust.append(np.hstack([-identity, -identity]))
    b_trust.append(-x_prev)
    A_trust.append(np.hstack([np.zeros((1, n)), np.ones((1, n))]))
    b_trust.append(np.array([delta], dtype=float))

    A_ub_base = np.hstack([A, zeros])
    b_ub_base = b

    if extra is not None and extra[0] is not None and extra[1] is not None:
        A_extra = np.asarray(extra[0], dtype=float)
        b_extra = np.asarray(extra[1], dtype=float)
        if A_extra.size and b_extra.size:
            if A_extra.shape[1] != n:
                raise ValueError("extra constraint matrix must have n columns")
            A_ub_base = np.vstack([A_ub_base, np.hstack([A_extra, np.zeros((A_extra.shape[0], n))])])
            b_ub_base = np.concatenate([b_ub_base, b_extra])

    A_ub = np.vstack([A_ub_base, *A_trust])
    b_ub = np.concatenate([b_ub_base, *b_trust])

    c_aug = np.concatenate([-c, np.zeros(n, dtype=float)])
    bounds_aug = bounds + [(0, None)] * n

    res = linprog(
        c_aug,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds_aug,
        method="highs",
    )
    if not res.success:
        raise LPSolverError(res.message)
    x_opt = res.x[:n]
    return LPSolution(x_opt, res.status, float(c @ x_opt))

