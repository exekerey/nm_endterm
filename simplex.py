"""Minimal implementation of the primal simplex method for <= LPs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class SimplexProblem:
    """Maximize c^T x subject to A x <= b, x >= 0."""

    A: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class SimplexSolution:
    x: np.ndarray
    objective: float
    status: str
    iterations: int


class SimplexError(RuntimeError):
    """Raised when the simplex solver cannot satisfy the constraints."""


class SimplexSolver:
    def __init__(self, *, tol: float = 1e-9, max_iterations: int = 10_000) -> None:
        self._tol = tol
        self._max_iterations = max_iterations

    def solve(self, problem: SimplexProblem) -> SimplexSolution:
        A = np.asarray(problem.A, dtype=float)
        b = np.asarray(problem.b, dtype=float).reshape(-1)
        c = np.asarray(problem.c, dtype=float).reshape(-1)

        if A.ndim != 2:
            raise SimplexError("constraint matrix A must be 2-D")
        m, n = A.shape
        if c.size != n:
            raise SimplexError("objective vector c must match number of columns in A")
        if b.size != m:
            raise SimplexError("constraint vector b must match rows in A")
        if np.any(b < -self._tol):
            raise SimplexError("simplex requires non-negative b; pre-process your rows")

        tableau = np.zeros((m + 1, n + m + 1), dtype=float)
        tableau[:m, :n] = A
        tableau[:m, n : n + m] = np.eye(m)
        tableau[:m, -1] = b
        tableau[-1, :n] = -c

        basis = list(range(n, n + m))
        status = "iteration_limit"
        iterations = 0

        while iterations < self._max_iterations:
            iterations += 1
            pivot_col = self._choose_entering_column(tableau)
            if pivot_col is None:
                status = "optimal"
                break
            pivot_row = self._choose_leaving_row(tableau, pivot_col)
            if pivot_row is None:
                status = "unbounded"
                break
            self._pivot(tableau, pivot_row, pivot_col)
            basis[pivot_row] = pivot_col

        x = np.zeros(n, dtype=float)
        for row, var in enumerate(basis):
            if var < n:
                x[var] = tableau[row, -1]
        objective = tableau[-1, -1]

        return SimplexSolution(x=x, objective=objective, status=status, iterations=iterations)

    def _choose_entering_column(self, tableau: np.ndarray) -> Optional[int]:
        last_row = tableau[-1, :-1]
        min_value = np.min(last_row)
        if min_value >= -self._tol:
            return None
        candidates = np.where(last_row == min_value)[0]
        return int(candidates[0])

    def _choose_leaving_row(self, tableau: np.ndarray, pivot_col: int) -> Optional[int]:
        column = tableau[:-1, pivot_col]
        rhs = tableau[:-1, -1]
        ratios = []
        for idx, coeff in enumerate(column):
            if coeff > self._tol:
                ratios.append((rhs[idx] / coeff, idx))
        if not ratios:
            return None
        ratios.sort()
        return ratios[0][1]

    def _pivot(self, tableau: np.ndarray, pivot_row: int, pivot_col: int) -> None:
        pivot_value = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_value
        for row in range(tableau.shape[0]):
            if row == pivot_row:
                continue
            factor = tableau[row, pivot_col]
            tableau[row, :] -= factor * tableau[pivot_row, :]
