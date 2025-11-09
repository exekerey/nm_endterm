from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from market import AssetSnapshot
from simplex import SimplexError, SimplexProblem, SimplexSolver


@dataclass(frozen=True)
class TradeDecision:
    symbol: str
    weight: float
    allocation: float
    quantity: float
    price: float
    expected_return: float


@dataclass(frozen=True)
class TradingResult:
    objective: float
    decisions: List[TradeDecision]
    cash_reserve: float


class SimplexTradeEngine:
    def __init__(
        self,
        *,
        budget: float,
        max_allocation: float = 0.4,
        risk_budget: float = 0.25,
        solver: SimplexSolver | None = None,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if not 0 < max_allocation <= 1.0:
            raise ValueError("max_allocation must lie in (0, 1]")
        if risk_budget <= 0:
            raise ValueError("risk_budget must be positive")
        self._budget = budget
        self._max_allocation = max_allocation
        self._risk_budget = risk_budget
        self._solver = solver or SimplexSolver()

    def allocate(self, snapshots: Sequence[AssetSnapshot]) -> TradingResult:
        if not snapshots:
            raise ValueError("no asset snapshots provided")
        c = np.array([snap.expected_return for snap in snapshots], dtype=float)
        if np.allclose(c, 0.0):
            raise SimplexError("objective is zero for all assets; provide returns")

        A_rows: List[np.ndarray] = []
        b_values: List[float] = []
        A_rows.append(np.ones_like(c))
        b_values.append(1.0)

        if self._max_allocation < 1.0:
            A_rows.extend(np.eye(len(snapshots)))
            b_values.extend([self._max_allocation] * len(snapshots))

        risk_vector = np.array([max(snap.risk, 0.0) for snap in snapshots], dtype=float)
        if np.any(risk_vector > 0.0):
            A_rows.append(risk_vector)
            b_values.append(self._risk_budget)

        A = np.vstack(A_rows)
        b = np.array(b_values, dtype=float)
        problem = SimplexProblem(A=A, b=b, c=c)
        solution = self._solver.solve(problem)
        if solution.status != "optimal":
            raise SimplexError(f"simplex solver failed: {solution.status}")

        weights = np.clip(solution.x, 0.0, 1.0)
        total_weight = float(weights.sum())
        cash_reserve = max(0.0, 1.0 - total_weight) * self._budget

        decisions: List[TradeDecision] = []
        for weight, snap in zip(weights, snapshots):
            allocation = self._budget * weight
            quantity = allocation / snap.price if snap.price > 0 else 0.0
            decisions.append(
                TradeDecision(
                    symbol=snap.symbol,
                    weight=weight,
                    allocation=allocation,
                    quantity=quantity,
                    price=snap.price,
                    expected_return=snap.expected_return,
                )
            )

        return TradingResult(
            objective=solution.objective,
            decisions=decisions,
            cash_reserve=cash_reserve,
        )
