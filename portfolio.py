from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence

import numpy as np

from market import AssetSnapshot
from solver.lp_solver import LPSolution, LPSolverError, solve_lp


@dataclass
class AllocationDecision:
    symbol: str
    price: float
    current_units: float
    target_units: float
    trade_units: float
    trade_usd: float
    allocation_usd: float
    expected_return_usd: float
    risk_contribution: float
    action: str


@dataclass
class TradingResult:
    solution: LPSolution
    decisions: List[AllocationDecision]
    total_expected_return: float
    total_risk: float
    total_capital: float
    total_equity: float
    cash_remaining: float


def _validate_inputs(
    assets: Sequence[AssetSnapshot],
    budget: float,
    risk_multiplier: float,
    max_weight: float,
) -> None:
    if not assets:
        raise ValueError("at least one asset snapshot is required.")
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if risk_multiplier <= 0:
        raise ValueError("risk multiplier must be positive.")
    if not 0 < max_weight <= 1.0:
        raise ValueError("max weight must lie in (0, 1].")


def build_lp(
    assets: Sequence[AssetSnapshot],
    budget: float,
    risk_multiplier: float,
    max_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_inputs(assets, budget, risk_multiplier, max_weight)

    expected_returns = np.array([asset.expected_return for asset in assets], dtype=float)
    risk_scores = np.array([asset.risk_score for asset in assets], dtype=float)
    n = len(assets)

    rows: List[np.ndarray] = []
    bounds: List[float] = []

    # Budget constraint
    rows.append(np.ones(n, dtype=float))
    bounds.append(budget)

    # Risk constraint
    rows.append(risk_scores)
    bounds.append(risk_multiplier * budget)

    # Max weight per asset
    max_allocation = max_weight * budget
    rows.extend(np.eye(n, dtype=float))
    bounds.extend([max_allocation] * n)

    A = np.vstack(rows)
    b = np.array(bounds, dtype=float)
    return expected_returns, A, b


def solve_trading_problem(
    assets: Sequence[AssetSnapshot],
    total_equity: float,
    risk_multiplier: float,
    max_weight: float,
    *,
    current_holdings: Mapping[str, float] | None = None,
) -> TradingResult:
    holdings = dict(current_holdings or {})
    current_values = {
        asset.symbol: float(holdings.get(asset.symbol, 0.0)) * float(asset.price)
        for asset in assets
    }
    invested_now = sum(current_values.values())
    cash_on_hand = float(total_equity) - invested_now
    if total_equity <= 0:
        raise ValueError("total equity must be positive.")
    if cash_on_hand < 0:
        # Numerical adjustments may drive this slightly negative; clamp at zero.
        if cash_on_hand < -1e-4:
            raise ValueError("current holdings exceed declared total equity.")
        cash_on_hand = 0.0

    c, A, b = build_lp(assets, total_equity, risk_multiplier, max_weight)
    solution = solve_lp(c, A, b)

    allocations = solution.x
    decisions: List[AllocationDecision] = []
    total_risk = 0.0

    for asset, allocation in zip(assets, allocations):
        amount = max(float(allocation), 0.0)
        price = float(asset.price)
        target_units = amount / price if price > 0 else 0.0
        current_units = float(holdings.get(asset.symbol, 0.0))
        current_value = current_units * price
        trade_value = amount - current_value
        trade_units = trade_value / price if price > 0 else 0.0
        action = "buy" if trade_value > 1e-6 else "sell" if trade_value < -1e-6 else "hold"
        expected_return = amount * asset.expected_return
        risk_contrib = amount * asset.risk_score
        total_risk += risk_contrib
        decisions.append(
            AllocationDecision(
                symbol=asset.symbol,
                price=price,
                current_units=current_units,
                target_units=target_units,
                trade_units=trade_units,
                trade_usd=trade_value,
                allocation_usd=amount,
                expected_return_usd=expected_return,
                risk_contribution=risk_contrib,
                action=action,
            )
        )

    total_capital = sum(decision.allocation_usd for decision in decisions)
    total_expected_return = sum(decision.expected_return_usd for decision in decisions)
    cash_remaining = max(float(total_equity) - total_capital, 0.0)

    return TradingResult(
        solution=solution,
        decisions=decisions,
        total_expected_return=total_expected_return,
        total_risk=total_risk,
        total_capital=total_capital,
        total_equity=float(total_equity),
        cash_remaining=cash_remaining,
    )
