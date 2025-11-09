"""Persistent portfolio state tracking target quantities per asset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

from trader import TradeDecision, TradingResult


@dataclass
class PortfolioState:
    positions: Dict[str, float] = field(default_factory=dict)
    cash: float = 0.0

    def to_dict(self) -> dict:
        return {"positions": self.positions, "cash": self.cash}

    @classmethod
    def from_dict(cls, payload: dict) -> PortfolioState:
        positions = {str(sym): float(qty) for sym, qty in payload.get("positions", {}).items()}
        cash = float(payload.get("cash", 0.0))
        return cls(positions=positions, cash=cash)


def load_state(path: Path) -> PortfolioState:
    if not path.exists():
        return PortfolioState()
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return PortfolioState()
    return PortfolioState.from_dict(payload)


def save_state(state: PortfolioState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state.to_dict(), indent=2))


def compute_trades(
    previous: PortfolioState,
    decisions: Iterable[TradeDecision],
    *,
    quantity_tol: float = 1e-9,
) -> List[dict]:
    trades: List[dict] = []
    for decision in decisions:
        current_qty = previous.positions.get(decision.symbol, 0.0)
        delta_qty = decision.quantity - current_qty
        if abs(delta_qty) <= quantity_tol:
            continue
        direction = "buy" if delta_qty > 0 else "sell"
        trades.append(
            {
                "symbol": decision.symbol,
                "action": direction,
                "quantity": delta_qty,
                "notional_usd": delta_qty * decision.price,
                "price": decision.price,
            }
        )
    return trades


def apply_target_state(result: TradingResult) -> PortfolioState:
    positions = {}
    for decision in result.decisions:
        qty = decision.quantity
        if abs(qty) > 1e-9:
            positions[decision.symbol] = qty
    return PortfolioState(positions=positions, cash=result.cash_reserve)
