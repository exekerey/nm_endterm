from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market import AssetSnapshot
from simplex import SimplexProblem, SimplexSolver
from trader import SimplexTradeEngine


def test_simplex_solver_finds_optimum():
    A = np.array(
        [
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    b = np.array([4.0, 2.0, 3.0])
    c = np.array([3.0, 2.0])
    solver = SimplexSolver()
    result = solver.solve(SimplexProblem(A=A, b=b, c=c))
    assert result.status == "optimal"
    np.testing.assert_allclose(result.x, [2.0, 2.0], atol=1e-6)
    assert result.objective == pytest.approx(10.0, rel=1e-6)


def test_trade_engine_respects_risk_budget():
    now = datetime.now(timezone.utc)
    snapshots = [
        AssetSnapshot(
            symbol="AAAUSDT",
            price=10.0,
            expected_return=0.2,
            risk=0.9,
            volume=1000.0,
            timestamp=now,
        ),
        AssetSnapshot(
            symbol="BBBUSDT",
            price=5.0,
            expected_return=0.1,
            risk=0.1,
            volume=2000.0,
            timestamp=now,
        ),
    ]
    engine = SimplexTradeEngine(budget=1_000.0, max_allocation=0.9, risk_budget=0.3)
    result = engine.allocate(snapshots)
    weights = np.array([decision.weight for decision in result.decisions])
    assert weights[0] == pytest.approx(0.25, rel=1e-6)
    assert weights[1] == pytest.approx(0.75, rel=1e-6)
    risk = 0.9 * weights[0] + 0.1 * weights[1]
    assert risk == pytest.approx(0.3, rel=1e-6)
    assert result.cash_reserve == pytest.approx(0.0, abs=1e-6)
