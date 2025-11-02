from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DB_PATH = Path(__file__).resolve().parent / "trades.db"


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA journal_mode=WAL;")
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                trigger TEXT NOT NULL,
                capital REAL NOT NULL,
                expected_pnl REAL NOT NULL,
                expected_roi REAL NOT NULL,
                risk_usage REAL NOT NULL,
                allocations TEXT NOT NULL,
                objective_value REAL NOT NULL,
                decision_vector TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS holdings (
                symbol TEXT PRIMARY KEY,
                units REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_meta (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO portfolio_meta (key, value)
            VALUES ('cash_balance', 0.0)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS price_history (
                symbol TEXT NOT NULL,
                timestamp REAL NOT NULL,
                price REAL NOT NULL,
                PRIMARY KEY (symbol, timestamp)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_price_history_symbol_time
            ON price_history (symbol, timestamp)
            """
        )
        conn.commit()


def append_trade(record: Dict[str, Any]) -> None:
    allocations_payload = json.dumps(record.get("allocations", []))
    decision_vector_payload = json.dumps(record.get("decision_vector", []))
    timestamp = record["timestamp"]
    if isinstance(timestamp, datetime):
        timestamp_str = timestamp.astimezone(timezone.utc).isoformat()
    else:
        timestamp_str = str(timestamp)

    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO trades (
                timestamp,
                trigger,
                capital,
                expected_pnl,
                expected_roi,
                risk_usage,
                allocations,
                objective_value,
                decision_vector
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp_str,
                record["trigger"],
                float(record["capital"]),
                float(record["expected_pnl"]),
                float(record["expected_roi"]),
                float(record["risk_usage"]),
                allocations_payload,
                float(record.get("objective_value", 0.0)),
                decision_vector_payload,
            ),
        )
        conn.commit()


def fetch_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    query = """
        SELECT
            timestamp,
            trigger,
            capital,
            expected_pnl,
            expected_roi,
            risk_usage,
            allocations,
            objective_value,
            decision_vector
        FROM trades
        ORDER BY id DESC
    """
    if limit is not None:
        query += " LIMIT ?"

    with _connect() as conn:
        cursor = conn.execute(query, (limit,) if limit is not None else ())
        rows = cursor.fetchall()

    history: List[Dict[str, Any]] = []
    for row in reversed(rows):
        timestamp_raw = row["timestamp"]
        try:
            timestamp = datetime.fromisoformat(timestamp_raw)
        except ValueError:
            timestamp = datetime.fromtimestamp(0, tz=timezone.utc)
        allocations = json.loads(row["allocations"]) if row["allocations"] else []
        decision_vector = json.loads(row["decision_vector"]) if row["decision_vector"] else []
        history.append(
            {
                "timestamp": timestamp,
                "trigger": row["trigger"],
                "capital": float(row["capital"]),
                "expected_pnl": float(row["expected_pnl"]),
                "expected_roi": float(row["expected_roi"]),
                "risk_usage": float(row["risk_usage"]),
                "allocations": allocations,
                "objective_value": float(row["objective_value"]),
                "decision_vector": decision_vector,
            }
        )
    return history


def load_portfolio() -> Tuple[float, Dict[str, float]]:
    with _connect() as conn:
        cash_row = conn.execute(
            "SELECT value FROM portfolio_meta WHERE key = 'cash_balance'"
        ).fetchone()
        cash_balance = float(cash_row["value"]) if cash_row else 0.0
        cursor = conn.execute("SELECT symbol, units FROM holdings")
        holdings = {
            row["symbol"]: float(row["units"])
            for row in cursor.fetchall()
            if abs(float(row["units"])) > 1e-9
        }
    return cash_balance, holdings


def save_portfolio(cash_balance: float, holdings: Mapping[str, float]) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    normalized = {
        symbol: float(units)
        for symbol, units in holdings.items()
        if abs(float(units)) > 1e-6
    }
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO portfolio_meta (key, value)
            VALUES ('cash_balance', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (float(cash_balance),),
        )
        conn.execute("DELETE FROM holdings")
        if normalized:
            conn.executemany(
                """
                INSERT INTO holdings (symbol, units, updated_at)
                VALUES (?, ?, ?)
                """,
                [
                    (symbol, units, timestamp)
                    for symbol, units in normalized.items()
                ],
            )
        conn.commit()


def record_price_history(snapshots: Iterable[Any]) -> None:
    timestamp = datetime.now(timezone.utc).timestamp()
    rows = [
        (
            snapshot.symbol,
            timestamp,
            float(snapshot.price),
        )
        for snapshot in snapshots
    ]
    if not rows:
        return
    cutoff = timestamp - 370 * 24 * 60 * 60
    with _connect() as conn:
        conn.executemany(
            """
            INSERT INTO price_history (symbol, timestamp, price)
            VALUES (?, ?, ?)
            """,
            rows,
        )
        conn.execute(
            "DELETE FROM price_history WHERE timestamp < ?",
            (cutoff,),
        )
        conn.commit()


def compute_price_returns(
    symbols: Sequence[str],
    current_prices: Mapping[str, float],
    horizons_days: Sequence[int],
) -> Dict[str, Dict[int, Optional[float]]]:
    now_ts = datetime.now(timezone.utc).timestamp()
    results: Dict[str, Dict[int, Optional[float]]] = {
        symbol: {h: None for h in horizons_days} for symbol in symbols
    }
    if not symbols:
        return results
    with _connect() as conn:
        for symbol in symbols:
            current_price = float(current_prices.get(symbol, 0.0))
            if current_price <= 0:
                continue
            for horizon in horizons_days:
                cutoff = now_ts - float(horizon) * 24.0 * 60.0 * 60.0
                row = conn.execute(
                    """
                    SELECT price
                    FROM price_history
                    WHERE symbol = ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (symbol, cutoff),
                ).fetchone()
                if row:
                    past_price = float(row["price"])
                    if past_price > 0:
                        results[symbol][horizon] = (current_price - past_price) / past_price
    return results


def reset_database() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM holdings")
        conn.execute("DELETE FROM price_history")
        conn.execute(
            """
            INSERT INTO portfolio_meta (key, value)
            VALUES ('cash_balance', 0.0)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """
        )
        conn.commit()
