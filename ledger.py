
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TradeRecord:
    timestamp: datetime
    iteration: int
    symbol: str
    weight: float
    allocation: float
    quantity: float
    price: float
    expected_return: float


class TradeLedger:
    FIELDNAMES = [
        "timestamp",
        "iteration",
        "symbol",
        "weight",
        "allocation",
        "quantity",
        "price",
        "expected_return",
    ]

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        if self._path.suffix.lower() != ".csv":
            raise ValueError("ledger path must end with .csv")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            with self._path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append(self, record: TradeRecord) -> None:
        with self._path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writerow(
                {
                    "timestamp": record.timestamp.isoformat(),
                    "iteration": record.iteration,
                    "symbol": record.symbol,
                    "weight": round(record.weight, 6),
                    "allocation": round(record.allocation, 2),
                    "quantity": round(record.quantity, 8),
                    "price": round(record.price, 6),
                    "expected_return": round(record.expected_return, 6),
                }
            )

    def extend(self, records: Iterable[TradeRecord]) -> None:
        for record in records:
            self.append(record)
