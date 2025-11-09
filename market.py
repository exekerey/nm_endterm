from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import requests

BINANCE_REST_URL = "https://api.binance.com"
DEFAULT_SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")


class BinanceAPIError(RuntimeError):
    """Raised when the Binance REST API cannot fulfill a request."""


@dataclass(frozen=True)
class AssetSnapshot:
    symbol: str
    price: float
    expected_return: float
    risk: float
    volume: float
    timestamp: datetime


def _fallback_rows(symbols: Sequence[str]) -> List[AssetSnapshot]:
    sample = {
        "BTCUSDT": (108_000.0, 0.012, 0.05),
        "ETHUSDT": (3_800.0, 0.018, 0.07),
        "BNBUSDT": (1_050.0, 0.009, 0.04),
        "SOLUSDT": (190.0, 0.022, 0.09),
    }
    now = datetime.now(timezone.utc)
    rows = []
    for symbol in symbols:
        price, exp_ret, risk = sample.get(symbol, (100.0, 0.0, 0.05))
        rows.append(
            AssetSnapshot(
                symbol=symbol,
                price=price,
                expected_return=exp_ret,
                risk=risk,
                volume=0.0,
                timestamp=now,
            )
        )
    return rows


class BinanceClient:
    """Small wrapper around the Binance public API."""

    def __init__(
        self,
        *,
        base_url: str = BINANCE_REST_URL,
        session: Optional[requests.Session] = None,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session = session or requests.Session()
        self._timeout = timeout

    def _get(self, path: str, params: dict) -> dict:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:
            raise BinanceAPIError(f"failed to reach binance api: {exc}") from exc
        if response.status_code != 200:
            raise BinanceAPIError(
                f"binance api error ({response.status_code}): {response.text}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise BinanceAPIError("binance api returned invalid json") from exc

    def fetch_snapshot(self, symbols: Iterable[str]) -> List[AssetSnapshot]:
        rows: List[AssetSnapshot] = []
        for symbol in symbols:
            payload = self._get("/api/v3/ticker/24hr", params={"symbol": symbol})
            try:
                price = float(payload["lastPrice"])
                change_pct = float(payload["priceChangePercent"]) / 100.0
                high_price = float(payload["highPrice"])
                low_price = float(payload["lowPrice"])
                volume = float(payload.get("volume", 0.0))
                close_time_ms = int(payload.get("closeTime", 0))
            except (KeyError, TypeError, ValueError) as exc:
                raise BinanceAPIError(
                    f"malformed payload received for {symbol}: {payload}"
                ) from exc
            risk = 0.0
            if price > 0.0:
                span = max(high_price - low_price, 0.0)
                risk = span / price
            rows.append(
                AssetSnapshot(
                    symbol=symbol,
                    price=price,
                    expected_return=change_pct,
                    risk=risk,
                    volume=volume,
                    timestamp=datetime.fromtimestamp(
                        close_time_ms / 1000.0, tz=timezone.utc
                    ),
                )
            )
        return rows


def _parse_custom_row(row: dict, seen_symbols: set[str], now: datetime) -> AssetSnapshot:
    symbol = str(row["symbol"]).upper()
    if symbol in seen_symbols:
        raise ValueError(f"duplicate symbol {symbol} in manual data")
    seen_symbols.add(symbol)
    price = float(row["price"])
    expected = float(row.get("expected_return", row.get("return", 0.0)))
    risk = float(row.get("risk", abs(expected)))
    volume = float(row.get("volume", 0.0))
    return AssetSnapshot(
        symbol=symbol,
        price=price,
        expected_return=expected,
        risk=max(risk, 0.0),
        volume=max(volume, 0.0),
        timestamp=now,
    )


def _load_manual_data(path: Path) -> List[AssetSnapshot]:
    now = datetime.now(timezone.utc)
    seen_symbols: set[str] = set()
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text())
        if not isinstance(payload, list):
            raise ValueError("manual JSON data must be a list of objects")
        return [_parse_custom_row(row, seen_symbols, now) for row in payload]
    if path.suffix.lower() == ".csv":
        rows = []
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                rows.append(_parse_custom_row(row, seen_symbols, now))
        return rows
    raise ValueError("manual data file must be .csv or .json")


class MarketDataSource:
    def __init__(
        self,
        symbols: Sequence[str] = DEFAULT_SYMBOLS,
        *,
        data_file: Optional[Path] = None,
        enable_fallback: bool = True,
        client: Optional[BinanceClient] = None,
    ) -> None:
        self._symbols = tuple(symbols) if symbols else DEFAULT_SYMBOLS
        self._data_file = Path(data_file).expanduser() if data_file else None
        self._enable_fallback = enable_fallback
        self._client = client or BinanceClient()

    def fetch(self) -> List[AssetSnapshot]:
        if self._data_file:
            return _load_manual_data(self._data_file)
        try:
            return self._client.fetch_snapshot(self._symbols)
        except BinanceAPIError:
            if not self._enable_fallback:
                raise
        return _fallback_rows(self._symbols)
