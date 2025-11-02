from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import requests


BINANCE_REST_URL = "https://api.binance.com"
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
]


class BinanceAPIError(Exception):
    """Raised when the Binance REST API returns an error."""


@dataclass
class AssetSnapshot:
    symbol: str
    price: float
    price_change_percent: float
    high_price: float
    low_price: float
    volume: float
    event_time: datetime
    feedback_z: float
    return_7d: Optional[float] = None
    return_30d: Optional[float] = None
    return_365d: Optional[float] = None

    @property
    def expected_return(self) -> float:
        return self.feedback_z

    @property
    def risk_score(self) -> float:
        if self.price <= 0:
            return 0.0
        span = self.high_price - self.low_price
        return max(span, 0.0) / self.price


def _sample_snapshot() -> List[AssetSnapshot]:
    sample_rows: List[Dict[str, float]] = [
        {
            "symbol": "BTCUSDT",
            "price": 108421.34,
            "price_change_percent": -4.485,
            "high_price": 111280.67,
            "low_price": 105562.01,
            "volume": 31250.0,
            "return_7d": -0.082,
            "return_30d": 0.014,
            "return_365d": 0.612,
        },
        {
            "symbol": "ETHUSDT",
            "price": 3815.92,
            "price_change_percent": -5.191,
            "high_price": 3932.90,
            "low_price": 3698.94,
            "volume": 265430.0,
            "return_7d": -0.096,
            "return_30d": 0.032,
            "return_365d": 0.702,
        },
        {
            "symbol": "BNBUSDT",
            "price": 1098.33,
            "price_change_percent": -1.729,
            "high_price": 1121.81,
            "low_price": 1074.85,
            "volume": 78542.0,
            "return_7d": -0.035,
            "return_30d": 0.027,
            "return_365d": 0.488,
        },
        {
            "symbol": "SOLUSDT",
            "price": 187.62,
            "price_change_percent": -6.842,
            "high_price": 195.23,
            "low_price": 180.01,
            "volume": 904210.0,
            "return_7d": -0.124,
            "return_30d": 0.041,
            "return_365d": 0.834,
        },
    ]
    return [
        AssetSnapshot(
            symbol=row["symbol"],
            price=row["price"],
            price_change_percent=row["price_change_percent"],
            high_price=row["high_price"],
            low_price=row["low_price"],
            volume=row["volume"],
            event_time=datetime(2025, 10, 30, 13, 25, tzinfo=timezone.utc),
            feedback_z=row["price_change_percent"] / 100.0,
            return_7d=row.get("return_7d"),
            return_30d=row.get("return_30d"),
            return_365d=row.get("return_365d"),
        )
        for row in sample_rows
    ]


class BinanceClient:
    """Lightweight wrapper around the Binance public REST API."""

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

    def _request(self, path: str, params: Dict[str, str]) -> Dict[str, str]:
        url = f"{self._base_url}{path}"
        try:
            response = self._session.get(url, params=params, timeout=self._timeout)
        except requests.RequestException as exc:  # pragma: no cover - network issue
            raise BinanceAPIError(f"failed to reach binance api: {exc}") from exc

        if response.status_code != 200:
            raise BinanceAPIError(
                f"binance api error ({response.status_code}): {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            raise BinanceAPIError("invalid json returned by binance api") from exc

        if isinstance(payload, dict) and payload.get("code"):
            raise BinanceAPIError(
                f"binance api returned error code {payload.get('code')}: "
                f"{payload.get('msg')}"
            )
        return payload

    def fetch_snapshot(
        self, symbols: Iterable[str]
    ) -> List[AssetSnapshot]:
        snapshots: List[AssetSnapshot] = []
        for symbol in symbols:
            payload = self._request("/api/v3/ticker/24hr", params={"symbol": symbol})
            try:
                price = float(payload["lastPrice"])
                price_change_percent = float(payload["priceChangePercent"])
                high_price = float(payload["highPrice"])
                low_price = float(payload["lowPrice"])
                volume = float(payload.get("volume", 0.0))
                event_time = datetime.fromtimestamp(
                    int(payload.get("closeTime", 0)) / 1000.0, tz=timezone.utc
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise BinanceAPIError(
                    f"malformed payload received for {symbol}: {payload}"
                ) from exc

            snapshots.append(
                AssetSnapshot(
                    symbol=symbol,
                    price=price,
                    price_change_percent=price_change_percent,
                    high_price=high_price,
                    low_price=low_price,
                    volume=volume,
                    event_time=event_time,
                    feedback_z=price_change_percent / 100.0,
                )
            )
        return snapshots


def get_market_snapshot(
    symbols: Optional[Iterable[str]] = None,
    *,
    client: Optional[BinanceClient] = None,
    enable_fallback: bool = True,
) -> List[AssetSnapshot]:
    symbols = list(symbols or DEFAULT_SYMBOLS)
    api_client = client or BinanceClient()
    try:
        return api_client.fetch_snapshot(symbols)
    except BinanceAPIError as exc:
        if not enable_fallback:
            raise
        logging.getLogger(__name__).warning(
            "Binance API unavailable; falling back to sample snapshot (%s)", exc
        )
        return _sample_snapshot()
