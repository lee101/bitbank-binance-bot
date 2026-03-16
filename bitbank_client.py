from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import httpx

log = logging.getLogger(__name__)


@dataclass
class Signal:
    pair: str
    buy_price: float
    sell_price: float
    confidence: float
    signal_type: str
    pnl_7d_pct: float
    trades_7d: int
    win_rate_7d: float

    @property
    def binance_symbol(self) -> str:
        """BTC_USDT -> BTCUSDT"""
        return self.pair.replace("_", "")

    @property
    def spread_pct(self) -> float:
        if self.buy_price <= 0:
            return 0.0
        return (self.sell_price - self.buy_price) / self.buy_price


class BitbankClient:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=30)

    async def fetch_signals(self) -> list[Signal]:
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        url = f"{self.base_url}/api/trading-bot/public-signals-latest"
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        signals = []
        items = data if isinstance(data, list) else data.get("signals", [])
        for item in items:
            try:
                sig = Signal(
                    pair=item["currency_pair"],
                    buy_price=float(item["buy_price"]),
                    sell_price=float(item["sell_price"]),
                    confidence=float(item.get("confidence", 0)),
                    signal_type=item.get("signal_type", "hold"),
                    pnl_7d_pct=float(item.get("pnl_7d_pct", 0)),
                    trades_7d=int(item.get("trades_7d", 0)),
                    win_rate_7d=float(item.get("win_rate_7d", 0)),
                )
                signals.append(sig)
            except (KeyError, ValueError) as e:
                log.warning("skip malformed signal: %s", e)
        return signals

    async def fetch_coins(self) -> list[dict]:
        url = f"{self.base_url}/api/coins"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        resp = await self._client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", data if isinstance(data, list) else [])

    async def close(self):
        await self._client.aclose()
