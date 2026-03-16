from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from binance.client import Client as BinanceSDK
from binance.exceptions import BinanceAPIException

log = logging.getLogger(__name__)

BINANCE_US_URL = "https://api.binance.us"


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_time: float = 0.0

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class OrderResult:
    symbol: str
    side: str
    qty: float
    price: float
    order_id: str = ""
    filled: bool = False
    error: str = ""


class BinanceClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.binance.com", dry_run: bool = True):
        self.dry_run = dry_run
        self.base_url = base_url
        tld = "us" if "binance.us" in base_url else "com"
        self._client = BinanceSDK(api_key, api_secret, tld=tld)
        self._symbol_info: dict[str, dict] = {}

    def _load_symbol_info(self, symbol: str) -> Optional[dict]:
        if symbol in self._symbol_info:
            return self._symbol_info[symbol]
        try:
            info = self._client.get_symbol_info(symbol)
            if info:
                self._symbol_info[symbol] = info
            return info
        except BinanceAPIException as e:
            log.warning("symbol info failed %s: %s", symbol, e)
            return None

    def get_price(self, symbol: str) -> Optional[float]:
        try:
            tick = self._client.get_symbol_ticker(symbol=symbol)
            return float(tick["price"])
        except BinanceAPIException as e:
            log.warning("price fetch failed %s: %s", symbol, e)
            return None

    def get_balances(self) -> dict[str, float]:
        try:
            account = self._client.get_account()
            return {
                b["asset"]: float(b["free"])
                for b in account["balances"]
                if float(b["free"]) > 0
            }
        except BinanceAPIException as e:
            log.error("balance fetch failed: %s", e)
            return {}

    def _step_size(self, symbol: str) -> float:
        info = self._load_symbol_info(symbol)
        if not info:
            return 0.00001
        for f in info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                return float(f["stepSize"])
        return 0.00001

    def _round_qty(self, symbol: str, qty: float) -> float:
        step = self._step_size(symbol)
        if step <= 0:
            return qty
        precision = max(0, len(str(step).rstrip('0').split('.')[-1]))
        return round(qty - (qty % step), precision)

    def place_limit_buy(self, symbol: str, qty: float, price: float) -> OrderResult:
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            return OrderResult(symbol=symbol, side="BUY", qty=0, price=price, error="qty too small")
        if self.dry_run:
            log.info("DRY BUY %s qty=%.6f @ %.4f", symbol, qty, price)
            return OrderResult(symbol=symbol, side="BUY", qty=qty, price=price, filled=True, order_id="dry")
        try:
            order = self._client.order_limit_buy(
                symbol=symbol,
                quantity=f"{qty:.8f}".rstrip('0').rstrip('.'),
                price=f"{price:.8f}".rstrip('0').rstrip('.'),
                timeInForce="GTC",
            )
            return OrderResult(
                symbol=symbol, side="BUY", qty=qty, price=price,
                order_id=str(order["orderId"]), filled=order["status"] == "FILLED"
            )
        except BinanceAPIException as e:
            log.error("buy order failed %s: %s", symbol, e)
            return OrderResult(symbol=symbol, side="BUY", qty=qty, price=price, error=str(e))

    def place_limit_sell(self, symbol: str, qty: float, price: float) -> OrderResult:
        qty = self._round_qty(symbol, qty)
        if qty <= 0:
            return OrderResult(symbol=symbol, side="SELL", qty=0, price=price, error="qty too small")
        if self.dry_run:
            log.info("DRY SELL %s qty=%.6f @ %.4f", symbol, qty, price)
            return OrderResult(symbol=symbol, side="SELL", qty=qty, price=price, filled=True, order_id="dry")
        try:
            order = self._client.order_limit_sell(
                symbol=symbol,
                quantity=f"{qty:.8f}".rstrip('0').rstrip('.'),
                price=f"{price:.8f}".rstrip('0').rstrip('.'),
                timeInForce="GTC",
            )
            return OrderResult(
                symbol=symbol, side="SELL", qty=qty, price=price,
                order_id=str(order["orderId"]), filled=order["status"] == "FILLED"
            )
        except BinanceAPIException as e:
            log.error("sell order failed %s: %s", symbol, e)
            return OrderResult(symbol=symbol, side="SELL", qty=qty, price=price, error=str(e))

    def cancel_open_orders(self, symbol: str) -> int:
        if self.dry_run:
            return 0
        try:
            orders = self._client.get_open_orders(symbol=symbol)
            for o in orders:
                self._client.cancel_order(symbol=symbol, orderId=o["orderId"])
            return len(orders)
        except BinanceAPIException as e:
            log.error("cancel orders failed %s: %s", symbol, e)
            return 0

    def get_klines(self, symbol: str, interval: str = "1h", limit: int = 168) -> list[dict]:
        try:
            raw = self._client.get_klines(symbol=symbol, interval=interval, limit=limit)
            return [
                {
                    "open_time": k[0],
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": k[6],
                }
                for k in raw
            ]
        except BinanceAPIException as e:
            log.error("klines failed %s: %s", symbol, e)
            return []
