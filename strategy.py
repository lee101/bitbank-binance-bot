import logging
import time
from dataclasses import dataclass, field

from bitbank_client import Signal
from config import Config

log = logging.getLogger(__name__)


@dataclass
class ActiveOrder:
    symbol: str
    side: str
    limit_price: float
    current_price: float
    qty: float
    notional: float
    forecasted_pnl: float
    entry_time: float = 0.0

    @property
    def distance_pct(self) -> float:
        if self.current_price <= 0:
            return 999.0
        return abs(self.current_price - self.limit_price) / self.current_price


@dataclass
class TradeRecord:
    symbol: str
    side: str
    price: float
    qty: float
    timestamp: float
    pnl_pct: float = 0.0
    fees: float = 0.0


@dataclass
class WorkStealingStrategy:
    config: Config
    positions: dict[str, float] = field(default_factory=dict)  # symbol -> qty
    entry_prices: dict[str, float] = field(default_factory=dict)
    active_orders: dict[str, ActiveOrder] = field(default_factory=dict)
    capital_used: float = 0.0
    cash: float = 0.0
    trade_log: list[TradeRecord] = field(default_factory=list)
    steal_history: list[tuple[float, str, str]] = field(default_factory=list)  # (time, stolen, stealer)
    last_entry_time: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.cash = self.config.capital

    @property
    def available_capital(self) -> float:
        return self.cash

    def filter_tradable(self, signals: list[Signal]) -> list[Signal]:
        return [
            s for s in signals
            if s.pnl_7d_pct > self.config.min_pnl_7d_pct
            and s.buy_price > 0
            and s.sell_price > s.buy_price
            and s.spread_pct > 2 * self.config.fee_rate  # spread must exceed round-trip fees
        ]

    def evaluate_signals(self, signals: list[Signal], current_prices: dict[str, float]) -> list[dict]:
        tradable = self.filter_tradable(signals)
        tradable.sort(key=lambda s: s.pnl_7d_pct, reverse=True)
        actions = []
        now = time.time()

        for sig in tradable:
            symbol = sig.binance_symbol
            price = current_prices.get(symbol)
            if price is None:
                continue

            # check cooldown
            last = self.last_entry_time.get(symbol, 0)
            if now - last < self.config.cooldown_seconds:
                continue

            # check if already holding
            if symbol in self.positions and self.positions[symbol] > 0:
                # check sell opportunity
                sell_distance = abs(price - sig.sell_price) / price if price > 0 else 999
                if sell_distance <= self.config.entry_tolerance_pct or price >= sig.sell_price:
                    actions.append({
                        "action": "sell",
                        "symbol": symbol,
                        "price": sig.sell_price,
                        "qty": self.positions[symbol],
                        "signal": sig,
                        "reason": f"within {sell_distance*10000:.1f}bp of sell target",
                    })
                continue

            # check buy opportunity - price must be near or below buy target
            buy_distance = (price - sig.buy_price) / price if price > 0 else 999
            if buy_distance > self.config.entry_tolerance_pct:
                continue

            # capital allocation
            alloc = self.config.capital * self.config.max_position_pct
            if alloc > self.available_capital:
                # attempt work stealing
                stolen = self._attempt_steal(symbol, sig.pnl_7d_pct, now)
                if stolen:
                    alloc = min(alloc, self.available_capital)
                else:
                    continue

            if alloc < 10:  # minimum $10
                continue

            qty = alloc / (sig.buy_price * (1 + self.config.fee_rate))
            actions.append({
                "action": "buy",
                "symbol": symbol,
                "price": sig.buy_price,
                "qty": qty,
                "notional": alloc,
                "signal": sig,
                "reason": f"within {buy_distance*10000:.1f}bp of buy target, 7d pnl={sig.pnl_7d_pct:.2f}%",
            })

        return actions

    def _attempt_steal(self, stealer_symbol: str, stealer_pnl: float, now: float) -> bool:
        if not self.active_orders:
            return False

        # sort by distance (furthest first), then lowest pnl
        candidates = sorted(
            self.active_orders.values(),
            key=lambda o: (-o.distance_pct, o.forecasted_pnl)
        )

        for candidate in candidates:
            # protect orders close to execution
            if candidate.distance_pct <= self.config.protection_bp / 10000:
                continue
            # don't steal from better performers
            if candidate.forecasted_pnl >= stealer_pnl:
                continue
            # check fight detection (max 5 steals in 600s)
            recent_steals = sum(1 for t, _, _ in self.steal_history if now - t < 600)
            if recent_steals >= 5:
                continue

            # steal
            self.cash += candidate.notional
            self.capital_used -= candidate.notional
            self.steal_history.append((now, candidate.symbol, stealer_symbol))
            del self.active_orders[candidate.symbol]
            log.info("STEAL %s from %s (dist=%.1fbp pnl=%.2f%%)",
                     stealer_symbol, candidate.symbol,
                     candidate.distance_pct * 10000, candidate.forecasted_pnl)
            return True
        return False

    def execute_buy(self, symbol: str, qty: float, price: float, pnl_7d: float):
        notional = qty * price * (1 + self.config.fee_rate)
        fees = qty * price * self.config.fee_rate
        self.positions[symbol] = self.positions.get(symbol, 0) + qty
        self.entry_prices[symbol] = price
        self.cash -= notional
        self.capital_used += notional
        self.last_entry_time[symbol] = time.time()
        self.active_orders[symbol] = ActiveOrder(
            symbol=symbol, side="BUY", limit_price=price,
            current_price=price, qty=qty, notional=notional,
            forecasted_pnl=pnl_7d, entry_time=time.time()
        )
        self.trade_log.append(TradeRecord(
            symbol=symbol, side="BUY", price=price,
            qty=qty, timestamp=time.time(), fees=fees
        ))

    def execute_sell(self, symbol: str, qty: float, price: float):
        fees = qty * price * self.config.fee_rate
        proceeds = qty * price * (1 - self.config.fee_rate)
        entry = self.entry_prices.get(symbol, price)
        pnl_pct = (price - entry) / entry - 2 * self.config.fee_rate
        self.cash += proceeds
        if symbol in self.positions:
            self.positions[symbol] -= qty
            if self.positions[symbol] <= 1e-10:
                del self.positions[symbol]
                del self.entry_prices[symbol]
        if symbol in self.active_orders:
            self.capital_used -= self.active_orders[symbol].notional
            del self.active_orders[symbol]
        self.trade_log.append(TradeRecord(
            symbol=symbol, side="SELL", price=price,
            qty=qty, timestamp=time.time(), pnl_pct=pnl_pct, fees=fees
        ))

    def portfolio_value(self, current_prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, qty in self.positions.items():
            price = current_prices.get(symbol, self.entry_prices.get(symbol, 0))
            value += qty * price
        return value

    def summary(self, current_prices: dict[str, float]) -> dict:
        pv = self.portfolio_value(current_prices)
        total_pnl = (pv - self.config.capital) / self.config.capital * 100
        sells = [t for t in self.trade_log if t.side == "SELL"]
        wins = [t for t in sells if t.pnl_pct > 0]
        return {
            "portfolio_value": pv,
            "total_pnl_pct": total_pnl,
            "cash": self.cash,
            "positions": len(self.positions),
            "total_trades": len(self.trade_log),
            "sells": len(sells),
            "wins": len(wins),
            "win_rate": len(wins) / len(sells) if sells else 0,
            "total_fees": sum(t.fees for t in self.trade_log),
        }
