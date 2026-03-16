"""Market simulator with hourly granularity and realistic execution modeling.

Simulates the work-stealing strategy over historical data with:
- 5bp/15bp bar margin buffers for execution realism
- Fee modeling (configurable per side)
- Intraday bounce detection (multiple executions per day)
- Hourly simulation even for daily trading signals
"""
import logging
from dataclasses import dataclass, field

from simulator.data_feed import Bar

log = logging.getLogger(__name__)


@dataclass
class SimTrade:
    symbol: str
    side: str
    price: float
    qty: float
    bar_idx: int
    timestamp: float
    fees: float = 0.0
    pnl_pct: float = 0.0
    trade_type: str = ""  # round_trip, buy_pending, sell_close, no_trade


@dataclass
class SimResult:
    symbol: str
    period_days: int
    bar_margin_bp: float
    total_return_pct: float
    gross_pnl: float
    net_pnl: float
    total_fees: float
    num_trades: int
    num_round_trips: int
    wins: int
    losses: int
    max_drawdown_pct: float
    trades: list[SimTrade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total else 0.0

    @property
    def avg_trade_pnl(self) -> float:
        if not self.num_round_trips:
            return 0.0
        return self.net_pnl / self.num_round_trips


class MarketSimulator:
    def __init__(
        self,
        fee_rate: float = 0.001,
        bar_margin_bp: float = 5.0,
        initial_capital: float = 1000.0,
    ):
        self.fee_rate = fee_rate
        self.bar_margin_pct = bar_margin_bp / 10000.0
        self.initial_capital = initial_capital

    def can_execute_at_bar(self, target_price: float, bar: Bar, side: str) -> bool:
        """Check if target price would execute within this bar, accounting for bar margin.

        For buys: price must be reachable from bar's low side (low <= target + margin)
        For sells: price must be reachable from bar's high side (high >= target - margin)
        """
        margin = target_price * self.bar_margin_pct
        if side == "buy":
            return bar.low <= target_price + margin
        else:
            return bar.high >= target_price - margin

    def simulate_pair(
        self,
        symbol: str,
        bars: list[Bar],
        buy_prices: list[float],
        sell_prices: list[float],
    ) -> SimResult:
        """Simulate trading a single pair over hourly bars.

        buy_prices/sell_prices: one per day. For hourly bars, each daily price
        applies to 24 consecutive bars (or whatever grouping is used).
        If len matches bars, treat as per-bar prices.
        """
        assert len(buy_prices) == len(sell_prices), "buy/sell price arrays must match"

        per_bar = len(buy_prices) == len(bars)
        cash = self.initial_capital
        position = 0.0
        entry_price = 0.0
        trades: list[SimTrade] = []
        equity_curve = [cash]
        peak_equity = cash
        max_drawdown = 0.0
        wins = 0
        losses = 0
        round_trips = 0
        total_fees = 0.0
        gross_pnl = 0.0

        for i, bar in enumerate(bars):
            price_idx = i if per_bar else min(i // 24, len(buy_prices) - 1)
            bp = buy_prices[price_idx]
            sp = sell_prices[price_idx]

            if bp <= 0 or sp <= 0 or sp <= bp:
                equity = cash + position * bar.close
                equity_curve.append(equity)
                continue

            # check sell first (if holding)
            if position > 0 and self.can_execute_at_bar(sp, bar, "sell"):
                fee = position * sp * self.fee_rate
                proceeds = position * sp - fee
                pnl = (sp - entry_price) / entry_price - 2 * self.fee_rate
                gross_pnl += (sp - entry_price) * position
                total_fees += fee
                cash += proceeds
                trades.append(SimTrade(
                    symbol=symbol, side="sell", price=sp, qty=position,
                    bar_idx=i, timestamp=bar.timestamp, fees=fee,
                    pnl_pct=pnl, trade_type="sell_close"
                ))
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                round_trips += 1
                position = 0.0
                entry_price = 0.0

            # check buy (if not holding)
            if position == 0 and cash > 10 and self.can_execute_at_bar(bp, bar, "buy"):
                alloc = cash
                fee = alloc * self.fee_rate
                qty = (alloc - fee) / bp
                total_fees += fee
                cash -= alloc
                position = qty
                entry_price = bp
                trades.append(SimTrade(
                    symbol=symbol, side="buy", price=bp, qty=qty,
                    bar_idx=i, timestamp=bar.timestamp, fees=fee,
                    trade_type="buy_pending"
                ))

                # check same-bar round trip (price bounced through both levels)
                if self.can_execute_at_bar(sp, bar, "sell"):
                    sell_fee = position * sp * self.fee_rate
                    proceeds = position * sp - sell_fee
                    pnl = (sp - entry_price) / entry_price - 2 * self.fee_rate
                    gross_pnl += (sp - entry_price) * position
                    total_fees += sell_fee
                    cash += proceeds
                    trades.append(SimTrade(
                        symbol=symbol, side="sell", price=sp, qty=position,
                        bar_idx=i, timestamp=bar.timestamp, fees=sell_fee,
                        pnl_pct=pnl, trade_type="round_trip"
                    ))
                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1
                    round_trips += 1
                    position = 0.0
                    entry_price = 0.0

            equity = cash + position * bar.close
            equity_curve.append(equity)
            if equity > peak_equity:
                peak_equity = equity
            dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
            if dd > max_drawdown:
                max_drawdown = dd

        # close any remaining position at last bar
        if position > 0 and bars:
            final_price = bars[-1].close
            fee = position * final_price * self.fee_rate
            proceeds = position * final_price - fee
            pnl = (final_price - entry_price) / entry_price - 2 * self.fee_rate
            gross_pnl += (final_price - entry_price) * position
            total_fees += fee
            cash += proceeds
            trades.append(SimTrade(
                symbol=symbol, side="sell", price=final_price, qty=position,
                bar_idx=len(bars) - 1, timestamp=bars[-1].timestamp, fees=fee,
                pnl_pct=pnl, trade_type="forced_close"
            ))
            if pnl > 0:
                wins += 1
            else:
                losses += 1
            round_trips += 1

        final_equity = cash
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100
        net_pnl = final_equity - self.initial_capital

        return SimResult(
            symbol=symbol,
            period_days=len(bars) // 24 if len(bars) >= 24 else len(bars),
            bar_margin_bp=self.bar_margin_pct * 10000,
            total_return_pct=total_return,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_fees=total_fees,
            num_trades=len(trades),
            num_round_trips=round_trips,
            wins=wins,
            losses=losses,
            max_drawdown_pct=max_drawdown * 100,
            trades=trades,
            equity_curve=equity_curve,
        )

    def simulate_multi_pair(
        self,
        symbols: list[str],
        bars_by_symbol: dict[str, list[Bar]],
        buy_prices_by_symbol: dict[str, list[float]],
        sell_prices_by_symbol: dict[str, list[float]],
    ) -> dict[str, SimResult]:
        results = {}
        for sym in symbols:
            if sym not in bars_by_symbol:
                continue
            results[sym] = self.simulate_pair(
                sym,
                bars_by_symbol[sym],
                buy_prices_by_symbol.get(sym, []),
                sell_prices_by_symbol.get(sym, []),
            )
        return results
