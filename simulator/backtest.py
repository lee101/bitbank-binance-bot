"""Backtester that runs the work-stealing strategy over historical data.

Supports 7d, 30d, 60d periods with hourly granularity and multiple bar margins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import Config
from simulator.data_feed import Bar, DataFeed
from simulator.market_simulator import MarketSimulator, SimResult

log = logging.getLogger(__name__)

DEFAULT_BAR_MARGINS = [5.0, 15.0]  # basis points
DEFAULT_PERIODS = [7, 30, 60]


@dataclass
class BacktestSuite:
    results: dict[str, list[SimResult]] = field(default_factory=dict)  # "sym_period_margin" -> results

    def summary(self) -> list[dict]:
        rows = []
        for key, results in self.results.items():
            for r in results:
                rows.append({
                    "key": key,
                    "symbol": r.symbol,
                    "period_days": r.period_days,
                    "bar_margin_bp": r.bar_margin_bp,
                    "return_pct": round(r.total_return_pct, 4),
                    "net_pnl": round(r.net_pnl, 4),
                    "fees": round(r.total_fees, 4),
                    "trades": r.num_trades,
                    "round_trips": r.num_round_trips,
                    "win_rate": round(r.win_rate, 4),
                    "max_dd_pct": round(r.max_drawdown_pct, 4),
                })
        return rows

    def is_strategy_viable(self, min_win_rate: float = 0.45, max_drawdown: float = 20.0) -> bool:
        for results in self.results.values():
            for r in results:
                if r.win_rate < min_win_rate:
                    return False
                if r.max_drawdown_pct > max_drawdown:
                    return False
                if r.total_return_pct < 0 and r.period_days >= 30:
                    return False
        return True


class Backtester:
    def __init__(self, config: Config | None = None, data_feed: DataFeed | None = None):
        self.config = config or Config()
        self.feed = data_feed or DataFeed()

    def run(
        self,
        symbol: str,
        buy_prices: list[float],
        sell_prices: list[float],
        bars: list[Bar] | None = None,
        periods: list[int] | None = None,
        bar_margins: list[float] | None = None,
        granularity: str = "hourly",
    ) -> BacktestSuite:
        if bars is None:
            loader = self.feed.load_hourly if granularity == "hourly" else self.feed.load_daily
            bars = loader(symbol)

        periods = periods or DEFAULT_PERIODS
        bar_margins = bar_margins or DEFAULT_BAR_MARGINS
        suite = BacktestSuite()

        for period in periods:
            bars_needed = period * 24 if granularity == "hourly" else period
            if len(bars) < bars_needed:
                log.warning("%s: only %d bars, need %d for %dd", symbol, len(bars), bars_needed, period)
                continue
            window = bars[-bars_needed:]

            # align buy/sell prices to window
            bp_window, sp_window = self._align_prices(buy_prices, sell_prices, len(window), len(bars), bars_needed)

            for margin in bar_margins:
                sim = MarketSimulator(
                    fee_rate=self.config.fee_rate,
                    bar_margin_bp=margin,
                    initial_capital=self.config.capital,
                )
                result = sim.simulate_pair(symbol, window, bp_window, sp_window)
                key = f"{symbol}_{period}d_{margin}bp"
                suite.results.setdefault(key, []).append(result)
                log.info("%s: ret=%.2f%% wr=%.1f%% dd=%.1f%% trades=%d",
                         key, result.total_return_pct, result.win_rate * 100,
                         result.max_drawdown_pct, result.num_trades)

        return suite

    def _align_prices(
        self,
        buy_prices: list[float],
        sell_prices: list[float],
        window_len: int,
        total_bars: int,
        bars_needed: int,
    ) -> tuple[list[float], list[float]]:
        if len(buy_prices) == total_bars:
            offset = total_bars - bars_needed
            return buy_prices[offset:], sell_prices[offset:]
        elif len(buy_prices) >= window_len:
            return buy_prices[-window_len:], sell_prices[-window_len:]
        else:
            # repeat last price to fill
            bp = buy_prices + [buy_prices[-1]] * (window_len - len(buy_prices)) if buy_prices else [0.0] * window_len
            sp = sell_prices + [sell_prices[-1]] * (window_len - len(sell_prices)) if sell_prices else [0.0] * window_len
            return bp[:window_len], sp[:window_len]

    def run_multi(
        self,
        symbols: list[str],
        buy_prices_by_symbol: dict[str, list[float]],
        sell_prices_by_symbol: dict[str, list[float]],
        **kwargs,
    ) -> dict[str, BacktestSuite]:
        results = {}
        for sym in symbols:
            bp = buy_prices_by_symbol.get(sym, [])
            sp = sell_prices_by_symbol.get(sym, [])
            if not bp or not sp:
                continue
            try:
                results[sym] = self.run(sym, bp, sp, **kwargs)
            except FileNotFoundError as e:
                log.warning("skip %s: %s", sym, e)
        return results
