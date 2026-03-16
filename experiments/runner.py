"""Experiment runner - evaluates all strategies across periods and bar margins."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from experiments.strategies import ALL_STRATEGIES, BaseStrategy, StrategyConfig
from optimizer import sortino_ratio, pnl_smoothness
from simulator.data_feed import Bar
from simulator.market_simulator import SimResult

log = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    strategy_name: str
    period_days: int
    bar_margin_bp: float
    return_pct: float
    sortino: float
    smoothness: float
    max_dd_pct: float
    trades: int
    round_trips: int
    win_rate: float
    fees: float
    avg_trade_pnl: float


@dataclass
class ExperimentSuite:
    results: list[ExperimentResult] = field(default_factory=list)

    def best_by_combined_score(self, top_n: int = 5) -> list[ExperimentResult]:
        scored = []
        for r in self.results:
            score = r.return_pct * 0.4 + r.sortino * 30 - r.max_dd_pct * 0.3 - r.smoothness * 100
            scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [r for _, r in scored[:top_n]]

    def best_by_sortino(self, top_n: int = 5) -> list[ExperimentResult]:
        return sorted(self.results, key=lambda r: -r.sortino)[:top_n]

    def best_by_smoothness(self, top_n: int = 5) -> list[ExperimentResult]:
        return sorted(self.results, key=lambda r: (r.smoothness, -r.return_pct))[:top_n]

    def to_table(self, results: list[ExperimentResult] | None = None) -> str:
        rows = results or self.results
        header = f"{'strategy':<25} {'period':>6} {'margin':>6} {'ret%':>8} {'sortino':>8} {'smooth':>8} {'dd%':>6} {'trades':>6} {'wr%':>6}"
        lines = [header, "-" * len(header)]
        for r in rows:
            lines.append(
                f"{r.strategy_name:<25} {r.period_days:>5}d {r.bar_margin_bp:>5.0f} "
                f"{r.return_pct:>7.2f}% {r.sortino:>8.3f} {r.smoothness:>8.5f} "
                f"{r.max_dd_pct:>5.1f}% {r.trades:>6} {r.win_rate*100:>5.1f}%"
            )
        return "\n".join(lines)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = []
        for r in self.results:
            data.append({
                "strategy": r.strategy_name,
                "period_days": r.period_days,
                "bar_margin_bp": r.bar_margin_bp,
                "return_pct": round(r.return_pct, 4),
                "sortino": round(r.sortino, 4),
                "smoothness": round(r.smoothness, 6),
                "max_dd_pct": round(r.max_dd_pct, 4),
                "trades": r.trades,
                "round_trips": r.round_trips,
                "win_rate": round(r.win_rate, 4),
                "fees": round(r.fees, 4),
                "avg_trade_pnl": round(r.avg_trade_pnl, 6),
            })
        path.write_text(json.dumps(data, indent=2))
        log.info("saved %d results to %s", len(data), path)


def run_experiments(
    bars: list[Bar],
    buy_prices: list[float],
    sell_prices: list[float],
    strategies: list[BaseStrategy] | None = None,
    periods: list[int] | None = None,
    bar_margins: list[float] | None = None,
    fee_rate: float = 0.001,
    initial_capital: float = 1000.0,
) -> ExperimentSuite:
    strategies = strategies or ALL_STRATEGIES
    periods = periods or [7, 30, 60]
    bar_margins = bar_margins or [5.0, 15.0]

    suite = ExperimentSuite()
    config = StrategyConfig(fee_rate=fee_rate, initial_capital=initial_capital)

    for strat in strategies:
        for period in periods:
            bars_needed = period * 24
            if len(bars) < bars_needed:
                continue
            window = bars[-bars_needed:]

            # align prices
            if len(buy_prices) == len(bars):
                offset = len(bars) - bars_needed
                bp_w = buy_prices[offset:]
                sp_w = sell_prices[offset:]
            elif len(buy_prices) >= bars_needed:
                bp_w = buy_prices[-bars_needed:]
                sp_w = sell_prices[-bars_needed:]
            else:
                bp_w = buy_prices
                sp_w = sell_prices

            for margin in bar_margins:
                config.bar_margin_bp = margin
                try:
                    result = strat.run(window, bp_w, sp_w, config)
                except Exception as e:
                    log.warning("strategy %s failed: %s", strat.name, e)
                    continue

                daily_rets = []
                for i in range(24, len(result.equity_curve), 24):
                    prev = result.equity_curve[i - 24]
                    if prev > 0:
                        daily_rets.append((result.equity_curve[i] - prev) / prev)

                suite.results.append(ExperimentResult(
                    strategy_name=strat.name,
                    period_days=period,
                    bar_margin_bp=margin,
                    return_pct=result.total_return_pct,
                    sortino=sortino_ratio(daily_rets),
                    smoothness=pnl_smoothness(result.equity_curve),
                    max_dd_pct=result.max_drawdown_pct,
                    trades=result.num_trades,
                    round_trips=result.num_round_trips,
                    win_rate=result.win_rate,
                    fees=result.total_fees,
                    avg_trade_pnl=result.avg_trade_pnl,
                ))

    log.info("ran %d experiments across %d strategies", len(suite.results), len(strategies))
    return suite
