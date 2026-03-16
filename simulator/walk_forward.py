"""Walk-forward validation simulator."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from optimizer import StrategyOptimizer, OptimizableParams, OptimizationResult, sortino_ratio, pnl_smoothness
from simulator.data_feed import Bar
from simulator.market_simulator import MarketSimulator, SimResult

log = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_result: SimResult | None = None
    test_result: SimResult | None = None
    optimized_params: OptimizableParams | None = None
    train_return_pct: float = 0.0
    test_return_pct: float = 0.0
    train_sortino: float = 0.0
    test_sortino: float = 0.0


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow] = field(default_factory=list)
    aggregate_return_pct: float = 0.0
    aggregate_sortino: float = 0.0
    consistency_ratio: float = 0.0  # % of windows where test > 0

    def summary(self) -> str:
        lines = [
            f"Walk-Forward: {len(self.windows)} windows",
            f"  aggregate return: {self.aggregate_return_pct:.2f}%",
            f"  aggregate sortino: {self.aggregate_sortino:.3f}",
            f"  consistency: {self.consistency_ratio:.1%}",
            "",
        ]
        for i, w in enumerate(self.windows):
            lines.append(
                f"  window {i}: train={w.train_return_pct:.2f}% test={w.test_return_pct:.2f}% "
                f"train_sort={w.train_sortino:.3f} test_sort={w.test_sortino:.3f}"
            )
        return "\n".join(lines)


class WalkForwardValidator:
    def __init__(
        self,
        bars: list[Bar],
        buy_prices: list[float],
        sell_prices: list[float],
        fee_rate: float = 0.001,
        initial_capital: float = 1000.0,
    ):
        self.bars = bars
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.fee_rate = fee_rate
        self.capital = initial_capital

    def run(
        self,
        train_hours: int = 30 * 24,
        test_hours: int = 7 * 24,
        step_hours: int = 7 * 24,
        opt_maxiter: int = 30,
        opt_popsize: int = 10,
    ) -> WalkForwardResult:
        n = len(self.bars)
        windows = []
        pos = 0

        while pos + train_hours + test_hours <= n:
            train_start = pos
            train_end = pos + train_hours
            test_start = train_end
            test_end = min(test_start + test_hours, n)

            # optimize on train
            train_bars = self.bars[train_start:train_end]
            train_bp = self._slice_prices(self.buy_prices, train_start, train_end)
            train_sp = self._slice_prices(self.sell_prices, train_start, train_end)

            optimizer = StrategyOptimizer(
                train_bars, train_bp, train_sp,
                fee_rate=self.fee_rate,
                initial_capital=self.capital,
                objective="combined",
            )
            opt_result = optimizer.optimize(maxiter=opt_maxiter, popsize=opt_popsize)
            params = opt_result.best_params

            # test on out-of-sample
            test_bars = self.bars[test_start:test_end]
            test_bp_raw = self._slice_prices(self.buy_prices, test_start, test_end)
            test_sp_raw = self._slice_prices(self.sell_prices, test_start, test_end)
            test_bp = [p * (1 - params.buy_offset_pct) for p in test_bp_raw]
            test_sp = [p * (1 + params.sell_offset_pct) for p in test_sp_raw]

            sim = MarketSimulator(self.fee_rate, params.bar_margin_bp, self.capital)
            test_result = sim.simulate_pair("WF_TEST", test_bars, test_bp, test_sp)

            train_rets = self._daily_returns(opt_result.sim_result.equity_curve if opt_result.sim_result else [])
            test_rets = self._daily_returns(test_result.equity_curve)

            w = WalkForwardWindow(
                train_start=train_start, train_end=train_end,
                test_start=test_start, test_end=test_end,
                train_result=opt_result.sim_result,
                test_result=test_result,
                optimized_params=params,
                train_return_pct=opt_result.sim_result.total_return_pct if opt_result.sim_result else 0,
                test_return_pct=test_result.total_return_pct,
                train_sortino=sortino_ratio(train_rets),
                test_sortino=sortino_ratio(test_rets),
            )
            windows.append(w)
            log.info("wf window %d: train=%.2f%% test=%.2f%%", len(windows), w.train_return_pct, w.test_return_pct)

            pos += step_hours

        # aggregate
        total_test_return = 1.0
        all_test_rets = []
        wins = 0
        for w in windows:
            total_test_return *= (1 + w.test_return_pct / 100)
            if w.test_result:
                all_test_rets.extend(self._daily_returns(w.test_result.equity_curve))
            if w.test_return_pct > 0:
                wins += 1

        return WalkForwardResult(
            windows=windows,
            aggregate_return_pct=(total_test_return - 1) * 100,
            aggregate_sortino=sortino_ratio(all_test_rets),
            consistency_ratio=wins / len(windows) if windows else 0,
        )

    def _slice_prices(self, prices: list[float], start: int, end: int) -> list[float]:
        if len(prices) == len(self.bars):
            return prices[start:end]
        # daily prices mapped to hourly
        daily_start = start // 24
        daily_end = (end + 23) // 24
        return prices[daily_start:daily_end]

    def _daily_returns(self, equity_curve: list[float]) -> list[float]:
        rets = []
        for i in range(24, len(equity_curve), 24):
            prev = equity_curve[i - 24]
            if prev > 0:
                rets.append((equity_curve[i] - prev) / prev)
        return rets
