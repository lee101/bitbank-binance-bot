"""Scipy-based strategy optimizer using differential evolution."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from config import Config
from simulator.data_feed import Bar
from simulator.market_simulator import MarketSimulator, SimResult

log = logging.getLogger(__name__)


@dataclass
class OptimizableParams:
    entry_tolerance_bp: float = 5.0
    buy_offset_pct: float = 0.002  # shift buy price down by this %
    sell_offset_pct: float = 0.003  # shift sell price up by this %
    position_size_pct: float = 0.25
    bar_margin_bp: float = 5.0
    take_profit_pct: float = 0.005  # min spread to take a trade
    stop_loss_pct: float = 0.02  # exit if loss exceeds this

    def to_array(self) -> np.ndarray:
        return np.array([
            self.entry_tolerance_bp,
            self.buy_offset_pct,
            self.sell_offset_pct,
            self.position_size_pct,
            self.bar_margin_bp,
            self.take_profit_pct,
            self.stop_loss_pct,
        ])

    @classmethod
    def from_array(cls, x: np.ndarray) -> "OptimizableParams":
        return cls(
            entry_tolerance_bp=x[0],
            buy_offset_pct=x[1],
            sell_offset_pct=x[2],
            position_size_pct=x[3],
            bar_margin_bp=x[4],
            take_profit_pct=x[5],
            stop_loss_pct=x[6],
        )

    @classmethod
    def bounds(cls) -> list[tuple[float, float]]:
        return [
            (1.0, 50.0),     # entry_tolerance_bp
            (0.0005, 0.02),  # buy_offset_pct
            (0.001, 0.03),   # sell_offset_pct
            (0.05, 0.5),     # position_size_pct
            (1.0, 30.0),     # bar_margin_bp
            (0.001, 0.02),   # take_profit_pct
            (0.005, 0.1),    # stop_loss_pct
        ]


def sortino_ratio(returns: list[float], risk_free: float = 0.0) -> float:
    if not returns:
        return 0.0
    arr = np.array(returns)
    excess = arr - risk_free
    mean_excess = np.mean(excess)
    downside = arr[arr < risk_free]
    if len(downside) < 2:
        return mean_excess * 100 if mean_excess > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return mean_excess * 100 if mean_excess > 0 else 0.0
    return float(mean_excess / downside_std)


def pnl_smoothness(equity_curve: list[float]) -> float:
    """Lower is smoother. Measures std of daily returns."""
    if len(equity_curve) < 2:
        return 0.0
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i - 1] > 0:
            returns.append((equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1])
    if not returns:
        return 0.0
    return float(np.std(returns))


@dataclass
class OptimizationResult:
    best_params: OptimizableParams
    best_score: float
    sim_result: SimResult | None = None
    sortino: float = 0.0
    smoothness: float = 0.0
    scipy_result: OptimizeResult | None = None
    all_evaluations: list[dict] = field(default_factory=list)


class StrategyOptimizer:
    def __init__(
        self,
        bars: list[Bar],
        base_buy_prices: list[float],
        base_sell_prices: list[float],
        fee_rate: float = 0.001,
        initial_capital: float = 1000.0,
        objective: str = "combined",  # "pnl", "sortino", "smoothness", "combined"
    ):
        self.bars = bars
        self.base_buy_prices = base_buy_prices
        self.base_sell_prices = base_sell_prices
        self.fee_rate = fee_rate
        self.initial_capital = initial_capital
        self.objective = objective
        self._eval_count = 0
        self._best_score = float("inf")
        self.evaluations: list[dict] = []

    def _apply_offsets(self, params: OptimizableParams) -> tuple[list[float], list[float]]:
        buy_prices = [p * (1 - params.buy_offset_pct) for p in self.base_buy_prices]
        sell_prices = [p * (1 + params.sell_offset_pct) for p in self.base_sell_prices]
        # enforce minimum spread
        for i in range(len(buy_prices)):
            if sell_prices[i] <= buy_prices[i] * (1 + params.take_profit_pct):
                sell_prices[i] = buy_prices[i] * (1 + params.take_profit_pct)
        return buy_prices, sell_prices

    def _simulate(self, params: OptimizableParams) -> SimResult:
        buy_p, sell_p = self._apply_offsets(params)
        sim = MarketSimulator(
            fee_rate=self.fee_rate,
            bar_margin_bp=params.bar_margin_bp,
            initial_capital=self.initial_capital,
        )
        return sim.simulate_pair("OPT", self.bars, buy_p, sell_p)

    def _objective(self, x: np.ndarray) -> float:
        params = OptimizableParams.from_array(x)
        result = self._simulate(params)
        self._eval_count += 1

        daily_returns = []
        for i in range(24, len(result.equity_curve), 24):
            prev = result.equity_curve[i - 24]
            curr = result.equity_curve[i]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        sort = sortino_ratio(daily_returns)
        smooth = pnl_smoothness(result.equity_curve)
        ret = result.total_return_pct

        if self.objective == "pnl":
            score = -ret
        elif self.objective == "sortino":
            score = -sort
        elif self.objective == "smoothness":
            score = smooth - ret * 0.1
        else:  # combined
            # maximize: return + sortino, minimize: drawdown + volatility
            score = -(ret * 0.4 + sort * 30.0) + result.max_drawdown_pct * 0.3 + smooth * 100.0

        self.evaluations.append({
            "eval": self._eval_count,
            "score": score,
            "return_pct": ret,
            "sortino": sort,
            "smoothness": smooth,
            "max_dd": result.max_drawdown_pct,
            "trades": result.num_trades,
            "win_rate": result.win_rate,
            "params": params,
        })

        if score < self._best_score:
            self._best_score = score
            log.info("eval=%d score=%.4f ret=%.2f%% sort=%.2f dd=%.1f%% wr=%.1f%% trades=%d",
                     self._eval_count, score, ret, sort,
                     result.max_drawdown_pct, result.win_rate * 100, result.num_trades)

        return score

    def optimize(
        self,
        maxiter: int = 100,
        popsize: int = 15,
        seed: int | None = 42,
        workers: int = 1,
    ) -> OptimizationResult:
        log.info("starting optimization: objective=%s maxiter=%d popsize=%d bars=%d",
                 self.objective, maxiter, popsize, len(self.bars))

        scipy_result = differential_evolution(
            self._objective,
            bounds=OptimizableParams.bounds(),
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            workers=workers,
            tol=1e-6,
            mutation=(0.5, 1.5),
            recombination=0.8,
        )

        best_params = OptimizableParams.from_array(scipy_result.x)
        best_sim = self._simulate(best_params)

        daily_returns = []
        for i in range(24, len(best_sim.equity_curve), 24):
            prev = best_sim.equity_curve[i - 24]
            curr = best_sim.equity_curve[i]
            if prev > 0:
                daily_returns.append((curr - prev) / prev)

        result = OptimizationResult(
            best_params=best_params,
            best_score=float(scipy_result.fun),
            sim_result=best_sim,
            sortino=sortino_ratio(daily_returns),
            smoothness=pnl_smoothness(best_sim.equity_curve),
            scipy_result=scipy_result,
            all_evaluations=self.evaluations,
        )

        log.info("optimization complete: evals=%d ret=%.2f%% sort=%.2f dd=%.1f%% smooth=%.4f",
                 self._eval_count, best_sim.total_return_pct, result.sortino,
                 best_sim.max_drawdown_pct, result.smoothness)
        log.info("best params: tol=%.1fbp buy_off=%.4f sell_off=%.4f pos=%.2f margin=%.1fbp tp=%.4f sl=%.4f",
                 best_params.entry_tolerance_bp, best_params.buy_offset_pct,
                 best_params.sell_offset_pct, best_params.position_size_pct,
                 best_params.bar_margin_bp, best_params.take_profit_pct,
                 best_params.stop_loss_pct)

        return result


class GridSearchOptimizer:
    """Exhaustive grid search over key parameters for comparison."""

    def __init__(
        self,
        bars: list[Bar],
        base_buy_prices: list[float],
        base_sell_prices: list[float],
        fee_rate: float = 0.001,
        initial_capital: float = 1000.0,
    ):
        self.bars = bars
        self.base_buy = base_buy_prices
        self.base_sell = base_sell_prices
        self.fee_rate = fee_rate
        self.capital = initial_capital

    def search(
        self,
        bar_margins: list[float] | None = None,
        buy_offsets: list[float] | None = None,
        sell_offsets: list[float] | None = None,
    ) -> list[dict]:
        bar_margins = bar_margins or [3.0, 5.0, 10.0, 15.0, 25.0]
        buy_offsets = buy_offsets or [0.001, 0.002, 0.003, 0.005, 0.008]
        sell_offsets = sell_offsets or [0.002, 0.003, 0.005, 0.008, 0.012]

        results = []
        total = len(bar_margins) * len(buy_offsets) * len(sell_offsets)
        count = 0

        for margin in bar_margins:
            for bo in buy_offsets:
                for so in sell_offsets:
                    if so <= bo:
                        continue
                    count += 1
                    buy_p = [p * (1 - bo) for p in self.base_buy]
                    sell_p = [p * (1 + so) for p in self.base_sell]
                    sim = MarketSimulator(self.fee_rate, margin, self.capital)
                    r = sim.simulate_pair("GRID", self.bars, buy_p, sell_p)

                    daily_rets = []
                    for i in range(24, len(r.equity_curve), 24):
                        prev = r.equity_curve[i - 24]
                        if prev > 0:
                            daily_rets.append((r.equity_curve[i] - prev) / prev)

                    results.append({
                        "bar_margin_bp": margin,
                        "buy_offset": bo,
                        "sell_offset": so,
                        "return_pct": r.total_return_pct,
                        "sortino": sortino_ratio(daily_rets),
                        "smoothness": pnl_smoothness(r.equity_curve),
                        "max_dd_pct": r.max_drawdown_pct,
                        "trades": r.num_trades,
                        "win_rate": r.win_rate,
                        "fees": r.total_fees,
                    })

        results.sort(key=lambda x: -(x["return_pct"] * 0.4 + x["sortino"] * 30 - x["max_dd_pct"] * 0.3))
        log.info("grid search: %d combos evaluated, best ret=%.2f%% sort=%.2f",
                 len(results), results[0]["return_pct"] if results else 0,
                 results[0]["sortino"] if results else 0)
        return results
