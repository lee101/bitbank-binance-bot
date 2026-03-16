"""Monte Carlo simulator for robustness testing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simulator.data_feed import Bar
from simulator.market_simulator import MarketSimulator, SimResult
from optimizer import sortino_ratio

log = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    n_simulations: int
    returns: list[float]
    sortinos: list[float]
    max_drawdowns: list[float]
    win_rates: list[float]

    @property
    def mean_return(self) -> float:
        return float(np.mean(self.returns)) if self.returns else 0

    @property
    def median_return(self) -> float:
        return float(np.median(self.returns)) if self.returns else 0

    @property
    def std_return(self) -> float:
        return float(np.std(self.returns)) if self.returns else 0

    @property
    def percentile_5(self) -> float:
        return float(np.percentile(self.returns, 5)) if self.returns else 0

    @property
    def percentile_95(self) -> float:
        return float(np.percentile(self.returns, 95)) if self.returns else 0

    @property
    def prob_positive(self) -> float:
        if not self.returns:
            return 0
        return sum(1 for r in self.returns if r > 0) / len(self.returns)

    @property
    def mean_sortino(self) -> float:
        return float(np.mean(self.sortinos)) if self.sortinos else 0

    @property
    def mean_max_dd(self) -> float:
        return float(np.mean(self.max_drawdowns)) if self.max_drawdowns else 0

    def summary(self) -> str:
        return (
            f"Monte Carlo ({self.n_simulations} sims):\n"
            f"  mean return: {self.mean_return:.2f}% (std={self.std_return:.2f}%)\n"
            f"  median return: {self.median_return:.2f}%\n"
            f"  5th/95th pctile: {self.percentile_5:.2f}% / {self.percentile_95:.2f}%\n"
            f"  P(positive): {self.prob_positive:.1%}\n"
            f"  mean sortino: {self.mean_sortino:.3f}\n"
            f"  mean max DD: {self.mean_max_dd:.1f}%"
        )


class MonteCarloSimulator:
    def __init__(
        self,
        bars: list[Bar],
        buy_prices: list[float],
        sell_prices: list[float],
        fee_rate: float = 0.001,
        bar_margin_bp: float = 5.0,
        initial_capital: float = 1000.0,
    ):
        self.bars = bars
        self.buy_prices = buy_prices
        self.sell_prices = sell_prices
        self.fee_rate = fee_rate
        self.bar_margin_bp = bar_margin_bp
        self.capital = initial_capital

    def run(
        self,
        n_simulations: int = 1000,
        price_noise_pct: float = 0.001,
        fee_noise_pct: float = 0.0002,
        block_size: int = 24,
        seed: int | None = 42,
    ) -> MonteCarloResult:
        rng = np.random.RandomState(seed)
        returns = []
        sortinos = []
        drawdowns = []
        win_rates = []

        n_bars = len(self.bars)
        n_blocks = n_bars // block_size

        for sim_i in range(n_simulations):
            # block bootstrap - sample blocks with replacement
            block_indices = rng.randint(0, n_blocks, size=n_blocks)
            sampled_bars = []
            sampled_bp = []
            sampled_sp = []

            for bi in block_indices:
                start = bi * block_size
                end = min(start + block_size, n_bars)
                sampled_bars.extend(self.bars[start:end])

                for j in range(start, end):
                    bp_idx = j if len(self.buy_prices) == n_bars else min(j // 24, len(self.buy_prices) - 1)
                    sp_idx = j if len(self.sell_prices) == n_bars else min(j // 24, len(self.sell_prices) - 1)

                    # add noise to prices
                    bp_noise = 1 + rng.normal(0, price_noise_pct)
                    sp_noise = 1 + rng.normal(0, price_noise_pct)
                    sampled_bp.append(self.buy_prices[bp_idx] * bp_noise)
                    sampled_sp.append(self.sell_prices[sp_idx] * sp_noise)

            # vary fee rate
            sim_fee = self.fee_rate + rng.normal(0, fee_noise_pct)
            sim_fee = max(0.0001, sim_fee)

            # vary bar margin
            sim_margin = self.bar_margin_bp * (1 + rng.normal(0, 0.2))
            sim_margin = max(1.0, sim_margin)

            sim = MarketSimulator(sim_fee, sim_margin, self.capital)
            result = sim.simulate_pair("MC", sampled_bars, sampled_bp, sampled_sp)

            returns.append(result.total_return_pct)
            drawdowns.append(result.max_drawdown_pct)
            win_rates.append(result.win_rate)

            daily_rets = []
            for i in range(24, len(result.equity_curve), 24):
                prev = result.equity_curve[i - 24]
                if prev > 0:
                    daily_rets.append((result.equity_curve[i] - prev) / prev)
            sortinos.append(sortino_ratio(daily_rets))

        log.info("MC done: %d sims, mean_ret=%.2f%% p(+)=%.1f%%",
                 n_simulations, np.mean(returns), sum(1 for r in returns if r > 0) / n_simulations * 100)

        return MonteCarloResult(
            n_simulations=n_simulations,
            returns=returns,
            sortinos=sortinos,
            max_drawdowns=drawdowns,
            win_rates=win_rates,
        )
