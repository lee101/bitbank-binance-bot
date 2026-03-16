"""Multi-pair portfolio simulator with correlation-aware position sizing."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from simulator.data_feed import Bar
from simulator.market_simulator import MarketSimulator, SimResult
from optimizer import sortino_ratio, pnl_smoothness

log = logging.getLogger(__name__)


@dataclass
class PortfolioResult:
    total_return_pct: float
    sortino: float
    smoothness: float
    max_drawdown_pct: float
    per_pair_returns: dict[str, float]
    equity_curve: list[float]
    pair_contributions: dict[str, float]  # % of total pnl from each pair
    correlation_matrix: dict[str, dict[str, float]] | None = None


class PortfolioSimulator:
    def __init__(
        self,
        fee_rate: float = 0.001,
        bar_margin_bp: float = 5.0,
        initial_capital: float = 1000.0,
        max_pair_allocation: float = 0.25,
        max_correlated_exposure: float = 0.5,
    ):
        self.fee_rate = fee_rate
        self.bar_margin_bp = bar_margin_bp
        self.capital = initial_capital
        self.max_pair_alloc = max_pair_allocation
        self.max_corr_exposure = max_correlated_exposure

    def compute_correlations(self, bars_by_pair: dict[str, list[Bar]]) -> dict[str, dict[str, float]]:
        pairs = list(bars_by_pair.keys())
        returns_by_pair = {}
        for pair in pairs:
            closes = [b.close for b in bars_by_pair[pair]]
            rets = []
            for i in range(1, len(closes)):
                if closes[i - 1] > 0:
                    rets.append((closes[i] - closes[i - 1]) / closes[i - 1])
            returns_by_pair[pair] = rets

        # align lengths
        min_len = min(len(r) for r in returns_by_pair.values()) if returns_by_pair else 0
        corr = {}
        for p1 in pairs:
            corr[p1] = {}
            r1 = returns_by_pair[p1][:min_len]
            for p2 in pairs:
                r2 = returns_by_pair[p2][:min_len]
                if min_len > 2:
                    corr[p1][p2] = float(np.corrcoef(r1, r2)[0, 1])
                else:
                    corr[p1][p2] = 0.0
        return corr

    def allocate_capital(
        self,
        pairs: list[str],
        pnl_7d: dict[str, float],
        correlations: dict[str, dict[str, float]] | None = None,
    ) -> dict[str, float]:
        """Allocate capital weighted by 7d PnL, penalized by correlation."""
        if not pairs:
            return {}
        total_pnl = sum(max(0, pnl_7d.get(p, 0)) for p in pairs)
        if total_pnl <= 0:
            return {p: self.capital * self.max_pair_alloc / len(pairs) for p in pairs}

        raw_allocs = {}
        for p in pairs:
            weight = max(0, pnl_7d.get(p, 0)) / total_pnl
            raw_allocs[p] = weight * self.capital

        # apply correlation penalty
        if correlations:
            for p1 in pairs:
                corr_exposure = 0.0
                for p2 in pairs:
                    if p1 != p2 and p2 in correlations.get(p1, {}):
                        corr = abs(correlations[p1][p2])
                        if corr > 0.7:
                            corr_exposure += raw_allocs.get(p2, 0) * corr
                if corr_exposure > self.capital * self.max_corr_exposure:
                    scale = (self.capital * self.max_corr_exposure) / corr_exposure
                    raw_allocs[p1] *= scale

        # cap per-pair
        for p in pairs:
            raw_allocs[p] = min(raw_allocs[p], self.capital * self.max_pair_alloc)

        return raw_allocs

    def simulate(
        self,
        bars_by_pair: dict[str, list[Bar]],
        buy_prices_by_pair: dict[str, list[float]],
        sell_prices_by_pair: dict[str, list[float]],
        pnl_7d: dict[str, float] | None = None,
    ) -> PortfolioResult:
        pairs = list(bars_by_pair.keys())
        if not pairs:
            return PortfolioResult(0, 0, 0, 0, {}, [self.capital], {})

        correlations = self.compute_correlations(bars_by_pair)
        pnl_7d = pnl_7d or {p: 1.0 for p in pairs}
        allocs = self.allocate_capital(pairs, pnl_7d, correlations)

        # simulate each pair independently with its allocation
        pair_results: dict[str, SimResult] = {}
        for pair in pairs:
            if pair not in buy_prices_by_pair or pair not in sell_prices_by_pair:
                continue
            alloc = allocs.get(pair, 0)
            if alloc < 10:
                continue
            sim = MarketSimulator(self.fee_rate, self.bar_margin_bp, alloc)
            result = sim.simulate_pair(pair, bars_by_pair[pair],
                                       buy_prices_by_pair[pair], sell_prices_by_pair[pair])
            pair_results[pair] = result

        # build combined equity curve
        max_len = max((len(r.equity_curve) for r in pair_results.values()), default=0)
        unallocated = self.capital - sum(allocs.get(p, 0) for p in pair_results)
        combined = []
        for i in range(max_len):
            total = unallocated
            for pair, result in pair_results.items():
                if i < len(result.equity_curve):
                    total += result.equity_curve[i]
                elif result.equity_curve:
                    total += result.equity_curve[-1]
            combined.append(total)

        # metrics
        peak = self.capital
        max_dd = 0.0
        for v in combined:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        final_return = (combined[-1] - self.capital) / self.capital * 100 if combined else 0

        daily_rets = []
        for i in range(24, len(combined), 24):
            prev = combined[i - 24]
            if prev > 0:
                daily_rets.append((combined[i] - prev) / prev)

        per_pair_returns = {p: r.total_return_pct for p, r in pair_results.items()}
        total_pnl = sum(r.net_pnl for r in pair_results.values())
        contributions = {}
        for p, r in pair_results.items():
            contributions[p] = r.net_pnl / total_pnl * 100 if total_pnl != 0 else 0

        return PortfolioResult(
            total_return_pct=final_return,
            sortino=sortino_ratio(daily_rets),
            smoothness=pnl_smoothness(combined),
            max_drawdown_pct=max_dd * 100,
            per_pair_returns=per_pair_returns,
            equity_curve=combined,
            pair_contributions=contributions,
            correlation_matrix=correlations,
        )
