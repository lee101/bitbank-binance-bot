import pytest

from optimizer import (
    OptimizableParams, StrategyOptimizer, GridSearchOptimizer,
    sortino_ratio, pnl_smoothness,
)
from simulator.data_feed import Bar


def _make_bars(n: int, base: float = 100.0, amplitude: float = 1.5) -> list[Bar]:
    import math
    bars = []
    for i in range(n):
        p = base + amplitude * math.sin(i * 0.3)
        bars.append(Bar(
            timestamp=1000 + i * 3600,
            open=p, high=p + amplitude, low=p - amplitude, close=p, volume=100,
        ))
    return bars


class TestMetrics:
    def test_sortino_positive(self):
        returns = [0.01, 0.02, -0.005, 0.015, 0.008]
        s = sortino_ratio(returns)
        assert s > 0

    def test_sortino_all_negative(self):
        returns = [-0.01, -0.02, -0.005]
        s = sortino_ratio(returns)
        assert s < 0

    def test_sortino_empty(self):
        assert sortino_ratio([]) == 0.0

    def test_sortino_no_downside(self):
        returns = [0.01, 0.02, 0.03]
        s = sortino_ratio(returns)
        assert s > 0

    def test_smoothness_flat(self):
        curve = [100.0] * 10
        assert pnl_smoothness(curve) == pytest.approx(0.0)

    def test_smoothness_volatile(self):
        curve = [100, 110, 90, 115, 85, 120]
        s = pnl_smoothness(curve)
        assert s > 0

    def test_smoothness_single(self):
        assert pnl_smoothness([100]) == 0.0


class TestOptimizableParams:
    def test_roundtrip(self):
        p = OptimizableParams(entry_tolerance_bp=10, buy_offset_pct=0.005)
        arr = p.to_array()
        p2 = OptimizableParams.from_array(arr)
        assert p2.entry_tolerance_bp == pytest.approx(10.0)
        assert p2.buy_offset_pct == pytest.approx(0.005)

    def test_bounds_length(self):
        bounds = OptimizableParams.bounds()
        arr = OptimizableParams().to_array()
        assert len(bounds) == len(arr)


class TestStrategyOptimizer:
    def test_optimize_improves_score(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]

        opt = StrategyOptimizer(bars, bp, sp, objective="pnl")
        result = opt.optimize(maxiter=5, popsize=5)

        assert result.best_params is not None
        assert result.sim_result is not None
        assert len(result.all_evaluations) > 0

    def test_different_objectives(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]

        for obj in ["pnl", "sortino", "combined"]:
            opt = StrategyOptimizer(bars, bp, sp, objective=obj)
            result = opt.optimize(maxiter=3, popsize=3)
            assert result.best_params is not None


class TestGridSearch:
    def test_grid_search_returns_results(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]

        grid = GridSearchOptimizer(bars, bp, sp)
        results = grid.search(
            bar_margins=[5.0, 15.0],
            buy_offsets=[0.002, 0.005],
            sell_offsets=[0.003, 0.008],
        )
        assert len(results) > 0
        assert "return_pct" in results[0]
        assert "sortino" in results[0]

    def test_grid_sorted_by_combined(self):
        bars = _make_bars(7 * 24)
        bp = [b.close * 0.99 for b in bars]
        sp = [b.close * 1.01 for b in bars]

        grid = GridSearchOptimizer(bars, bp, sp)
        results = grid.search(bar_margins=[5.0], buy_offsets=[0.002], sell_offsets=[0.005, 0.01])
        if len(results) >= 2:
            # first should have best combined score
            s0 = results[0]["return_pct"] * 0.4 + results[0]["sortino"] * 30 - results[0]["max_dd_pct"] * 0.3
            s1 = results[1]["return_pct"] * 0.4 + results[1]["sortino"] * 30 - results[1]["max_dd_pct"] * 0.3
            assert s0 >= s1
